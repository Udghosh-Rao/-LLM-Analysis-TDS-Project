from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from shared_store import url_time
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool,
    transcribe_audio, encode_image_to_base64, get_stock_data
)
import os, time
from dotenv import load_dotenv

load_dotenv()

EMAIL         = os.getenv("EMAIL")
SECRET        = os.getenv("SECRET")
RECURSION_LIMIT = 5000
MAX_TOKENS    = 60000

TOOLS = [
    run_code, get_rendered_html, download_file, post_request,
    add_dependencies, ocr_image_tool, transcribe_audio,
    encode_image_to_base64, get_stock_data
]


# ── STATE ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# ── LLM INIT ──────────────────────────────────────────────────────────
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are an autonomous task-solving agent with access to 9 specialized tools.

Your capabilities:
- Render and parse any webpage (including JavaScript-heavy pages)
- Write and execute Python code in a sandboxed environment
- Fetch and analyze live stock/financial data
- Perform OCR on images and transcribe audio files
- Download files, send HTTP requests, install dependencies on the fly

Your operating rules:
1. Load each task page from the given URL.
2. Extract instructions, parameters, and the submission endpoint carefully.
3. Solve the task exactly — do not approximate or hallucinate answers.
4. Submit answers ONLY to the correct endpoint with the required fields.
5. Always include email={EMAIL} and secret={SECRET} in submissions.
6. For Base64 image encoding, ALWAYS use the encode_image_to_base64 tool — never write your own.
7. Never hallucinate URLs, field names, or endpoint paths.
8. Always inspect server responses before proceeding.
9. Follow next URLs until none remain, then output END.
"""


# ── MALFORMED JSON RECOVERY NODE ──────────────────────────────────────
def handle_malformed_node(state: AgentState):
    print("--- MALFORMED JSON DETECTED: Asking agent to retry ---")
    return {
        "messages": [{
            "role": "user",
            "content": (
                "SYSTEM ERROR: Your last tool call was malformed (Invalid JSON). "
                "Please rewrite the tool call and try again. "
                "Ensure all strings have escaped newlines and quotes inside JSON."
            )
        }]
    }


# ── AGENT NODE ────────────────────────────────────────────────────────
def agent_node(state: AgentState):
    cur_time = time.time()
    cur_url  = os.getenv("url")
    prev_time = url_time.get(cur_url)
    offset   = os.getenv("offset", "0")

    # Timeout watchdog: gracefully fail after 180 seconds
    if prev_time is not None:
        diff = cur_time - float(prev_time)
        offset_exceeded = offset != "0" and (cur_time - float(offset)) > 90
        if diff >= 180 or offset_exceeded:
            print(f"Timeout exceeded ({diff:.0f}s) — submitting fallback answer.")
            fail_msg = HumanMessage(content=(
                "You have exceeded the time limit (180s) for this task. "
                "Immediately call post_request and submit a WRONG answer for the CURRENT quiz to move on."
            ))
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}

    # Trim context to avoid token overflow
    trimmed = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm,
    )

    # Safety: re-inject context if trimmed too aggressively
    if not any(msg.type == "human" for msg in trimmed):
        print("WARNING: Context over-trimmed. Injecting URL reminder.")
        trimmed.append(HumanMessage(
            content=f"Context cleared due to length. Continue processing URL: {os.getenv('url', 'Unknown')}"
        ))

    print(f"--- INVOKING AGENT (Context: {len(trimmed)} messages) ---")
    result = llm.invoke(trimmed)
    return {"messages": [result]}


# ── ROUTING LOGIC ─────────────────────────────────────────────────────
def route(state: AgentState):
    last = state["messages"][-1]

    # Check for malformed function call
    if last.response_metadata.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
        return "handle_malformed"

    # Check for valid tool calls
    if getattr(last, "tool_calls", None):
        return "tools"

    # Check for END signal
    content = getattr(last, "content", "")
    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content and content[0].get("text", "").strip() == "END":
        return END

    return "agent"


# ── GRAPH ASSEMBLY ────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("agent",            agent_node)
graph.add_node("tools",            ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START,              "agent")
graph.add_edge("tools",            "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools":            "tools",
        "agent":            "agent",
        "handle_malformed": "handle_malformed",
        END:                END
    }
)

app = graph.compile()


# ── RUNNER ────────────────────────────────────────────────────────────
def run_agent(url: str):
    print(f"\n{'='*60}")
    print(f"Agent starting on: {url}")
    print(f"{'='*60}\n")

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": url}
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("\nAll tasks completed successfully!")
