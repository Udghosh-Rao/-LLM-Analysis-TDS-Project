from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Chat"])

SYSTEM_PROMPT = """You are Nexus AI, a friendly and knowledgeable stock analysis assistant.
Your job is to explain financial concepts and stock performance in simple, human language.
NEVER give specific investment advice. Always remind users it is for educational purposes.

Key traits:
- Use simple language that non-technical people understand
- Be concise but informative (2-4 sentences per response)
- Reference actual data when context is provided
- Never fabricate numbers or data you don't have
- If unsure, say so politely"""

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        import yfinance as yf
        from app.services.groq_client import get_groq_llm

        llm = get_groq_llm(temperature=0.3)
        context_text = ""

        if request.context:
            ctx = request.context
            if "price_summary" in ctx:
                ps = ctx["price_summary"]
                context_text += f"Current price: ${ps.get('current_price', 'N/A')}. "
                context_text += f"Daily change: {ps.get('daily_change_pct', 'N/A')}%\n"
            if "indicators" in ctx:
                ind = ctx["indicators"]
                context_text += f"RSI: {ind.get('rsi_14', 'N/A')}, "
                context_text += f"MA20: {ind.get('ma_20', 'N/A')}, "
                context_text += f"MA50: {ind.get('ma_50', 'N/A')}\n"
            if "ml_prediction" in ctx:
                ml = ctx["ml_prediction"]
                context_text += f"ML anomaly score: {ml.get('anomaly_score', 'N/A')}, "
                context_text += f"label: {ml.get('label', 'N/A')}\n"
            if "sentiment" in ctx:
                sent = ctx["sentiment"]
                context_text += f"News sentiment: {sent.get('verdict', 'N/A')} "
                context_text += f"({sent.get('positive_pct', 0)}% positive)\n"
            if "recommendation" in ctx:
                rec = ctx["recommendation"]
                context_text += f"Algorithmic signal: {rec.get('label', 'N/A')} "
                context_text += f"with {rec.get('confidence', 0)}% confidence\n"
        elif request.ticker:
            try:
                info = yf.Ticker(request.ticker).info
                ticker_name = info.get("shortName", request.ticker)
                current = info.get("currentPrice", info.get("previousClose", "N/A"))
                context_text = f"Ticker: {request.ticker} ({ticker_name}). "
                context_text += f"Recent price: ${current}\n"
            except Exception:
                context_text = f"Ticker: {request.ticker}\n"

        user_prompt = request.message.strip()
        full_prompt = f"""{SYSTEM_PROMPT}

Market Context:
{context_text}

User Question: {user_prompt}

Answer:"""

        if llm is None:
            return ChatResponse(
                response=f"LLM not available. Your question was: '{user_prompt}'. "
                        f"Please set GROQ_API_KEY to enable AI responses.",
                ticker_mentioned=request.ticker,
            )

        response = llm.invoke(full_prompt)
        content = response.content if hasattr(response, "content") else str(response)

        return ChatResponse(
            response=content.strip(),
            ticker_mentioned=request.ticker,
            sources=["Groq LLM (llama-3.3-70b-versatile)"] if llm else [],
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))