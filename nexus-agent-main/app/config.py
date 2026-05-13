import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # App identity
    app_name = "Nexus AI Agent"
    app_version = "4.0.0"
    app_subtitle = "AI-Powered Stock Analysis Assistant"

    # Auth & API
    secret = os.getenv("SECRET", "")
    email = os.getenv("EMAIL", "")

    # Groq LLM config
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")

    # LangGraph config
    recursion_limit = int(os.getenv("RECURSION_LIMIT", "300"))
    max_tokens = int(os.getenv("MAX_TOKENS", "24000"))

    # Cache config (seconds)
    cache_ttl = int(os.getenv("CACHE_TTL", "300"))
    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    # Demo mode: show sample data when API unavailable
    demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"


settings = Settings()