"""Provider definitions, base URLs, and free-tier model allowlists."""

from __future__ import annotations

from typing import Final

# Supported provider names
PROVIDER_GROQ: Final[str] = "groq"
PROVIDER_GEMINI: Final[str] = "gemini"
PROVIDER_OPENROUTER: Final[str] = "openrouter"
PROVIDER_ANTHROPIC: Final[str] = "anthropic"

SUPPORTED_PROVIDERS: Final[frozenset[str]] = frozenset(
    {
        PROVIDER_GROQ,
        PROVIDER_GEMINI,
        PROVIDER_OPENROUTER,
        PROVIDER_ANTHROPIC,
    }
)

# Canonical provider base URLs
GROQ_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"

PROVIDER_BASE_URLS: Final[dict[str, str]] = {
    PROVIDER_GROQ: GROQ_BASE_URL,
    PROVIDER_GEMINI: GEMINI_BASE_URL,
    PROVIDER_OPENROUTER: OPENROUTER_BASE_URL,
}

# Explicit free-tier model allowlists
GROQ_ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    }
)

GEMINI_ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    }
)

OPENROUTER_ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "nvidia/nemotron-3.5-lightning:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
    }
)

ANTHROPIC_ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
    }
)


def is_model_allowed(provider: str, model: str) -> bool:
    """Validate whether the requested model is permitted for the given provider.

    Rules:
    - OpenRouter: Model ID must explicitly end with ':free'.
    - Groq: Model ID must be in GROQ_ALLOWED_MODELS.
    - Gemini: Model ID must be in GEMINI_ALLOWED_MODELS.
    - Anthropic: Model ID must be in ANTHROPIC_ALLOWED_MODELS.
    - Any unknown provider: Fails closed (False).
    """
    clean_provider = provider.strip().lower()
    clean_model = model.strip()

    if not clean_model:
        return False

    if clean_provider == PROVIDER_OPENROUTER:
        # OpenRouter models MUST end with :free and have a non-empty name before :free
        return clean_model.endswith(":free") and len(clean_model) > len(":free")

    if clean_provider == PROVIDER_GROQ:
        return clean_model in GROQ_ALLOWED_MODELS

    if clean_provider == PROVIDER_GEMINI:
        return clean_model in GEMINI_ALLOWED_MODELS

    if clean_provider == PROVIDER_ANTHROPIC:
        return clean_model in ANTHROPIC_ALLOWED_MODELS

    return False
