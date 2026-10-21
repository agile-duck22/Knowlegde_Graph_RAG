"""Unified multi-provider LLM adapter supporting Groq, Gemini, OpenRouter, and Anthropic.

Design & Invariants:
---------------------
- Zero external package dependencies beyond existing `openai` and `anthropic` SDKs.
- Supports Groq, Google Gemini, and OpenRouter via their official OpenAI-compatible endpoints.
- Strict Free-Tier Enforcement: Validates all requested model IDs against explicit allowlists.
  OpenRouter models must end with ':free'. Disallowed models fail closed with DisallowedModelError.
- Explicit credential verification: Fails closed with MissingAPIKeyError if the selected provider's
  API key is missing, empty, or set to a placeholder. Never silently falls back to Anthropic or paid APIs.
- Backward compatibility: Implements client.messages.create(...) to match existing Anthropic
  contract used across extraction, routing, and synthesis.
- Clean dependency injection: Accepts pre-configured or mocked client instances for offline testing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anthropic
import openai

from src.config import Settings, get_settings
from src.llm.exceptions import (
    DisallowedModelError,
    MissingAPIKeyError,
    ProviderAuthenticationError,
    ProviderOutputTruncatedError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    UnsupportedProviderError,
)
from src.llm.models import (
    PROVIDER_ANTHROPIC,
    PROVIDER_BASE_URLS,
    PROVIDER_OPENROUTER,
    SUPPORTED_PROVIDERS,
    is_model_allowed,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalized Response Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResponseContent:
    """Represents a text content block in an LLM response."""

    text: str


@dataclass(frozen=True)
class LLMResponseUsage:
    """Token usage metrics matching Anthropic/OpenAI usage contracts."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """Unified LLM completion response matching the Anthropic response shape."""

    content: list[LLMResponseContent] = field(default_factory=list)
    usage: LLMResponseUsage = field(default_factory=LLMResponseUsage)
    model: str = ""
    provider: str = ""

    @property
    def text(self) -> str:
        """Convenience property returning the primary response text."""
        if self.content:
            return self.content[0].text
        return ""


# ---------------------------------------------------------------------------
# Messages Endpoint Wrapper (Anthropic Contract Compatibility)
# ---------------------------------------------------------------------------


class MessagesEndpoint:
    """Wrapper exposing client.messages.create() for Anthropic SDK compatibility."""

    def __init__(self, adapter: MultiProviderClient) -> None:
        self._adapter = adapter

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a chat completion request through the configured provider adapter."""
        return self._adapter.execute_chat(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Multi-Provider Client Adapter
# ---------------------------------------------------------------------------


class MultiProviderClient:
    """Unified multi-provider client routing requests to Groq, Gemini, OpenRouter, or Anthropic."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: str | None = None,
        openai_client: openai.OpenAI | Any | None = None,
        anthropic_client: anthropic.Anthropic | Any | None = None,
        client: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = (provider or self.settings.llm_provider).strip().lower()

        if self.provider not in SUPPORTED_PROVIDERS:
            raise UnsupportedProviderError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"Allowed providers: {sorted(SUPPORTED_PROVIDERS)}"
            )

        if client is not None:
            if self.provider == PROVIDER_ANTHROPIC:
                anthropic_client = anthropic_client or client
            else:
                openai_client = openai_client or client

        self._injected_openai_client = openai_client
        self._injected_anthropic_client = anthropic_client
        self._timeout_seconds = timeout_seconds
        self._openai_client: openai.OpenAI | None = None
        self._anthropic_client: anthropic.Anthropic | None = None

        # Expose .messages.create for exact Anthropic SDK contract compatibility
        self.messages = MessagesEndpoint(self)

    def get_underlying_client(self) -> Any:
        """Return the underlying active SDK client instance."""
        if self.provider == PROVIDER_ANTHROPIC:
            return self._get_anthropic_client()
        return self._get_openai_client()

    # -----------------------------------------------------------------------
    # Lazy Upstream Client Factories
    # -----------------------------------------------------------------------

    def _get_api_key(self) -> str:
        """Retrieve and validate the API key for the active provider."""
        key_secret = self.settings.get_api_key_for_provider(self.provider)
        if key_secret is None:
            raise MissingAPIKeyError(
                f"API key for provider '{self.provider}' ({self.provider.upper()}_API_KEY) "
                f"is not configured in settings or environment."
            )

        key_value = key_secret.get_secret_value().strip()
        if not key_value or key_value.startswith("your_") or key_value == "placeholder":
            raise MissingAPIKeyError(
                f"API key for provider '{self.provider}' ({self.provider.upper()}_API_KEY) "
                f"is unset or contains a placeholder value."
            )
        return key_value

    def _get_openai_client(self) -> openai.OpenAI:
        """Lazily initialize or return the injected OpenAI-compatible client."""
        if self._injected_openai_client is not None:
            return self._injected_openai_client

        if self._openai_client is not None:
            return self._openai_client

        base_url = PROVIDER_BASE_URLS.get(self.provider)
        if not base_url:
            raise UnsupportedProviderError(f"No base URL configured for provider '{self.provider}'")

        api_key = self._get_api_key()
        default_headers: dict[str, str] | None = None
        if self.provider == PROVIDER_OPENROUTER:
            default_headers = {
                "HTTP-Referer": "https://github.com/kgr",
                "X-Title": "Knowledge Graph RAG",
            }

        timeout = self._timeout_seconds or self.settings.request_timeout_seconds
        self._openai_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            default_headers=default_headers,
        )
        return self._openai_client

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        """Lazily initialize or return the injected Anthropic client."""
        if self._injected_anthropic_client is not None:
            return self._injected_anthropic_client

        if self._anthropic_client is not None:
            return self._anthropic_client

        api_key = self._get_api_key()
        timeout = self._timeout_seconds or self.settings.request_timeout_seconds
        self._anthropic_client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout,
        )
        return self._anthropic_client

    # -----------------------------------------------------------------------
    # Core Execution
    # -----------------------------------------------------------------------

    def execute_chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Dispatch a chat request to the active provider with allowlist enforcement."""
        # 1. Enforce Model Allowlist (Fail closed)
        if not is_model_allowed(self.provider, model):
            if self.provider == PROVIDER_OPENROUTER:
                raise DisallowedModelError(
                    f"Model '{model}' is not allowed for OpenRouter. "
                    f"OpenRouter models must end with ':free' (e.g. 'meta-llama/llama-3.3-70b-instruct:free')."
                )
            raise DisallowedModelError(
                f"Model '{model}' is not in the allowed free-tier model list for provider '{self.provider}'."
            )

        # 2. Dispatch based on provider architecture
        if self.provider == PROVIDER_ANTHROPIC:
            return self._execute_anthropic(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
                **kwargs,
            )

        # Groq, Gemini, OpenRouter dispatch through OpenAI-compatible chat completions
        return self._execute_openai_compatible(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            **kwargs,
        )

    def _execute_anthropic(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute request via Anthropic Messages API."""
        client = self._get_anthropic_client()
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens or self.settings.max_completion_tokens,
            "timeout": timeout or self.settings.request_timeout_seconds,
        }
        if system:
            call_kwargs["system"] = system
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        call_kwargs.update(kwargs)

        resp = client.messages.create(**call_kwargs)

        content_blocks = [
            LLMResponseContent(text=getattr(block, "text", ""))
            for block in getattr(resp, "content", [])
            if hasattr(block, "text")
        ]
        input_tokens = getattr(getattr(resp, "usage", None), "input_tokens", 0)
        output_tokens = getattr(getattr(resp, "usage", None), "output_tokens", 0)

        return LLMResponse(
            content=content_blocks,
            usage=LLMResponseUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=model,
            provider=self.provider,
        )

    def _execute_openai_compatible(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute request via OpenAI-compatible Chat Completions API."""
        client = self._get_openai_client()

        # Format messages with optional system prompt
        formatted_messages: list[dict[str, str]] = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            formatted_messages.append({"role": role, "content": content})

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.settings.max_completion_tokens,
            "timeout": timeout or self.settings.request_timeout_seconds,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        call_kwargs.update(kwargs)

        try:
            resp = client.chat.completions.create(**call_kwargs)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            logger.error("Provider authentication error on '%s': %s", self.provider, exc)
            raise ProviderAuthenticationError(
                f"Authentication failed for provider '{self.provider}': {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            logger.warning("Provider rate limit error on '%s': %s", self.provider, exc)
            raise ProviderRateLimitError(
                f"Rate limit exceeded on provider '{self.provider}': {exc}"
            ) from exc
        except openai.APITimeoutError as exc:
            logger.warning("Provider timeout on '%s': %s", self.provider, exc)
            raise ProviderTimeoutError(
                f"Timeout calling provider '{self.provider}': {exc}"
            ) from exc
        except (openai.InternalServerError, openai.APIConnectionError) as exc:
            logger.warning("Provider server/connection error on '%s': %s", self.provider, exc)
            raise ProviderServerError(
                f"Server/connection error on provider '{self.provider}': {exc}"
            ) from exc

        choices = getattr(resp, "choices", [])
        if not choices:
            return LLMResponse(
                content=[],
                usage=LLMResponseUsage(),
                model=model,
                provider=self.provider,
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            max_tokens_val = call_kwargs.get("max_tokens")
            raise ProviderOutputTruncatedError(
                f"Response truncated by provider '{self.provider}' due to token limit (finish_reason='length'). "
                f"Consider increasing max_completion_tokens (current: {max_tokens_val})."
            )

        text_content = ""
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            text_content = choice.message.content or ""

        usage_obj = getattr(resp, "usage", None)
        input_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
        output_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0

        return LLMResponse(
            content=[LLMResponseContent(text=text_content)],
            usage=LLMResponseUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=model,
            provider=self.provider,
        )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_llm_client(
    settings: Settings | None = None,
    *,
    provider: str | None = None,
    openai_client: openai.OpenAI | Any | None = None,
    anthropic_client: anthropic.Anthropic | Any | None = None,
    client: Any | None = None,
    timeout_seconds: float | None = None,
) -> MultiProviderClient:
    """Create a managed MultiProviderClient for the configured or requested provider."""
    cfg = settings or get_settings()
    return MultiProviderClient(
        settings=cfg,
        provider=provider,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        client=client,
        timeout_seconds=timeout_seconds,
    )
