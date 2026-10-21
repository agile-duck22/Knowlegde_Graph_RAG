"""Exceptions for multi-provider LLM adapter."""

from __future__ import annotations


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""


class UnsupportedProviderError(LLMProviderError):
    """Raised when an unrecognized LLM provider name is configured."""


class MissingAPIKeyError(LLMProviderError):
    """Raised when the required API key for the selected provider is not configured."""


class DisallowedModelError(LLMProviderError):
    """Raised when the requested model is not in the configured free-tier allowlist."""


class ProviderAPIError(LLMProviderError):
    """Base exception for upstream provider API invocation failures."""


class ProviderAuthenticationError(ProviderAPIError):
    """Raised when upstream provider rejects credentials (HTTP 401/403)."""


class ProviderRateLimitError(ProviderAPIError):
    """Raised when upstream provider rate limits the request (HTTP 429)."""


class ProviderTimeoutError(ProviderAPIError):
    """Raised when upstream provider call times out."""


class ProviderServerError(ProviderAPIError):
    """Raised when upstream provider encounters an internal error (HTTP 500/503)."""


class ProviderOutputTruncatedError(ProviderAPIError):
    """Raised when upstream provider truncates response due to max completion token limit."""
