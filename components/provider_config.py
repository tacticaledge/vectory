"""Typed provider credentials shared by Streamlit evaluation pages."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, SecretStr


class ProviderSettings(BaseModel):
    """Provider API keys loaded and validated at one environment boundary."""

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ProviderSettings":
        source = os.environ if environ is None else environ
        return cls(
            openai_api_key=source.get("OPENAI_API_KEY") or None,
            anthropic_api_key=source.get("ANTHROPIC_API_KEY") or None,
        )

    def api_key(self, provider: str) -> str:
        field_by_provider = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        secret = field_by_provider.get(provider)
        return secret.get_secret_value() if secret else ""


def get_provider_api_key(provider: str) -> str:
    """Return one provider key from the typed environment settings."""
    return ProviderSettings.from_environment().api_key(provider)
