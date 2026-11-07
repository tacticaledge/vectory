"""Typed provider credentials shared by Streamlit evaluation pages."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    """Provider API keys loaded and validated at one environment boundary."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    aws_region: str | None = None
    aws_default_region: str | None = None

    def api_key(self, provider: str) -> str:
        field_by_provider = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        secret = field_by_provider.get(provider)
        return secret.get_secret_value() if secret else ""

    def bedrock_region(self) -> str | None:
        return self.aws_region or self.aws_default_region


def get_provider_api_key(provider: str) -> str:
    """Return one provider key from the typed environment settings."""
    return ProviderSettings().api_key(provider)


def get_bedrock_region() -> str | None:
    """Return the AWS region from AWS_REGION or AWS_DEFAULT_REGION, if set."""
    return ProviderSettings().bedrock_region()
