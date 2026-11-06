"""Shared Streamlit controls for choosing an LLM provider and its credentials."""

from __future__ import annotations

from dataclasses import dataclass

from components.model_catalog import BEDROCK_REGIONS, PROVIDER_LABELS
from components.provider_config import get_bedrock_region, get_provider_api_key

PROVIDERS = list(PROVIDER_LABELS)

_SESSION_KEYS = {"openai": "openai_api_key", "anthropic": "anthropic_api_key"}


@dataclass(frozen=True)
class ProviderCredentials:
    """What a page needs to build a ``Bot``: an API key, or an AWS region for Bedrock."""

    api_key: str
    aws_region: str | None
    ready: bool


def provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def get_api_key(st, provider: str) -> str:
    """Get an API key from session state (set on the Settings page) or the environment."""
    session_key = _SESSION_KEYS.get(provider)
    if not session_key:
        return ""
    if st.session_state.get(session_key):
        return st.session_state[session_key]
    return get_provider_api_key(provider)


def default_bedrock_region(st) -> str:
    region = st.session_state.get("aws_region") or get_bedrock_region()
    return region if region in BEDROCK_REGIONS else BEDROCK_REGIONS[0]


def provider_credentials(st, provider: str, key_prefix: str) -> ProviderCredentials:
    """Render the credential control for one provider and return what was configured."""
    if provider == "bedrock":
        default_region = default_bedrock_region(st)
        region = st.selectbox(
            "AWS Region",
            BEDROCK_REGIONS,
            index=BEDROCK_REGIONS.index(default_region),
            key=f"{key_prefix}_aws_region",
            help="Uses your AWS credentials from the environment, profile, or SSO login.",
        )
        st.session_state["aws_region"] = region
        st.caption("Signs requests with your AWS credentials (AWS_PROFILE, SSO, or environment).")
        return ProviderCredentials(api_key="", aws_region=region, ready=True)

    label = provider_label(provider)
    default_key = get_api_key(st, provider)
    if default_key:
        st.success(f"✓ {label} API key loaded")
        return ProviderCredentials(api_key=default_key, aws_region=None, ready=True)
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder=f"Enter your {label} API key",
        key=f"{key_prefix}_{provider}_api_key",
        help="Enter your API key here or configure it in the Settings page",
    )
    if api_key:
        st.session_state[_SESSION_KEYS[provider]] = api_key
    return ProviderCredentials(api_key=api_key, aws_region=None, ready=bool(api_key))
