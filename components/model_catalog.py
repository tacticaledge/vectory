"""Central model catalog for provider-backed evaluation flows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderModel:
    provider: str
    model_id: str
    label: str
    input_per_million: float
    output_per_million: float
    note: str
    # None bills cache traffic at the input rate. OpenAI caches automatically with no write premium.
    cached_input_per_million: float | None = None
    cache_write_per_million: float | None = None


OPENAI_MODELS = [
    ProviderModel(
        provider="openai",
        model_id="gpt-5.1",
        label="GPT-5.1",
        input_per_million=1.25,
        output_per_million=10.00,
        note="Flagship model for coding and agentic tasks.",
        cached_input_per_million=0.125,
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-5-mini",
        label="GPT-5 mini",
        input_per_million=0.25,
        output_per_million=2.00,
        note="Lower latency and cost for well-defined evaluations.",
        cached_input_per_million=0.025,
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-5-nano",
        label="GPT-5 nano",
        input_per_million=0.05,
        output_per_million=0.40,
        note="Fastest low-cost option for lightweight checks.",
        cached_input_per_million=0.005,
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-4.1",
        label="GPT-4.1",
        input_per_million=2.00,
        output_per_million=8.00,
        note="Strong non-reasoning model with broad compatibility.",
        cached_input_per_million=0.50,
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-4o-mini",
        label="GPT-4o mini",
        input_per_million=0.15,
        output_per_million=0.60,
        note="Legacy-compatible economical fallback.",
        cached_input_per_million=0.075,
    ),
]


# Anthropic first-party list prices. Cache writes use the default 5-minute TTL (1.25x input).
ANTHROPIC_MODELS = [
    ProviderModel(
        provider="anthropic",
        model_id="claude-opus-5-5",
        label="Claude Opus 5.5",
        input_per_million=4.00,
        output_per_million=20.00,
        note="Most capable Opus model for demanding judgments.",
        cached_input_per_million=0.20,
        cache_write_per_million=5.00,
    ),
    ProviderModel(
        provider="anthropic",
        model_id="claude-sonnet-5-5",
        label="Claude Sonnet 5.5",
        input_per_million=2.00,
        output_per_million=10.00,
        note="Balanced speed and capability for everyday evaluations.",
        cached_input_per_million=0.20,
        cache_write_per_million=2.50,
    ),
    ProviderModel(
        provider="anthropic",
        model_id="claude-haiku-5-5",
        label="Claude Haiku 5.5",
        input_per_million=0.10,
        output_per_million=0.50,
        note="Fastest low-cost option; prices apply to prompts up to 100K tokens.",
        cached_input_per_million=0.01,
        cache_write_per_million=0.125,
    ),
]


MODELS_BY_PROVIDER = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
}


DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-opus-5-5",
}


def get_provider_models(provider: str) -> list[ProviderModel]:
    return MODELS_BY_PROVIDER.get(provider, [])


def get_model_ids(provider: str) -> list[str]:
    return [model.model_id for model in get_provider_models(provider)]


def get_model_label(provider: str, model_id: str) -> str:
    for model in get_provider_models(provider):
        if model.model_id == model_id:
            return model.label
    return model_id


def get_model_pricing(provider: str, model_id: str) -> dict[str, float] | None:
    for model in get_provider_models(provider):
        if model.model_id == model_id:
            return {
                "input": model.input_per_million,
                "output": model.output_per_million,
                "cached_input": model.cached_input_per_million,
                "cache_write": model.cache_write_per_million,
            }
    return None
