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


OPENAI_MODELS = [
    ProviderModel(
        provider="openai",
        model_id="gpt-5.1",
        label="GPT-5.1",
        input_per_million=1.25,
        output_per_million=10.00,
        note="Flagship model for coding and agentic tasks.",
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-5-mini",
        label="GPT-5 mini",
        input_per_million=0.25,
        output_per_million=2.00,
        note="Lower latency and cost for well-defined evaluations.",
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-5-nano",
        label="GPT-5 nano",
        input_per_million=0.05,
        output_per_million=0.40,
        note="Fastest low-cost option for lightweight checks.",
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-4.1",
        label="GPT-4.1",
        input_per_million=2.00,
        output_per_million=8.00,
        note="Strong non-reasoning model with broad compatibility.",
    ),
    ProviderModel(
        provider="openai",
        model_id="gpt-4o-mini",
        label="GPT-4o mini",
        input_per_million=0.15,
        output_per_million=0.60,
        note="Legacy-compatible economical fallback.",
    ),
]


ANTHROPIC_MODELS: list[ProviderModel] = []


MODELS_BY_PROVIDER = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
}


DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-5-mini",
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
            }
    return None
