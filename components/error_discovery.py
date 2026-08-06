"""Helpers for AI-assisted error discovery workflows."""

import json
from typing import Any

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


TAXONOMY_SYSTEM_PROMPT = """You help evaluation teams build failure-mode taxonomies from human review notes.
Use only the evidence in the notes. Prefer a small, coherent set of categories over many narrow labels."""


def build_taxonomy_prompt(
    open_codes: list[str],
    existing_failure_modes: dict[str, dict[str, Any]] | None = None,
    max_codes: int = 40,
) -> str:
    """Build a prompt for clustering open-code notes into failure modes."""
    trimmed_codes = [code.strip() for code in open_codes if code and code.strip()][:max_codes]
    existing_failure_modes = existing_failure_modes or {}

    existing_section = "None yet."
    if existing_failure_modes:
        existing_section = "\n".join(
            f"- {name}: {details.get('description', '')}"
            for name, details in existing_failure_modes.items()
        )

    notes_section = "\n".join(f"{i + 1}. {code}" for i, code in enumerate(trimmed_codes))

    return f"""Cluster these human review notes into a concise failure-mode taxonomy.

Existing failure modes:
{existing_section}

Open-code notes:
{notes_section}

Return only valid JSON with this shape:
{{
  "suggestions": [
    {{
      "name": "Short failure mode name",
      "description": "One sentence definition of when this failure occurs.",
      "examples": ["Exact or lightly shortened notes that support this mode"],
      "rationale": "Why these notes belong together"
    }}
  ]
}}

Rules:
- Suggest at most 8 failure modes.
- Do not duplicate existing failure modes; suggest refinements only when materially clearer.
- Use names that are specific enough for reviewers to apply consistently.
- Do not invent examples or failure modes not supported by the notes."""


def _extract_json_payload(text: str) -> Any:
    """Extract the first JSON object or array from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for i, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[i:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object or array found in taxonomy suggestion response")


def parse_taxonomy_suggestions(response_text: str) -> list[dict[str, Any]]:
    """Parse and normalize taxonomy suggestions returned by an LLM."""
    payload = _extract_json_payload(response_text)
    raw_suggestions = payload.get("suggestions", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_suggestions, list):
        raise ValueError("Taxonomy suggestion response must contain a suggestions list")

    suggestions = []
    seen_names = set()
    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or item.get("title") or item.get("failure_mode") or "").strip()
        description = str(item.get("description") or item.get("definition") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        examples = item.get("examples") or []

        if isinstance(examples, str):
            examples = [examples]
        examples = [str(example).strip() for example in examples if str(example).strip()]

        if not name or not description:
            continue

        dedupe_key = name.casefold()
        if dedupe_key in seen_names:
            continue
        seen_names.add(dedupe_key)

        suggestions.append(
            {
                "name": name[:80],
                "description": description,
                "examples": examples[:5],
                "rationale": rationale,
            }
        )

    return suggestions


def generate_taxonomy_suggestions(
    open_codes: list[str],
    existing_failure_modes: dict[str, dict[str, Any]] | None,
    provider: str,
    api_key: str,
    model: str,
) -> list[dict[str, Any]]:
    """Call an LLM provider and return normalized taxonomy suggestions."""
    prompt = build_taxonomy_prompt(open_codes, existing_failure_modes)

    if provider == "openai":
        if openai is None:
            raise ImportError("openai package not installed")
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content
    elif provider == "anthropic":
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1200,
            temperature=0.2,
            system=TAXONOMY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return parse_taxonomy_suggestions(response_text)
