"""Repository-native provider abstraction for Vectory's public package."""

from __future__ import annotations

from typing import Any

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


class Bot:
    """Small provider-neutral completion boundary used by Vectory workflows."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        model: str,
        client: Any | None = None,
    ) -> None:
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.model = model
        self.client = client or self._create_client(api_key)

    def _create_client(self, api_key: str | None) -> Any:
        if self.provider == "openai":
            if openai is None:
                raise ImportError("openai package not installed")
            return openai.OpenAI(api_key=api_key)
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        return anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return text from one provider while applying the requested output contract."""
        if self.provider == "openai":
            response_format = None
            if response_schema is not None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "vectory_structured_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            elif json_mode:
                response_format = {"type": "json_object"}
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **({"response_format": response_format} if response_format else {}),
            )
            content = response.choices[0].message.content
        else:
            system_parts = [message["content"] for message in messages if message["role"] == "system"]
            provider_messages = [message for message in messages if message["role"] != "system"]
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=provider_messages,
                **({"system": "\n\n".join(system_parts)} if system_parts else {}),
            )
            content = response.content[0].text
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{self.provider} returned an empty completion")
        return content
