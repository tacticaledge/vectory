"""Repository-native provider abstraction for Vectory's public package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from components.provider_config import get_bedrock_region
from components.usage import TokenUsage, usage_from_anthropic, usage_from_openai

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


class BilledCompletionError(ValueError):
    """A billed provider call that produced no usable text; ``usage`` is what was charged."""

    def __init__(self, message: str, usage: TokenUsage | None) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Completion:
    text: str
    usage: TokenUsage | None


class Bot:
    """Small provider-neutral completion boundary used by Vectory workflows."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        model: str,
        client: Any | None = None,
        aws_region: str | None = None,
    ) -> None:
        if provider not in {"openai", "anthropic", "bedrock"}:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.model = model
        self.aws_region = aws_region
        self.client = client or self._create_client(api_key)

    def _create_client(self, api_key: str | None) -> Any:
        if self.provider == "openai":
            if openai is None:
                raise ImportError("openai package not installed")
            return openai.OpenAI(api_key=api_key)
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        if self.provider == "bedrock":
            # Claude on Bedrock speaks the Messages API and signs with the standard AWS credential chain.
            try:
                import boto3  # noqa: F401
            except ImportError as error:
                raise ImportError("Amazon Bedrock needs boto3: pip install 'vectoryai[bedrock]'") from error
            region = self.aws_region or get_bedrock_region()
            if not region:
                raise ValueError("Amazon Bedrock needs an AWS region (set AWS_REGION)")
            return anthropic.AnthropicBedrock(aws_region=region)
        return anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
        json_mode: bool = False,
        cache_prefix: str | None = None,
    ) -> str:
        """Return text from one provider while applying the requested output contract."""
        return self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            json_mode=json_mode,
            cache_prefix=cache_prefix,
        ).text

    def complete_with_usage(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
        json_mode: bool = False,
        cache_prefix: str | None = None,
    ) -> Completion:
        """Return text plus billed token usage.

        ``cache_prefix`` marks the stable leading text of the last user message as
        cacheable on Anthropic. OpenAI caches stable prefixes automatically.
        """
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
            usage = usage_from_openai(getattr(response, "usage", None))
        else:
            # Current Claude models reject non-default sampling parameters, so temperature is not sent.
            system, provider_messages = self._anthropic_messages(messages, cache_prefix)
            output_config = (
                {"format": {"type": "json_schema", "schema": response_schema}}
                if response_schema is not None
                else None
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=provider_messages,
                **({"system": system} if system else {}),
                **({"output_config": output_config} if output_config else {}),
            )
            usage = usage_from_anthropic(getattr(response, "usage", None))
            if getattr(response, "stop_reason", None) == "refusal":
                raise BilledCompletionError("anthropic declined the request (stop_reason=refusal)", usage)
            content = next(
                (block.text for block in response.content if getattr(block, "type", None) == "text"),
                None,
            )
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise BilledCompletionError("anthropic reached max_tokens before completing its response", usage)
        if not isinstance(content, str) or not content.strip():
            raise BilledCompletionError(f"{self.provider} returned an empty completion", usage)
        return Completion(text=content, usage=usage)

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int | None:
        """Return the exact input token count where the provider exposes one, else None.

        The Anthropic SDK's Bedrock client does not support token counting.
        """
        if self.provider != "anthropic":
            return None
        system, provider_messages = self._anthropic_messages(messages, None)
        result = self.client.messages.count_tokens(
            model=self.model,
            messages=provider_messages,
            **({"system": system} if system else {}),
        )
        return result.input_tokens

    @staticmethod
    def _anthropic_messages(
        messages: list[dict[str, str]], cache_prefix: str | None
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        provider_messages: list[dict[str, Any]] = [
            message for message in messages if message["role"] != "system"
        ]
        last = provider_messages[-1] if provider_messages else None
        if (
            cache_prefix
            and last is not None
            and last["role"] == "user"
            and isinstance(last["content"], str)
            and last["content"].startswith(cache_prefix)
            and len(last["content"]) > len(cache_prefix)
        ):
            provider_messages[-1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": last["content"][len(cache_prefix):]},
                ],
            }
        return "\n\n".join(system_parts), provider_messages
