import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

import pytest

from components.agentic import BilledCompletionError, Bot
from components.usage import TokenUsage


def test_openai_bot_applies_strict_response_schema():
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": '{"result":"Pass"}'})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    bot = Bot(provider="openai", api_key="unused", model="test-model", client=client)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["result"],
        "properties": {"result": {"type": "string", "enum": ["Pass", "Fail"]}},
    }

    result = bot.complete(
        [{"role": "user", "content": "Judge this"}],
        max_tokens=50,
        temperature=0,
        response_schema=schema,
    )

    assert result == '{"result":"Pass"}'
    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema


class AnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def count_tokens(self, **kwargs):
        self.count_kwargs = kwargs
        return SimpleNamespace(input_tokens=42)


def anthropic_bot(content, stop_reason="end_turn", usage=None):
    response = SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)
    messages = AnthropicMessages(response)
    client = SimpleNamespace(messages=messages)
    return Bot(provider="anthropic", api_key="unused", model="claude-sonnet-5-5", client=client), messages


def test_anthropic_bot_returns_text_after_thinking_and_normalized_usage():
    usage = SimpleNamespace(
        input_tokens=10, cache_read_input_tokens=600, cache_creation_input_tokens=0, output_tokens=80
    )
    bot, messages = anthropic_bot(
        [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="Pass")],
        usage=usage,
    )

    completion = bot.complete_with_usage(
        [{"role": "system", "content": "Be strict."}, {"role": "user", "content": "Judge"}],
        max_tokens=100,
        temperature=0.3,
    )

    assert completion.text == "Pass"
    assert completion.usage == TokenUsage(input_tokens=10, cached_input_tokens=600, output_tokens=80)
    assert "temperature" not in messages.kwargs
    assert messages.kwargs["system"] == "Be strict."


def test_anthropic_response_schema_is_sent_as_output_config():
    bot, messages = anthropic_bot([SimpleNamespace(type="text", text="{}")])
    schema = {"type": "object", "additionalProperties": False, "properties": {}}

    bot.complete([{"role": "user", "content": "Judge"}], max_tokens=10, temperature=0, response_schema=schema)

    assert messages.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


def test_anthropic_request_has_no_output_config_without_schema():
    bot, messages = anthropic_bot([SimpleNamespace(type="text", text="ok")])

    bot.complete([{"role": "user", "content": "Judge"}], max_tokens=10, temperature=0)

    assert "output_config" not in messages.kwargs


def test_anthropic_max_tokens_rejects_partial_text():
    bot, _ = anthropic_bot([SimpleNamespace(type="text", text="Score: 4\nReason")], stop_reason="max_tokens")

    with pytest.raises(BilledCompletionError, match="max_tokens"):
        bot.complete([{"role": "user", "content": "Judge"}], max_tokens=10, temperature=0)


def test_anthropic_cache_prefix_marks_stable_leading_text():
    bot, messages = anthropic_bot([SimpleNamespace(type="text", text="Pass")])

    bot.complete(
        [{"role": "user", "content": "RUBRIC\nrow data"}],
        max_tokens=100,
        temperature=0,
        cache_prefix="RUBRIC\n",
    )

    content = messages.kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "RUBRIC\n", "cache_control": {"type": "ephemeral"}}
    assert content[1] == {"type": "text", "text": "row data"}


def test_anthropic_cache_prefix_is_ignored_when_prompt_does_not_start_with_it():
    bot, messages = anthropic_bot([SimpleNamespace(type="text", text="Pass")])

    bot.complete([{"role": "user", "content": "row data"}], max_tokens=10, temperature=0, cache_prefix="RUBRIC")

    assert messages.kwargs["messages"][0]["content"] == "row data"


@pytest.mark.parametrize(
    ("stop_reason", "message"),
    [("refusal", "declined"), ("max_tokens", "max_tokens")],
)
def test_anthropic_bot_explains_missing_text_and_keeps_billed_usage(stop_reason, message):
    usage = SimpleNamespace(input_tokens=30, output_tokens=10)
    bot, _ = anthropic_bot([SimpleNamespace(type="thinking", thinking="")], stop_reason=stop_reason, usage=usage)

    with pytest.raises(BilledCompletionError, match=message) as raised:
        bot.complete([{"role": "user", "content": "Judge"}], max_tokens=10, temperature=0)

    assert raised.value.usage == TokenUsage(input_tokens=30, output_tokens=10)


def test_bedrock_bot_uses_messages_api_and_has_no_token_counting():
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="Pass")], stop_reason="end_turn", usage=None)
    messages = AnthropicMessages(response)
    bot = Bot(
        provider="bedrock",
        api_key=None,
        model="us.anthropic.claude-haiku-5-5",
        client=SimpleNamespace(messages=messages),
    )

    assert bot.complete([{"role": "user", "content": "Judge"}], max_tokens=10, temperature=0.3) == "Pass"
    assert "temperature" not in messages.kwargs
    assert bot.count_input_tokens([{"role": "user", "content": "Judge"}]) is None


def test_bedrock_client_uses_explicit_region_then_environment(monkeypatch):
    import anthropic

    created = {}
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace())
    monkeypatch.setattr(anthropic, "AnthropicBedrock", lambda **kwargs: created.update(kwargs) or "client")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

    Bot(provider="bedrock", api_key=None, model="m")
    assert created == {"aws_region": "us-west-2"}

    Bot(provider="bedrock", api_key=None, model="m", aws_region="us-east-2")
    assert created == {"aws_region": "us-east-2"}


def test_bedrock_client_requires_a_region(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace())
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    with pytest.raises(ValueError, match="AWS region"):
        Bot(provider="bedrock", api_key=None, model="m")


def test_bedrock_client_explains_missing_boto3(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(ImportError, match=r"vectoryai\[bedrock\]"):
        Bot(provider="bedrock", api_key=None, model="m", aws_region="us-east-1")


def test_token_counting_is_exact_for_anthropic_and_unavailable_for_openai():
    bot, messages = anthropic_bot([])
    assert bot.count_input_tokens([{"role": "user", "content": "Judge"}]) == 42
    assert messages.count_kwargs["model"] == "claude-sonnet-5-5"

    openai_bot = Bot(provider="openai", api_key="unused", model="gpt-5-mini", client=object())
    assert openai_bot.count_input_tokens([{"role": "user", "content": "Judge"}]) is None


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
