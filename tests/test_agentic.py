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


def test_token_counting_is_exact_for_anthropic_and_unavailable_for_openai():
    bot, messages = anthropic_bot([])
    assert bot.count_input_tokens([{"role": "user", "content": "Judge"}]) == 42
    assert messages.count_kwargs["model"] == "claude-sonnet-5-5"

    openai_bot = Bot(provider="openai", api_key="unused", model="gpt-5-mini", client=object())
    assert openai_bot.count_input_tokens([{"role": "user", "content": "Judge"}]) is None


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
