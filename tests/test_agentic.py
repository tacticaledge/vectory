import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.agentic import Bot


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


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
