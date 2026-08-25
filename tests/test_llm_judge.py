import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from components.evaluators.llm_judge import LLMJudgeEvaluator


def make_evaluator(decision_mode="binary"):
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)
    evaluator.decision_mode = decision_mode
    evaluator.custom_prompt = None
    evaluator.criteria = "The response must answer the request without inventing facts."
    evaluator.include_reference = True
    return evaluator


def test_binary_judge_prompt_requires_critique_before_pass_fail_json():
    evaluator = make_evaluator()

    prompt = evaluator._build_prompt("A response", "A reference", "A question")

    assert "Do not use a numeric scale" in prompt
    assert '"critique"' in prompt
    assert '"result"' in prompt
    assert prompt.index("critique") < prompt.index("result")


def test_binary_judge_parses_strict_json():
    evaluator = make_evaluator()

    passed = evaluator._parse_response('{"critique": "Grounded and complete.", "result": "Pass"}')
    failed = evaluator._parse_response('{"critique": "Invents a refund policy.", "result": "Fail"}')

    assert passed["verdict"] == "Pass" and passed["score"] == 1
    assert failed["verdict"] == "Fail" and failed["score"] == 0
    assert failed["reasoning"] == "Invents a refund policy."


@pytest.mark.parametrize(
    "response",
    [
        '{"critique": "No evidence.", "outcome": "bad"}',
        '{"result": "Pass"}',
        '{"critique": "", "result": "Fail"}',
        '{"critique": "Fine.", "result": "good"}',
        'Result: Pass\nCritique: Fine.',
        '```json\n{"critique": "Fine.", "result": "Pass"}\n```',
        '{"critique": "Fine.", "result": "Pass", "extra": true}',
    ],
)
def test_binary_judge_rejects_malformed_or_reinterpreted_output(response):
    evaluator = make_evaluator()

    with pytest.raises(ValueError, match="invalid output"):
        evaluator._parse_response(response)


def test_openai_binary_call_enforces_json_schema():
    evaluator = make_evaluator()
    evaluator.model = "gpt-5-mini"

    class Completions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": '{"critique":"Fine.","result":"Pass"}'})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    completions = Completions()
    evaluator.client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()

    evaluator._call_openai("Judge this")

    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_scale_parser_remains_available_for_legacy_workflows():
    evaluator = make_evaluator("scale_1_5")

    parsed = evaluator._parse_response("Score: 4\nReasoning: Mostly correct")

    assert parsed["score"] == 4
    assert parsed["reasoning"] == "Mostly correct"


def test_custom_prompt_preserves_json_examples_while_replacing_known_placeholders():
    evaluator = make_evaluator()
    evaluator.custom_prompt = 'Example: {"result": "Pass"}\nInput: {input}\nOutput: {output}'

    prompt = evaluator._build_prompt("Actual output", input_text="Actual input")

    assert '{"result": "Pass"}' in prompt
    assert "Input: Actual input" in prompt
    assert "Output: Actual output" in prompt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
