import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd

from components.agentic import Completion
from components.evaluators.llm_judge import (
    LLMJudgeEvaluator,
    estimate_cost,
    get_result_column,
    summarize_usage,
)
from components.usage import TokenUsage


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

    class FakeBot:
        def __init__(self):
            self.kwargs = None

        def complete_with_usage(self, messages, **kwargs):
            self.kwargs = kwargs
            self.messages = messages
            return Completion(text='{"critique":"Fine.","result":"Pass"}', usage=TokenUsage())

    bot = FakeBot()
    evaluator.client = bot

    evaluator._call_openai("Judge this")

    response_schema = bot.kwargs["response_schema"]
    assert response_schema["additionalProperties"] is False
    assert response_schema["properties"]["result"]["enum"] == ["Pass", "Fail"]


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


def test_all_error_batch_has_safe_empty_verdict_column():
    results = pd.DataFrame(
        [{"score": None, "reasoning": None, "error": "invalid output"}]
    )

    verdicts = get_result_column(results, "verdict")

    assert len(verdicts) == 1
    assert verdicts.isna().all()


class UsageBot:
    def __init__(self, text, usage):
        self.text = text
        self.usage = usage
        self.calls = []

    def complete_with_usage(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return Completion(text=self.text, usage=self.usage)

    def count_input_tokens(self, messages):
        return len(messages[0]["content"])


def make_usage_evaluator(text, provider="openai", model="gpt-5-mini"):
    evaluator = make_evaluator()
    evaluator.provider = provider
    evaluator.model = model
    evaluator.client = UsageBot(text, TokenUsage(input_tokens=1000, cached_input_tokens=0, output_tokens=200))
    return evaluator


def test_evaluate_single_records_token_usage_and_cost():
    evaluator = make_usage_evaluator('{"critique":"Fine.","result":"Pass"}')

    result = evaluator.evaluate_single("Answer", input_text="Question")

    assert result["verdict"] == "Pass"
    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 200
    # gpt-5-mini: 1000 * $0.25/M + 200 * $2.00/M
    assert result["cost_usd"] == pytest.approx(0.00065)


def test_unparseable_response_still_records_billed_usage():
    evaluator = make_usage_evaluator("not json")

    result = evaluator.evaluate_single("Answer")

    assert "invalid output" in result["error"]
    assert result["input_tokens"] == 1000
    assert result["cost_usd"] is not None


def test_anthropic_cache_prefix_is_custom_prompt_text_before_first_placeholder():
    evaluator = make_usage_evaluator('{"critique":"Fine.","result":"Pass"}', "anthropic", "claude-sonnet-5-5")
    evaluator.custom_prompt = "Rubric and expert examples.\nInput: {input}\nOutput: {output}"

    evaluator.evaluate_single("Answer", input_text="Question")

    messages, kwargs = evaluator.client.calls[0]
    assert kwargs["cache_prefix"] == "Rubric and expert examples.\nInput: "
    assert messages[0]["content"].startswith(kwargs["cache_prefix"])


def test_cache_prefix_is_not_used_for_openai_or_default_templates():
    evaluator = make_usage_evaluator("{}", "anthropic", "claude-sonnet-5-5")
    assert evaluator._cache_prefix() is None
    evaluator.provider = "openai"
    evaluator.custom_prompt = "Rubric {output}"
    assert evaluator._cache_prefix() is None


def test_summarize_usage_totals_cost_and_cache_hit_rate():
    results = pd.DataFrame(
        [
            {"input_tokens": 100, "cached_input_tokens": 0, "cache_write_tokens": 900,
             "output_tokens": 50, "reasoning_tokens": 10, "cost_usd": 0.01},
            {"input_tokens": 100, "cached_input_tokens": 900, "cache_write_tokens": 0,
             "output_tokens": 50, "reasoning_tokens": 0, "cost_usd": 0.002},
            {"score": None, "error": "network"},
        ]
    )

    summary = summarize_usage(results)

    assert summary["billed_calls"] == 2
    assert summary["cost_usd"] == pytest.approx(0.012)
    assert summary["output_tokens"] == 100
    assert summary["cache_hit_rate"] == pytest.approx(0.45)


def test_summarize_usage_reports_unknown_cost_when_any_call_is_unpriced():
    results = pd.DataFrame(
        [
            {"input_tokens": 10, "cached_input_tokens": 0, "cache_write_tokens": 0,
             "output_tokens": 5, "reasoning_tokens": 0, "cost_usd": None},
        ]
    )

    assert summarize_usage(results)["cost_usd"] is None
    assert summarize_usage(pd.DataFrame([{"score": 1}])) is None


def test_measure_input_tokens_averages_a_sample_of_rendered_prompts():
    evaluator = make_usage_evaluator("{}", "anthropic", "claude-sonnet-5-5")
    evaluator.custom_prompt = "{output}"

    mean = evaluator.measure_input_tokens(["aa", "aaaa", "aaaaaa"], sample_size=2)

    assert mean == pytest.approx(3.0)


def test_estimate_cost_uses_measured_input_tokens():
    estimate = estimate_cost(10, "anthropic", "claude-sonnet-5-5", input_tokens_per_sample=1000)

    assert estimate["input_tokens"] == 10_000
    assert estimate["output_tokens"] == 3_000
    # 10_000 * $2/M + 3_000 * $10/M
    assert estimate["estimated_cost"] == "$0.0500"


def test_anthropic_judge_defaults_to_an_anthropic_model():
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)
    evaluator.provider = "anthropic"

    assert evaluator._default_model().startswith("claude-")


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
