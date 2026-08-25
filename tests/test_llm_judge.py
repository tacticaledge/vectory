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


def test_binary_judge_parses_json_and_fenced_json():
    evaluator = make_evaluator()

    passed = evaluator._parse_response('{"critique": "Grounded and complete.", "result": "Pass"}')
    failed = evaluator._parse_response(
        '```json\n{"critique": "Invents a refund policy.", "result": "Fail"}\n```'
    )

    assert passed["verdict"] == "Pass" and passed["score"] == 1
    assert failed["verdict"] == "Fail" and failed["score"] == 0
    assert failed["reasoning"] == "Invents a refund policy."


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
