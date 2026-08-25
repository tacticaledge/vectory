import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from .base import BaseEvaluator
from components.model_catalog import DEFAULT_MODEL_BY_PROVIDER, get_model_pricing

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


DEFAULT_EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of an LLM response.

{context}

## Response to Evaluate
{output}

{reference_section}

## Evaluation Criteria
{criteria}

## Instructions
Provide your evaluation in the following format:
1. Score: [1-5]
2. Reasoning: [Brief explanation]

Be objective and consistent in your scoring."""


BINARY_EVALUATION_PROMPT = """You are an expert evaluator making one clear product-quality decision.

{context}

## Response to Evaluate
{output}

{reference_section}

## Passing Requirement
{criteria}

## Instructions
First write a specific critique grounded in the supplied input, response, reference, and context.
Then decide whether the response satisfies the requirement overall.
Return a JSON object with exactly two fields, for example:
{{"critique": "detailed explanation", "result": "Pass"}}
The result value must be exactly "Pass" or "Fail".
Do not use a numeric scale."""


CRITERIA_TEMPLATES = {
    "accuracy": "The response is factually accurate and correct.",
    "relevance": "The response is relevant and on-topic for the input question or request.",
    "coherence": "The response has clear, logical, and coherent reasoning.",
    "completeness": "The response addresses every material part of the input.",
    "helpfulness": "The response achieves the user's desired outcome and is useful to them.",
    "safety": "The response is safe, appropriate, and free of materially harmful content.",
    "custom": "",
}


class BinaryJudgeResponse(BaseModel):
    """Strict response contract for release-relevant binary judge decisions."""

    model_config = ConfigDict(extra="forbid")

    critique: str = Field(min_length=1)
    result: Literal["Pass", "Fail"]


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluates outputs using an LLM as a judge."""

    @property
    def name(self) -> str:
        return "LLM-as-Judge"

    @property
    def requires_reference(self) -> bool:
        return False

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = None,
        model: str = None,
        criteria: str = "helpfulness",
        custom_criteria: str = None,
        custom_prompt: str = None,
        include_reference: bool = True,
        rate_limit_delay: float = 0.5,
        decision_mode: str = "binary",
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self.criteria = CRITERIA_TEMPLATES.get(criteria, criteria)
        if criteria == "custom" and custom_criteria:
            self.criteria = custom_criteria
        self.custom_prompt = custom_prompt
        self.include_reference = include_reference
        self.rate_limit_delay = rate_limit_delay
        if decision_mode not in {"binary", "scale_1_5"}:
            raise ValueError("decision_mode must be binary or scale_1_5")
        self.decision_mode = decision_mode
        self.client = self._init_client()

    def _default_model(self) -> str:
        return DEFAULT_MODEL_BY_PROVIDER.get(self.provider, DEFAULT_MODEL_BY_PROVIDER["openai"])

    def _init_client(self):
        if self.provider == "openai":
            if openai is None:
                raise ImportError("openai package not installed")
            return openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic package not installed")
            return anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _build_prompt(self, output: str, reference: str = None, input_text: str = None) -> str:
        if self.custom_prompt:
            prompt = self.custom_prompt
            for placeholder, value in (
                ("{output}", output),
                ("{reference}", reference or ""),
                ("{input}", input_text or ""),
            ):
                prompt = prompt.replace(placeholder, value)
            return prompt

        context = ""
        if input_text:
            context = f"## Original Input/Prompt\n{input_text}"

        reference_section = ""
        if reference and self.include_reference:
            reference_section = f"## Reference/Expected Response\n{reference}"

        prompt_template = (
            BINARY_EVALUATION_PROMPT
            if self.decision_mode == "binary"
            else DEFAULT_EVALUATION_PROMPT
        )
        return prompt_template.format(
            context=context,
            output=output,
            reference_section=reference_section,
            criteria=self.criteria,
        )

    def _parse_response(self, response_text: str) -> dict:
        """Parse the LLM response to extract score and reasoning."""
        if self.decision_mode == "binary":
            try:
                payload = BinaryJudgeResponse.model_validate_json(response_text)
            except ValidationError as error:
                raise ValueError(
                    'Binary judge returned invalid output; expected exactly '
                    '{"critique": "...", "result": "Pass"|"Fail"}'
                ) from error
            return {
                "score": 1 if payload.result == "Pass" else 0,
                "verdict": payload.result,
                "reasoning": payload.critique,
                "raw_response": response_text,
            }

        result = {
            "score": None,
            "reasoning": None,
            "raw_response": response_text,
        }

        lines = response_text.strip().split("\n")
        for line in lines:
            line_lower = line.lower()
            if "score:" in line_lower:
                # Extract number from the line
                numbers = re.findall(r'\d+', line)
                if numbers:
                    score = int(numbers[0])
                    result["score"] = min(max(score, 1), 5)  # Clamp to 1-5
            elif "reasoning:" in line_lower:
                # Get everything after "reasoning:"
                idx = line_lower.index("reasoning:")
                result["reasoning"] = line[idx + len("reasoning:"):].strip()
            elif result["reasoning"] is None and result["score"] is not None:
                # Continuation of reasoning
                if line.strip() and not any(kw in line_lower for kw in ["score", "evaluation"]):
                    result["reasoning"] = (result.get("reasoning") or "") + " " + line.strip()

        return result

    def _call_openai(self, prompt: str) -> str:
        response_format = None
        if self.decision_mode == "binary":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "binary_judge_response",
                    "strict": True,
                    "schema": BinaryJudgeResponse.model_json_schema(),
                },
            }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
            **({"response_format": response_format} if response_format else {}),
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        prompt = self._build_prompt(output, reference, input_text)

        try:
            if self.provider == "openai":
                response_text = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response_text = self._call_anthropic(prompt)
            else:
                return {"score": None, "error": f"Unsupported provider: {self.provider}"}

            result = self._parse_response(response_text)
            return result

        except Exception as e:
            return {"score": None, "reasoning": None, "error": str(e)}

    def evaluate_batch(
        self,
        outputs: list[str],
        references: list[str] = None,
        inputs: list[str] = None,
        progress_callback=None
    ):
        """Override to add rate limiting."""
        import pandas as pd

        results = []
        total = len(outputs)

        for i, output in enumerate(outputs):
            ref = references[i] if references else None
            inp = inputs[i] if inputs else None

            result = self.evaluate_single(output, ref, inp)
            result["index"] = i
            results.append(result)

            if progress_callback:
                progress_callback((i + 1) / total)

            # Rate limiting
            if self.rate_limit_delay > 0 and i < total - 1:
                time.sleep(self.rate_limit_delay)

        return pd.DataFrame(results)


def estimate_cost(
    num_samples: int,
    provider: str,
    model: str,
    avg_tokens_per_sample: int = 500
) -> dict:
    """Estimate the cost of running LLM evaluation."""
    model_pricing = get_model_pricing(provider, model)
    if model_pricing is None:
        return {"estimated_cost": "Unknown", "warning": "Model pricing not available"}

    total_tokens = num_samples * avg_tokens_per_sample
    input_tokens = total_tokens * 0.7  # Assume 70% input
    output_tokens = total_tokens * 0.3  # Assume 30% output

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "estimated_cost": f"${total_cost:.4f}",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }
