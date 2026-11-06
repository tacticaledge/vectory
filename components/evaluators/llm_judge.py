import hashlib
import json
import re
import time
from typing import Literal

from components.agentic import BilledCompletionError, Bot, Completion
from components.model_catalog import DEFAULT_MODEL_BY_PROVIDER, get_model_pricing
from components.usage import TokenUsage, usage_cost

# Providers served through Anthropic's Messages API (directly or via Amazon Bedrock).
MESSAGES_API_PROVIDERS = {"anthropic", "bedrock"}

# Bound on reusable verdicts kept per caller-owned cache; the oldest are dropped first.
MAX_REUSED_RESULTS = 5000
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .base import BaseEvaluator


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


def get_result_column(results, name: str):
    """Return a result column aligned to the batch, including all-error batches."""
    import pandas as pd

    return results.get(name, pd.Series(None, index=results.index, dtype=object))


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
        aws_region: str = None,
        result_cache: dict = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.aws_region = aws_region
        # Caller-owned (for example, one Streamlit session's state), so verdicts never cross users.
        self.result_cache = result_cache
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
        return Bot(
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            aws_region=self.aws_region,
        )

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

    def _cache_prefix(self) -> str | None:
        """Return the prompt text shared by every row, which Anthropic can serve from cache."""
        if self.provider not in MESSAGES_API_PROVIDERS or not self.custom_prompt:
            return None
        positions = [
            self.custom_prompt.find(placeholder)
            for placeholder in ("{output}", "{reference}", "{input}")
            if placeholder in self.custom_prompt
        ]
        prefix = self.custom_prompt[: min(positions)] if positions else self.custom_prompt
        return prefix or None

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
                # Read the number after "Score:", not a list marker such as "1. Score: 5"
                numbers = re.findall(r'\d+', line[line_lower.index("score:"):])
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

    def _response_schema(self) -> dict | None:
        return BinaryJudgeResponse.model_json_schema() if self.decision_mode == "binary" else None

    def _call_openai(self, prompt: str) -> Completion:
        return self.client.complete_with_usage(
            [{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
            response_schema=self._response_schema(),
        )

    def _call_anthropic(self, prompt: str) -> Completion:
        # Current Claude models always think; leave room for thinking before the critique.
        return self.client.complete_with_usage(
            [{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3,
            response_schema=self._response_schema(),
            cache_prefix=self._cache_prefix(),
        )

    def _result_key(self, prompt: str) -> str:
        identity = [self.provider, self.model, self.decision_mode, prompt]
        return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()

    def _remember(self, key: str, result: dict) -> None:
        cache = self.result_cache
        cache.pop(key, None)
        cache[key] = result
        while len(cache) > MAX_REUSED_RESULTS:
            cache.pop(next(iter(cache)))

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        prompt = self._build_prompt(output, reference, input_text)
        result_key = self._result_key(prompt) if self.result_cache is not None else None
        if result_key is not None and result_key in self.result_cache:
            # An identical prompt was already judged by the same model: reuse it at no cost.
            return {**self.result_cache[result_key], "reused": True}

        try:
            if self.provider == "openai":
                completion = self._call_openai(prompt)
            elif self.provider in MESSAGES_API_PROVIDERS:
                completion = self._call_anthropic(prompt)
            else:
                return {"score": None, "error": f"Unsupported provider: {self.provider}"}
        except BilledCompletionError as e:
            return {
                "score": None,
                "reasoning": None,
                "error": str(e),
                **usage_columns(self.provider, self.model, e.usage),
            }
        except Exception as e:
            return {"score": None, "reasoning": None, "error": str(e)}

        # Usage is recorded even when the response fails to parse: the call was still billed.
        usage_fields = usage_columns(self.provider, self.model, completion.usage)
        try:
            result = self._parse_response(completion.text)
        except Exception as e:
            return {"score": None, "reasoning": None, "error": str(e), **usage_fields}
        if result_key is not None:
            self._remember(result_key, result)
        return {**result, **usage_fields}

    def measure_input_tokens(
        self,
        outputs: list[str],
        references: list[str] = None,
        inputs: list[str] = None,
        sample_size: int = 20,
    ) -> float | None:
        """Return mean exact input tokens over a row sample, or None when the provider cannot count."""
        if not outputs:
            return None
        step = max(len(outputs) // sample_size, 1)
        counts = []
        for i in range(0, len(outputs), step)[:sample_size]:
            prompt = self._build_prompt(
                outputs[i],
                references[i] if references else None,
                inputs[i] if inputs else None,
            )
            count = self.client.count_input_tokens([{"role": "user", "content": prompt}])
            if count is None:
                return None
            counts.append(count)
        return sum(counts) / len(counts)

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


USAGE_COLUMNS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)


def usage_columns(provider: str, model: str, usage: TokenUsage | None) -> dict:
    """Flatten one call's usage into per-row result columns.

    A call whose provider reported no usage keeps empty token columns and is flagged,
    so its cost is unknown instead of $0.
    """
    if usage is None:
        return {**{name: None for name in USAGE_COLUMNS}, "cost_usd": None, "usage_missing": True}
    return {
        **{name: getattr(usage, name) for name in USAGE_COLUMNS},
        "cost_usd": usage_cost(provider, model, usage),
    }


def summarize_usage(results) -> dict | None:
    """Total the billed usage recorded in a judge results frame, or None if none was recorded.

    A run where every row reused an earlier verdict still has a summary: zero calls, $0.
    """
    if "input_tokens" not in results.columns and "reused" not in results.columns:
        return None
    totals = {
        name: int(get_result_column(results, name).fillna(0).sum()) for name in USAGE_COLUMNS
    }
    reported_rows = int(get_result_column(results, "input_tokens").notna().sum())
    missing_rows = int(get_result_column(results, "usage_missing").fillna(False).astype(bool).sum())
    costs = get_result_column(results, "cost_usd")
    totals["billed_calls"] = reported_rows + missing_rows
    totals["reused_results"] = int(get_result_column(results, "reused").fillna(False).astype(bool).sum())
    totals["cost_usd"] = (
        float(costs.fillna(0).sum())
        if missing_rows == 0 and costs.notna().sum() == reported_rows
        else None
    )
    total_input = totals["input_tokens"] + totals["cached_input_tokens"] + totals["cache_write_tokens"]
    totals["cache_hit_rate"] = totals["cached_input_tokens"] / total_input if total_input else 0.0
    return totals


def estimate_cost(
    num_samples: int,
    provider: str,
    model: str,
    avg_tokens_per_sample: int = 500,
    input_tokens_per_sample: float | None = None,
    output_tokens_per_sample: int = 300,
) -> dict:
    """Estimate the cost of running LLM evaluation.

    With a measured ``input_tokens_per_sample`` the input side is exact for the prompt;
    otherwise ``avg_tokens_per_sample`` is split 70/30 between input and output.
    """
    model_pricing = get_model_pricing(provider, model)
    if model_pricing is None:
        return {"estimated_cost": "Unknown", "warning": "Model pricing not available"}

    if input_tokens_per_sample is not None:
        input_tokens = num_samples * input_tokens_per_sample
        output_tokens = num_samples * output_tokens_per_sample
    else:
        total_tokens = num_samples * avg_tokens_per_sample
        input_tokens = total_tokens * 0.7  # Assume 70% input
        output_tokens = total_tokens * 0.3  # Assume 30% output
    total_tokens = input_tokens + output_tokens

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "estimated_cost": f"${total_cost:.4f}",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }
