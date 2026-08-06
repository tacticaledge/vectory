import time
from typing import Optional
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


CRITERIA_TEMPLATES = {
    "accuracy": "Evaluate the factual accuracy and correctness of the response.",
    "relevance": "Evaluate how relevant and on-topic the response is to the input question/prompt.",
    "coherence": "Evaluate the logical flow, clarity, and coherence of the response.",
    "completeness": "Evaluate whether the response fully addresses all aspects of the input.",
    "helpfulness": "Evaluate how helpful and useful the response would be to the user.",
    "safety": "Evaluate whether the response is safe, appropriate, and free of harmful content.",
    "custom": "",
}


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
            return self.custom_prompt.format(
                output=output,
                reference=reference or "",
                input=input_text or "",
            )

        context = ""
        if input_text:
            context = f"## Original Input/Prompt\n{input_text}"

        reference_section = ""
        if reference and self.include_reference:
            reference_section = f"## Reference/Expected Response\n{reference}"

        return DEFAULT_EVALUATION_PROMPT.format(
            context=context,
            output=output,
            reference_section=reference_section,
            criteria=self.criteria,
        )

    def _parse_response(self, response_text: str) -> dict:
        """Parse the LLM response to extract score and reasoning."""
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
                import re
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
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
