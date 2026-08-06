from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the evaluator."""
        pass

    @property
    @abstractmethod
    def requires_reference(self) -> bool:
        """Return True if this evaluator requires reference/expected outputs."""
        pass

    @abstractmethod
    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        """Evaluate a single output.

        Args:
            output: The LLM output to evaluate
            reference: The expected/reference output (optional)
            input_text: The input prompt (optional)

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def evaluate_batch(
        self,
        outputs: list[str],
        references: list[str] = None,
        inputs: list[str] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """Evaluate a batch of outputs.

        Args:
            outputs: List of LLM outputs
            references: List of expected/reference outputs
            inputs: List of input prompts
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with evaluation results
        """
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

        return pd.DataFrame(results)
