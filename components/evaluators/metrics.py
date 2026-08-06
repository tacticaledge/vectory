import re
from typing import Optional
from .base import BaseEvaluator

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    levenshtein_distance = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import evaluate
    bleu_metric = evaluate.load("bleu")
except Exception:
    bleu_metric = None


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluates exact string match between output and reference."""

    @property
    def name(self) -> str:
        return "Exact Match"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        if reference is None:
            return {"exact_match": None, "error": "Reference required"}

        out = str(output)
        ref = str(reference)

        if self.strip_whitespace:
            out = out.strip()
            ref = ref.strip()

        if not self.case_sensitive:
            out = out.lower()
            ref = ref.lower()

        match = out == ref
        return {"exact_match": 1.0 if match else 0.0}


class ContainsEvaluator(BaseEvaluator):
    """Evaluates if the output contains the reference."""

    @property
    def name(self) -> str:
        return "Contains Reference"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        if reference is None:
            return {"contains": None, "error": "Reference required"}

        out = str(output)
        ref = str(reference)

        if not self.case_sensitive:
            out = out.lower()
            ref = ref.lower()

        contains = ref in out
        return {"contains": 1.0 if contains else 0.0}


class RegexEvaluator(BaseEvaluator):
    """Evaluates if the output matches a regex pattern."""

    @property
    def name(self) -> str:
        return "Regex Match"

    @property
    def requires_reference(self) -> bool:
        return False

    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = re.compile(pattern, flags)

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        match = bool(self.pattern.search(str(output)))
        return {"regex_match": 1.0 if match else 0.0}


class LevenshteinEvaluator(BaseEvaluator):
    """Evaluates similarity using Levenshtein distance."""

    @property
    def name(self) -> str:
        return "Levenshtein Similarity"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        if reference is None:
            return {"levenshtein_similarity": None, "error": "Reference required"}

        if levenshtein_distance is None:
            return {"levenshtein_similarity": None, "error": "python-Levenshtein not installed"}

        out = str(output)
        ref = str(reference)

        if not self.case_sensitive:
            out = out.lower()
            ref = ref.lower()

        dist = levenshtein_distance(out, ref)
        max_len = max(len(out), len(ref))

        similarity = 1 - (dist / max_len) if max_len > 0 else 1.0
        return {
            "levenshtein_distance": dist,
            "levenshtein_similarity": similarity
        }


class BLEUEvaluator(BaseEvaluator):
    """Evaluates BLEU score for translation/generation quality."""

    @property
    def name(self) -> str:
        return "BLEU Score"

    @property
    def requires_reference(self) -> bool:
        return True

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        if reference is None:
            return {"bleu": None, "error": "Reference required"}

        if bleu_metric is None:
            # Fallback simple BLEU implementation
            return self._simple_bleu(str(output), str(reference))

        try:
            result = bleu_metric.compute(
                predictions=[str(output)],
                references=[[str(reference)]]
            )
            return {"bleu": result["bleu"]}
        except Exception as e:
            return {"bleu": None, "error": str(e)}

    def _simple_bleu(self, output: str, reference: str) -> dict:
        """Simple BLEU approximation when evaluate library unavailable."""
        out_tokens = output.lower().split()
        ref_tokens = reference.lower().split()

        if not out_tokens or not ref_tokens:
            return {"bleu": 0.0}

        # Count matching unigrams
        matches = sum(1 for t in out_tokens if t in ref_tokens)
        precision = matches / len(out_tokens) if out_tokens else 0

        # Brevity penalty
        bp = min(1.0, len(out_tokens) / len(ref_tokens)) if ref_tokens else 0

        return {"bleu": bp * precision}


class ROUGEEvaluator(BaseEvaluator):
    """Evaluates ROUGE scores for summarization quality."""

    @property
    def name(self) -> str:
        return "ROUGE Score"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, rouge_types: list = None):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        if rouge_scorer:
            self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        else:
            self.scorer = None

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        if reference is None:
            return {"rouge1": None, "rouge2": None, "rougeL": None, "error": "Reference required"}

        if self.scorer is None:
            return {"rouge1": None, "rouge2": None, "rougeL": None, "error": "rouge-score not installed"}

        try:
            scores = self.scorer.score(str(reference), str(output))
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception as e:
            return {"rouge1": None, "rouge2": None, "rougeL": None, "error": str(e)}


class LengthEvaluator(BaseEvaluator):
    """Evaluates output length characteristics."""

    @property
    def name(self) -> str:
        return "Length Analysis"

    @property
    def requires_reference(self) -> bool:
        return False

    def evaluate_single(self, output: str, reference: str = None, input_text: str = None) -> dict:
        out = str(output)
        result = {
            "char_count": len(out),
            "word_count": len(out.split()),
            "sentence_count": len([s for s in out.split(".") if s.strip()]),
        }

        if reference:
            ref = str(reference)
            result["length_ratio"] = len(out) / len(ref) if len(ref) > 0 else 0

        return result


def get_all_evaluators() -> dict:
    """Return a dictionary of all available evaluators."""
    return {
        "exact_match": ExactMatchEvaluator,
        "contains": ContainsEvaluator,
        "levenshtein": LevenshteinEvaluator,
        "bleu": BLEUEvaluator,
        "rouge": ROUGEEvaluator,
        "length": LengthEvaluator,
        "regex": RegexEvaluator,
    }
