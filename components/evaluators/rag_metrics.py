"""
RAG (Retrieval-Augmented Generation) Evaluation Metrics.

Based on best practices from "Application-Centric AI Evals" by Shankar & Husain.
Includes retrieval quality metrics and generation faithfulness evaluation.
"""

from typing import List, Optional, Dict, Any
from .base import BaseEvaluator
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class RetrievalPrecisionEvaluator(BaseEvaluator):
    """
    Evaluates Precision@k for retrieval quality.

    Measures what fraction of retrieved documents are relevant.
    Precision@k = (# relevant docs in top k) / k
    """

    @property
    def name(self) -> str:
        return "Retrieval Precision@k"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate_single(
        self,
        output: str,  # Retrieved doc IDs or texts, comma-separated
        reference: str = None,  # Relevant doc IDs or texts, comma-separated
        input_text: str = None
    ) -> dict:
        if reference is None:
            return {"precision_at_k": None, "error": "Reference required"}

        retrieved = [x.strip() for x in str(output).split(",") if x.strip()]
        relevant = set(x.strip() for x in str(reference).split(",") if x.strip())

        if not retrieved:
            return {"precision_at_k": 0.0, "k": self.k}

        top_k = retrieved[:self.k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        precision = relevant_in_top_k / len(top_k)

        return {
            "precision_at_k": precision,
            "k": self.k,
            "retrieved_count": len(top_k),
            "relevant_in_top_k": relevant_in_top_k,
        }


class RetrievalRecallEvaluator(BaseEvaluator):
    """
    Evaluates Recall@k for retrieval quality.

    Measures what fraction of all relevant documents were retrieved.
    Recall@k = (# relevant docs in top k) / (total # relevant docs)
    """

    @property
    def name(self) -> str:
        return "Retrieval Recall@k"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate_single(
        self,
        output: str,  # Retrieved doc IDs, comma-separated
        reference: str = None,  # Relevant doc IDs, comma-separated
        input_text: str = None
    ) -> dict:
        if reference is None:
            return {"recall_at_k": None, "error": "Reference required"}

        retrieved = [x.strip() for x in str(output).split(",") if x.strip()]
        relevant = set(x.strip() for x in str(reference).split(",") if x.strip())

        if not relevant:
            return {"recall_at_k": 1.0 if not retrieved else 0.0, "k": self.k}

        top_k = retrieved[:self.k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        recall = relevant_in_top_k / len(relevant)

        return {
            "recall_at_k": recall,
            "k": self.k,
            "total_relevant": len(relevant),
            "relevant_in_top_k": relevant_in_top_k,
        }


class MRREvaluator(BaseEvaluator):
    """
    Evaluates Mean Reciprocal Rank (MRR) for retrieval quality.

    MRR = 1/rank of first relevant document.
    Higher is better (1.0 = first doc is relevant).
    """

    @property
    def name(self) -> str:
        return "Mean Reciprocal Rank (MRR)"

    @property
    def requires_reference(self) -> bool:
        return True

    def evaluate_single(
        self,
        output: str,  # Retrieved doc IDs, comma-separated
        reference: str = None,  # Relevant doc IDs, comma-separated
        input_text: str = None
    ) -> dict:
        if reference is None:
            return {"mrr": None, "error": "Reference required"}

        retrieved = [x.strip() for x in str(output).split(",") if x.strip()]
        relevant = set(x.strip() for x in str(reference).split(",") if x.strip())

        if not retrieved or not relevant:
            return {"mrr": 0.0, "first_relevant_rank": None}

        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return {
                    "mrr": 1.0 / rank,
                    "first_relevant_rank": rank,
                }

        return {"mrr": 0.0, "first_relevant_rank": None}


class NDCGEvaluator(BaseEvaluator):
    """
    Evaluates Normalized Discounted Cumulative Gain (NDCG).

    Accounts for position-weighted relevance (top results matter more).
    """

    @property
    def name(self) -> str:
        return "NDCG@k"

    @property
    def requires_reference(self) -> bool:
        return True

    def __init__(self, k: int = 5):
        self.k = k

    def _dcg(self, relevances: List[float]) -> float:
        """Compute DCG for a list of relevance scores."""
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
        return dcg

    def evaluate_single(
        self,
        output: str,  # Retrieved doc IDs, comma-separated
        reference: str = None,  # Relevant doc IDs with optional scores: "doc1:3,doc2:2,doc3:1"
        input_text: str = None
    ) -> dict:
        if reference is None:
            return {"ndcg_at_k": None, "error": "Reference required"}

        retrieved = [x.strip() for x in str(output).split(",") if x.strip()][:self.k]

        # Parse reference with optional relevance scores
        relevance_map = {}
        for item in str(reference).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                doc, score = item.rsplit(":", 1)
                relevance_map[doc.strip()] = float(score)
            else:
                relevance_map[item] = 1.0  # Binary relevance

        if not relevance_map:
            return {"ndcg_at_k": 0.0, "k": self.k}

        # Get relevance scores for retrieved docs
        retrieved_relevances = [relevance_map.get(doc, 0.0) for doc in retrieved]

        # Ideal relevances (sorted descending)
        ideal_relevances = sorted(relevance_map.values(), reverse=True)[:self.k]

        dcg = self._dcg(retrieved_relevances)
        idcg = self._dcg(ideal_relevances)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "ndcg_at_k": ndcg,
            "dcg": dcg,
            "idcg": idcg,
            "k": self.k,
        }


class ContextRelevanceEvaluator(BaseEvaluator):
    """
    Evaluates semantic relevance between retrieved context and the query.

    Uses sentence embeddings to compute similarity.
    Reference-free: only needs the query (input) and retrieved context (output).
    """

    @property
    def name(self) -> str:
        return "Context Relevance"

    @property
    def requires_reference(self) -> bool:
        return False

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        if self.model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)

    def evaluate_single(
        self,
        output: str,  # Retrieved context
        reference: str = None,
        input_text: str = None  # Query
    ) -> dict:
        if not input_text:
            return {"context_relevance": None, "error": "Input query required"}

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return {"context_relevance": None, "error": "sentence-transformers not installed"}

        self._load_model()

        try:
            query_embedding = self.model.encode(input_text, convert_to_tensor=True)
            context_embedding = self.model.encode(str(output), convert_to_tensor=True)

            similarity = util.cos_sim(query_embedding, context_embedding).item()

            return {
                "context_relevance": similarity,
                "is_relevant": similarity >= self.threshold,
                "threshold": self.threshold,
            }
        except Exception as e:
            return {"context_relevance": None, "error": str(e)}


class AnswerFaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluates if the generated answer is faithful to the retrieved context.

    Checks if claims in the answer are supported by the context.
    Uses semantic similarity as a proxy for faithfulness.
    """

    @property
    def name(self) -> str:
        return "Answer Faithfulness"

    @property
    def requires_reference(self) -> bool:
        return True  # Context is passed as reference

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.6):
        self.threshold = threshold
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        if self.model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)

    def evaluate_single(
        self,
        output: str,  # Generated answer
        reference: str = None,  # Retrieved context
        input_text: str = None
    ) -> dict:
        if reference is None:
            return {"faithfulness": None, "error": "Context (reference) required"}

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return {"faithfulness": None, "error": "sentence-transformers not installed"}

        self._load_model()

        try:
            # Split answer into sentences/claims
            answer_sentences = [s.strip() for s in str(output).split(".") if s.strip()]
            if not answer_sentences:
                return {"faithfulness": 1.0, "supported_ratio": 1.0}

            context = str(reference)
            context_embedding = self.model.encode(context, convert_to_tensor=True)

            supported_count = 0
            sentence_scores = []

            for sentence in answer_sentences:
                sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
                similarity = util.cos_sim(sentence_embedding, context_embedding).item()
                sentence_scores.append(similarity)
                if similarity >= self.threshold:
                    supported_count += 1

            faithfulness = supported_count / len(answer_sentences)
            avg_similarity = sum(sentence_scores) / len(sentence_scores)

            return {
                "faithfulness": faithfulness,
                "avg_similarity": avg_similarity,
                "supported_sentences": supported_count,
                "total_sentences": len(answer_sentences),
                "threshold": self.threshold,
            }
        except Exception as e:
            return {"faithfulness": None, "error": str(e)}


class AnswerCompletenessEvaluator(BaseEvaluator):
    """
    Evaluates if the answer covers all key aspects from the query.

    Uses keyword/entity overlap as a proxy for completeness.
    """

    @property
    def name(self) -> str:
        return "Answer Completeness"

    @property
    def requires_reference(self) -> bool:
        return False

    def evaluate_single(
        self,
        output: str,  # Generated answer
        reference: str = None,
        input_text: str = None  # Query
    ) -> dict:
        if not input_text:
            return {"completeness": None, "error": "Input query required"}

        # Extract significant words (simple heuristic)
        def get_keywords(text: str) -> set:
            words = text.lower().split()
            # Filter out common stop words
            stopwords = {
                "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "must", "shall", "can", "to", "of", "in",
                "for", "on", "with", "at", "by", "from", "as", "into", "through",
                "during", "before", "after", "above", "below", "between", "under",
                "again", "further", "then", "once", "here", "there", "when", "where",
                "why", "how", "all", "each", "few", "more", "most", "other", "some",
                "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                "too", "very", "just", "and", "but", "if", "or", "because", "until",
                "while", "about", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "i", "me", "my", "we", "our", "you", "your",
            }
            return {w for w in words if len(w) > 2 and w not in stopwords}

        query_keywords = get_keywords(input_text)
        answer_keywords = get_keywords(str(output))

        if not query_keywords:
            return {"completeness": 1.0, "coverage_ratio": 1.0}

        covered = query_keywords.intersection(answer_keywords)
        completeness = len(covered) / len(query_keywords)

        return {
            "completeness": completeness,
            "query_keywords": len(query_keywords),
            "covered_keywords": len(covered),
            "missing_keywords": list(query_keywords - covered),
        }


def get_rag_evaluators() -> dict:
    """Return a dictionary of all RAG evaluators."""
    return {
        "precision_at_k": RetrievalPrecisionEvaluator,
        "recall_at_k": RetrievalRecallEvaluator,
        "mrr": MRREvaluator,
        "ndcg": NDCGEvaluator,
        "context_relevance": ContextRelevanceEvaluator,
        "answer_faithfulness": AnswerFaithfulnessEvaluator,
        "answer_completeness": AnswerCompletenessEvaluator,
    }
