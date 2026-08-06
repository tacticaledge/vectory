"""MTEB leaderboard data fetcher with automatic background refresh."""

import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import threading

# Cache for leaderboard data
_leaderboard_cache: Optional[pd.DataFrame] = None
_cache_timestamp: Optional[datetime] = None
_cache_source = "static"
_cache_error: Optional[str] = None
_CACHE_DURATION = timedelta(minutes=30)
_BACKGROUND_REFRESH_INTERVAL_SECONDS = int(os.getenv("VECTORY_LEADERBOARD_REFRESH_SECONDS", "1800"))
_CACHE_PATH = Path(os.getenv("VECTORY_LEADERBOARD_CACHE_PATH", "/tmp/vectory_mteb_leaderboard_cache.json"))
_MTEB_BACKEND_URL = os.getenv("VECTORY_MTEB_BACKEND_URL", "https://mteb-leaderboard-backend.hf.space/v1")
_MTEB_BENCHMARK_NAME = os.getenv("VECTORY_MTEB_BENCHMARK", "MTEB(eng, v2)")
_refresh_lock = threading.Lock()
_refresh_thread: Optional[threading.Thread] = None
_stop_refresh = threading.Event()


# Static fallback data (last updated: Jan 2025)
FALLBACK_LEADERBOARD = [
    {"rank": 1, "model": "voyage-3-large", "provider": "Voyage AI", "mean_score": 0.7120, "retrieval": 0.7456, "sts": 0.8723, "classification": 0.8234, "clustering": 0.5678},
    {"rank": 2, "model": "NV-Embed-v2", "provider": "NVIDIA", "mean_score": 0.6920, "retrieval": 0.6978, "sts": 0.8891, "classification": 0.8567, "clustering": 0.5234},
    {"rank": 3, "model": "voyage-3", "provider": "Voyage AI", "mean_score": 0.6780, "retrieval": 0.7120, "sts": 0.8560, "classification": 0.7890, "clustering": 0.5123},
    {"rank": 4, "model": "gte-Qwen2-7B-instruct", "provider": "Alibaba", "mean_score": 0.6750, "retrieval": 0.6890, "sts": 0.8780, "classification": 0.8456, "clustering": 0.4890},
    {"rank": 5, "model": "e5-mistral-7b-instruct", "provider": "Microsoft", "mean_score": 0.6680, "retrieval": 0.6567, "sts": 0.8670, "classification": 0.8345, "clustering": 0.4789},
    {"rank": 6, "model": "text-embedding-3-large", "provider": "OpenAI", "mean_score": 0.6540, "retrieval": 0.6234, "sts": 0.8450, "classification": 0.8123, "clustering": 0.4567},
    {"rank": 7, "model": "bge-m3", "provider": "BAAI", "mean_score": 0.6520, "retrieval": 0.6345, "sts": 0.8230, "classification": 0.7890, "clustering": 0.4456},
    {"rank": 8, "model": "bge-large-en-v1.5", "provider": "BAAI", "mean_score": 0.6410, "retrieval": 0.5890, "sts": 0.8010, "classification": 0.7678, "clustering": 0.4234},
    {"rank": 9, "model": "stella-en-1.5B-v5", "provider": "Stella", "mean_score": 0.6350, "retrieval": 0.5789, "sts": 0.7890, "classification": 0.7567, "clustering": 0.4123},
    {"rank": 10, "model": "jina-embeddings-v3", "provider": "Jina AI", "mean_score": 0.6280, "retrieval": 0.5678, "sts": 0.7780, "classification": 0.7456, "clustering": 0.4012},
    {"rank": 11, "model": "multilingual-e5-large-instruct", "provider": "Microsoft", "mean_score": 0.6210, "retrieval": 0.5567, "sts": 0.7670, "classification": 0.7345, "clustering": 0.3901},
    {"rank": 12, "model": "text-embedding-3-small", "provider": "OpenAI", "mean_score": 0.6100, "retrieval": 0.5234, "sts": 0.7560, "classification": 0.7234, "clustering": 0.3789},
    {"rank": 13, "model": "e5-large-v2", "provider": "Microsoft", "mean_score": 0.5980, "retrieval": 0.5123, "sts": 0.7450, "classification": 0.7123, "clustering": 0.3678},
    {"rank": 14, "model": "all-mpnet-base-v2", "provider": "Sentence Transformers", "mean_score": 0.5870, "retrieval": 0.4890, "sts": 0.7340, "classification": 0.7012, "clustering": 0.3567},
    {"rank": 15, "model": "bge-base-en-v1.5", "provider": "BAAI", "mean_score": 0.5760, "retrieval": 0.4678, "sts": 0.7230, "classification": 0.6890, "clustering": 0.3456},
    {"rank": 16, "model": "e5-base-v2", "provider": "Microsoft", "mean_score": 0.5650, "retrieval": 0.4567, "sts": 0.7120, "classification": 0.6789, "clustering": 0.3345},
    {"rank": 17, "model": "all-MiniLM-L6-v2", "provider": "Sentence Transformers", "mean_score": 0.5680, "retrieval": 0.4120, "sts": 0.7820, "classification": 0.6723, "clustering": 0.3234},
    {"rank": 18, "model": "bge-small-en-v1.5", "provider": "BAAI", "mean_score": 0.5430, "retrieval": 0.3890, "sts": 0.6910, "classification": 0.6567, "clustering": 0.3012},
    {"rank": 19, "model": "e5-small-v2", "provider": "Microsoft", "mean_score": 0.5320, "retrieval": 0.3789, "sts": 0.6800, "classification": 0.6456, "clustering": 0.2901},
    {"rank": 20, "model": "paraphrase-MiniLM-L6-v2", "provider": "Sentence Transformers", "mean_score": 0.5100, "retrieval": 0.3456, "sts": 0.6590, "classification": 0.6234, "clustering": 0.2678},
]


def _parse_provider_from_model_name(model_name: str) -> str:
    """Extract provider from model name."""
    model_lower = model_name.lower()

    # Map common prefixes/patterns to providers
    provider_patterns = {
        "voyage": "Voyage AI",
        "openai": "OpenAI",
        "text-embedding": "OpenAI",
        "nv-embed": "NVIDIA",
        "nvidia": "NVIDIA",
        "bge": "BAAI",
        "baai": "BAAI",
        "e5-": "Microsoft",
        "multilingual-e5": "Microsoft",
        "intfloat": "Microsoft",
        "gte-": "Alibaba",
        "qwen": "Alibaba",
        "alibaba": "Alibaba",
        "jina": "Jina AI",
        "sentence-transformers": "Sentence Transformers",
        "all-minilm": "Sentence Transformers",
        "all-mpnet": "Sentence Transformers",
        "paraphrase": "Sentence Transformers",
        "stella": "Stella",
        "cohere": "Cohere",
        "nomic": "Nomic AI",
        "mistral": "Mistral AI",
        "gemini": "Google",
        "google": "Google",
        "amazon": "Amazon",
        "titan": "Amazon",
    }

    for pattern, provider in provider_patterns.items():
        if pattern in model_lower:
            return provider

    # Try to extract from model path (e.g., "BAAI/bge-large")
    if "/" in model_name:
        org = model_name.split("/")[0]
        org_map = {
            "BAAI": "BAAI",
            "intfloat": "Microsoft",
            "sentence-transformers": "Sentence Transformers",
            "Alibaba-NLP": "Alibaba",
            "jinaai": "Jina AI",
            "nomic-ai": "Nomic AI",
            "Cohere": "Cohere",
        }
        if org in org_map:
            return org_map[org]
        return org

    return "Unknown"


def _fetch_live_backend_leaderboard(top_n: int = 50) -> Optional[pd.DataFrame]:
    """Fetch compact live scores from the current leaderboard backend."""
    try:
        benchmark = requests.utils.quote(_MTEB_BENCHMARK_NAME, safe="")
        response = requests.get(
            f"{_MTEB_BACKEND_URL}/benchmarks/{benchmark}/scores",
            timeout=30,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        rows = payload.get("rows") or []
        if not rows:
            return None

        results = []
        for row in rows[:top_n]:
            model = row.get("model") or {}
            model_name = model.get("name") or row.get("model_name") or "Unknown"
            scores_by_type = row.get("scoresByTaskType") or {}
            results.append(
                {
                    "rank": row.get("rank"),
                    "model": model_name,
                    "provider": _parse_provider_from_model_name(model_name),
                    "mean_score": row.get("meanTask") or row.get("meanTaskType") or 0,
                    "retrieval": scores_by_type.get("Retrieval", 0),
                    "sts": scores_by_type.get("STS", 0),
                    "classification": scores_by_type.get("Classification", 0),
                    "clustering": scores_by_type.get("Clustering", 0),
                    "release_date": model.get("releaseDate"),
                    "embedding_dim": model.get("embeddingDim"),
                    "max_tokens": model.get("maxTokens"),
                    "open_weights": model.get("openWeights"),
                }
            )

        if results:
            return _normalize_leaderboard(pd.DataFrame(results), top_n)
    except Exception:
        return None

    return None


def fetch_live_leaderboard(top_n: int = 50) -> Optional[pd.DataFrame]:
    """
    Fetch live MTEB leaderboard data from HuggingFace.

    Attempts multiple methods:
    1. HuggingFace Datasets API for mteb/results
    2. Direct API fetch from the leaderboard space

    Args:
        top_n: Number of top models to return

    Returns:
        DataFrame with leaderboard data, or None if fetch fails
    """
    # Method 1: current compact leaderboard backend.
    backend_df = _fetch_live_backend_leaderboard(top_n=top_n)
    if backend_df is not None and not backend_df.empty:
        return backend_df

    # Method 2: Try to use datasets library if available.
    try:
        from datasets import load_dataset

        # Load the MTEB results dataset
        ds = load_dataset("mteb/results", "en", split="test")

        # Process into leaderboard format
        results = []
        for item in ds:
            model_name = item.get("model_name", item.get("model", "Unknown"))
            results.append({
                "model": model_name,
                "provider": _parse_provider_from_model_name(model_name),
                "mean_score": item.get("mean_score", item.get("avg", 0)),
                "retrieval": item.get("retrieval", item.get("Retrieval", 0)),
                "sts": item.get("sts", item.get("STS", 0)),
                "classification": item.get("classification", item.get("Classification", 0)),
                "clustering": item.get("clustering", item.get("Clustering", 0)),
            })

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("mean_score", ascending=False).head(top_n).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
            return df

    except Exception:
        pass  # Fall through to next method

    # Method 3: Try the Gradio API endpoint.
    try:
        # The MTEB leaderboard space often exposes data via API
        api_url = "https://mteb-leaderboard.hf.space/api/predict"

        response = requests.post(
            api_url,
            json={"data": []},
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            # Parse Gradio response format
            if "data" in data and len(data["data"]) > 0:
                leaderboard_data = data["data"][0]
                if isinstance(leaderboard_data, list):
                    # Process the data
                    results = []
                    for row in leaderboard_data[:top_n]:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            model_name = str(row[0]) if row[0] else "Unknown"
                            results.append({
                                "model": model_name,
                                "provider": _parse_provider_from_model_name(model_name),
                                "mean_score": float(row[1]) if len(row) > 1 and row[1] else 0,
                                "retrieval": float(row[2]) if len(row) > 2 and row[2] else 0,
                                "sts": float(row[3]) if len(row) > 3 and row[3] else 0,
                                "classification": float(row[4]) if len(row) > 4 and row[4] else 0,
                                "clustering": float(row[5]) if len(row) > 5 and row[5] else 0,
                            })

                    if results:
                        df = pd.DataFrame(results)
                        df["rank"] = range(1, len(df) + 1)
                        return df
    except Exception:
        pass

    # Method 4: Try to fetch from HuggingFace Hub file.
    try:
        # MTEB often stores results in a JSON file
        hub_url = "https://huggingface.co/datasets/mteb/results/resolve/main/results.json"
        response = requests.get(hub_url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            results = []

            for model_name, scores in data.items():
                if isinstance(scores, dict):
                    mean_score = scores.get("mean", scores.get("avg", 0))
                    results.append({
                        "model": model_name,
                        "provider": _parse_provider_from_model_name(model_name),
                        "mean_score": mean_score,
                        "retrieval": scores.get("Retrieval", scores.get("retrieval", 0)),
                        "sts": scores.get("STS", scores.get("sts", 0)),
                        "classification": scores.get("Classification", scores.get("classification", 0)),
                        "clustering": scores.get("Clustering", scores.get("clustering", 0)),
                    })

            if results:
                df = pd.DataFrame(results)
                df = df.sort_values("mean_score", ascending=False).head(top_n).reset_index(drop=True)
                df["rank"] = range(1, len(df) + 1)
                return df
    except Exception:
        pass

    return None


def _normalize_leaderboard(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = df.copy()
    if "mean_score" in df.columns:
        df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce").fillna(0)
        df = df.sort_values("mean_score", ascending=False)
    df = df.head(top_n).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def _write_disk_cache(df: pd.DataFrame, timestamp: datetime, source: str) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "timestamp": timestamp.isoformat(),
            "rows": df.to_dict(orient="records"),
        }
        _CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _read_disk_cache() -> tuple[Optional[pd.DataFrame], Optional[datetime], str]:
    try:
        if not _CACHE_PATH.exists():
            return None, None, "missing"
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        if not rows:
            return None, None, "empty"
        timestamp = datetime.fromisoformat(payload["timestamp"])
        return pd.DataFrame(rows), timestamp, payload.get("source", "disk")
    except Exception:
        return None, None, "invalid"


def refresh_leaderboard_cache(top_n: int = 100) -> bool:
    """Refresh live leaderboard data and persist a process/disk cache."""
    global _leaderboard_cache, _cache_timestamp, _cache_source, _cache_error

    if not _refresh_lock.acquire(blocking=False):
        return False

    try:
        now = datetime.now()
        live_data = fetch_live_leaderboard(top_n=max(top_n, 100))
        if live_data is None or live_data.empty:
            _cache_error = "Live leaderboard fetch returned no rows."
            return False

        normalized = _normalize_leaderboard(live_data, max(top_n, len(live_data)))
        _leaderboard_cache = normalized
        _cache_timestamp = now
        _cache_source = "live"
        _cache_error = None
        _write_disk_cache(normalized, now, "live")
        return True
    except Exception as exc:
        _cache_error = str(exc)
        return False
    finally:
        _refresh_lock.release()


def _background_refresh_loop() -> None:
    refresh_leaderboard_cache()
    while not _stop_refresh.wait(_BACKGROUND_REFRESH_INTERVAL_SECONDS):
        refresh_leaderboard_cache()


def ensure_background_refresh() -> None:
    """Start one daemon refresh worker for the Streamlit process."""
    global _refresh_thread
    if _refresh_thread is not None and _refresh_thread.is_alive():
        return
    _stop_refresh.clear()
    _refresh_thread = threading.Thread(
        target=_background_refresh_loop,
        name="vectory-leaderboard-refresh",
        daemon=True,
    )
    _refresh_thread.start()


def fetch_mteb_leaderboard(top_n: int = 20, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch MTEB leaderboard data with caching.

    Attempts to fetch live data from HuggingFace, falls back to cached/static data.

    Args:
        top_n: Number of top models to return
        force_refresh: Force a refresh of the cache

    Returns:
        DataFrame with leaderboard data
    """
    global _leaderboard_cache, _cache_timestamp, _cache_source, _cache_error

    # Check cache
    now = datetime.now()
    if force_refresh:
        refresh_leaderboard_cache(top_n=max(top_n, 100))

    if _leaderboard_cache is None:
        disk_df, disk_timestamp, disk_source = _read_disk_cache()
        if disk_df is not None and disk_timestamp is not None:
            _leaderboard_cache = _normalize_leaderboard(disk_df, max(top_n, len(disk_df)))
            _cache_timestamp = disk_timestamp
            _cache_source = disk_source

    if _leaderboard_cache is not None and _cache_timestamp is not None:
        ensure_background_refresh()
        if now - _cache_timestamp >= _CACHE_DURATION and not force_refresh:
            threading.Thread(
                target=refresh_leaderboard_cache,
                kwargs={"top_n": max(top_n, 100)},
                name="vectory-leaderboard-refresh-on-demand",
                daemon=True,
            ).start()
        return _leaderboard_cache.head(top_n).copy()

    # First run without any cache: fetch synchronously so the page gets fresh data.
    if refresh_leaderboard_cache(top_n=max(top_n, 100)) and _leaderboard_cache is not None:
        ensure_background_refresh()
        return _leaderboard_cache.head(top_n).copy()

    # Fall back to static data.
    ensure_background_refresh()
    _cache_source = "static"
    df = pd.DataFrame(FALLBACK_LEADERBOARD[:top_n])
    if "rank" not in df.columns:
        df["rank"] = range(1, len(df) + 1)

    return df


def get_leaderboard_info() -> Dict[str, Any]:
    """
    Get information about the leaderboard data source.

    Returns:
        Dictionary with data source info
    """
    global _cache_timestamp, _cache_source, _cache_error

    if _cache_timestamp is not None:
        age_seconds = max(0, int((datetime.now() - _cache_timestamp).total_seconds()))
        return {
            "source": _cache_source,
            "last_updated": _cache_timestamp.isoformat(),
            "age_seconds": age_seconds,
            "refresh_interval_seconds": _BACKGROUND_REFRESH_INTERVAL_SECONDS,
            "cache_path": str(_CACHE_PATH),
            "error": _cache_error,
        }

    return {
        "source": "static",
        "last_updated": "2025-01",
        "refresh_interval_seconds": _BACKGROUND_REFRESH_INTERVAL_SECONDS,
        "note": "Using cached data. Live data fetch may have failed.",
        "error": _cache_error,
    }


# Available models for custom benchmarking (doesn't require PyTorch)
AVAILABLE_MODELS_INFO = [
    {"name": "all-MiniLM-L6-v2", "provider": "Sentence Transformers", "dimensions": 384, "max_tokens": 256},
    {"name": "all-mpnet-base-v2", "provider": "Sentence Transformers", "dimensions": 768, "max_tokens": 384},
    {"name": "bge-small-en-v1.5", "provider": "BAAI", "dimensions": 384, "max_tokens": 512},
    {"name": "bge-base-en-v1.5", "provider": "BAAI", "dimensions": 768, "max_tokens": 512},
    {"name": "bge-large-en-v1.5", "provider": "BAAI", "dimensions": 1024, "max_tokens": 512},
    {"name": "e5-small-v2", "provider": "Microsoft", "dimensions": 384, "max_tokens": 512},
    {"name": "e5-base-v2", "provider": "Microsoft", "dimensions": 768, "max_tokens": 512},
    {"name": "e5-large-v2", "provider": "Microsoft", "dimensions": 1024, "max_tokens": 512},
    {"name": "jina-embeddings-v3", "provider": "Jina AI", "dimensions": 1024, "max_tokens": 8192},
    {"name": "nomic-embed-text-v1.5", "provider": "Nomic AI", "dimensions": 768, "max_tokens": 8192},
]


def get_available_models() -> List[Dict[str, Any]]:
    """Get list of models available for custom evaluation."""
    return AVAILABLE_MODELS_INFO
