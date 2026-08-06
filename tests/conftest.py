"""
Pytest configuration and shared fixtures for LLM Eval Studio tests.
"""

import pytest
import pandas as pd
import json
import sys
from pathlib import Path
from io import BytesIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, content: bytes, name: str):
        self._content = content
        self.name = name

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content


@pytest.fixture
def sample_json_array():
    """Sample JSON array data."""
    data = [
        {"input": "What is 2+2?", "output": "4", "expected": "4"},
        {"input": "Capital of France?", "output": "Paris", "expected": "Paris"},
        {"input": "Largest planet?", "output": "Jupiter", "expected": "Jupiter"},
    ]
    content = json.dumps(data).encode("utf-8")
    return MockUploadedFile(content, "test.json")


@pytest.fixture
def sample_json_with_data_key():
    """Sample JSON with 'data' wrapper key."""
    data = {
        "data": [
            {"prompt": "Hello", "response": "Hi there!", "reference": "Hello!"},
            {"prompt": "Goodbye", "response": "Bye!", "reference": "Goodbye!"},
        ]
    }
    content = json.dumps(data).encode("utf-8")
    return MockUploadedFile(content, "test_wrapped.json")


@pytest.fixture
def sample_jsonl():
    """Sample JSONL data."""
    lines = [
        '{"input": "Line 1", "output": "Response 1"}',
        '{"input": "Line 2", "output": "Response 2"}',
        '{"input": "Line 3", "output": "Response 3"}',
    ]
    content = "\n".join(lines).encode("utf-8")
    return MockUploadedFile(content, "test.jsonl")


@pytest.fixture
def sample_csv():
    """Sample CSV data."""
    content = """input,output,expected
"What is AI?","Artificial Intelligence","Artificial Intelligence"
"Define ML","Machine Learning","Machine Learning"
"What is NLP?","Natural Language Processing","Natural Language Processing"
""".encode("utf-8")
    return MockUploadedFile(content, "test.csv")


@pytest.fixture
def sample_csv_latin1():
    """Sample CSV with Latin-1 encoding."""
    content = """question,answer
"Café?","Yes"
"Naïve?","No"
""".encode("latin-1")
    return MockUploadedFile(content, "test_latin1.csv")


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame."""
    return pd.DataFrame({
        "input": ["Question 1", "Question 2", "Question 3"],
        "output": ["Answer 1", "Answer 2", "Answer 3"],
        "expected": ["Expected 1", "Expected 2", "Expected 3"],
    })


@pytest.fixture
def sample_dataframe_with_prompts():
    """DataFrame with 'prompt' column for inference testing."""
    return pd.DataFrame({
        "prompt": ["Hello", "Goodbye"],
        "response": ["Hi", "Bye"],
        "reference": ["Hello!", "Goodbye!"],
    })
