"""
Tests for document_loader module.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.document_loader import (
    load_json,
    load_csv,
    load_file,
    infer_columns,
)
from components.models import ColumnMapping


class TestLoadJson:
    """Tests for JSON loading functionality."""

    def test_load_json_array(self, sample_json_array):
        """Test loading a JSON array."""
        df = load_json(sample_json_array.getvalue())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "input" in df.columns
        assert "output" in df.columns

    def test_load_json_with_data_key(self, sample_json_with_data_key):
        """Test loading JSON with 'data' wrapper key."""
        df = load_json(sample_json_with_data_key.getvalue())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "prompt" in df.columns
        assert "response" in df.columns

    def test_load_jsonl(self, sample_jsonl):
        """Test loading JSONL (newline-delimited JSON)."""
        df = load_json(sample_jsonl.getvalue())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "input" in df.columns


class TestLoadCsv:
    """Tests for CSV loading functionality."""

    def test_load_csv_utf8(self, sample_csv):
        """Test loading UTF-8 encoded CSV."""
        df = load_csv(sample_csv.getvalue())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "input" in df.columns
        assert "output" in df.columns

    def test_load_csv_encoding_fallback(self, sample_csv_latin1):
        """Test CSV loading with encoding fallback."""
        df = load_csv(sample_csv_latin1.getvalue())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestLoadFile:
    """Tests for unified file loading."""

    def test_load_json_file(self, sample_json_array):
        """Test loading JSON through load_file."""
        df = load_file(sample_json_array)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_csv_file(self, sample_csv):
        """Test loading CSV through load_file."""
        df = load_file(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_file_no_file(self):
        """Test error handling when no file provided."""
        with pytest.raises(ValueError, match="No file provided"):
            load_file(None)


class TestInferColumns:
    """Tests for column inference."""

    def test_infer_standard_columns(self, sample_dataframe):
        """Test inference of standard column names."""
        mapping = infer_columns(sample_dataframe)
        assert isinstance(mapping, ColumnMapping)
        assert mapping.input_col == "input"
        assert mapping.output_col == "output"
        assert mapping.expected_col == "expected"

    def test_infer_prompt_columns(self, sample_dataframe_with_prompts):
        """Test inference of 'prompt' style column names."""
        mapping = infer_columns(sample_dataframe_with_prompts)
        assert isinstance(mapping, ColumnMapping)
        assert mapping.input_col == "prompt"
        assert mapping.output_col == "response"
        assert mapping.expected_col == "reference"

    def test_infer_empty_dataframe(self):
        """Test inference on empty DataFrame."""
        df = pd.DataFrame()
        mapping = infer_columns(df)
        assert isinstance(mapping, ColumnMapping)
        assert mapping.input_col is None
        assert mapping.output_col is None
