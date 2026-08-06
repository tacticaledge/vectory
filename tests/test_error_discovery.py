"""
Tests for AI-assisted error discovery helpers.
"""

import pytest

from components.error_discovery import build_taxonomy_prompt, parse_taxonomy_suggestions


def test_build_taxonomy_prompt_includes_notes_and_existing_modes():
    prompt = build_taxonomy_prompt(
        ["Too verbose", "Invents unsupported dates"],
        {"Hallucination": {"description": "Adds facts not present in source"}},
    )

    assert "Too verbose" in prompt
    assert "Invents unsupported dates" in prompt
    assert "Hallucination" in prompt
    assert "valid JSON" in prompt


def test_parse_taxonomy_suggestions_from_json_object():
    response = """
    {
      "suggestions": [
        {
          "name": "Unsupported Specificity",
          "description": "Adds precise details that are not supported by the source.",
          "examples": ["Invents unsupported dates"],
          "rationale": "Both notes mention unsupported concrete claims."
        }
      ]
    }
    """

    suggestions = parse_taxonomy_suggestions(response)

    assert suggestions == [
        {
            "name": "Unsupported Specificity",
            "description": "Adds precise details that are not supported by the source.",
            "examples": ["Invents unsupported dates"],
            "rationale": "Both notes mention unsupported concrete claims.",
        }
    ]


def test_parse_taxonomy_suggestions_from_fenced_json_array():
    response = """```json
    [
      {
        "title": "Formulaic Phrasing",
        "definition": "Uses a repeated rhetorical pattern that reviewers dislike.",
        "examples": "I don't like the it's not X it's Y phrasing"
      }
    ]
    ```"""

    suggestions = parse_taxonomy_suggestions(response)

    assert suggestions[0]["name"] == "Formulaic Phrasing"
    assert suggestions[0]["description"] == "Uses a repeated rhetorical pattern that reviewers dislike."
    assert suggestions[0]["examples"] == ["I don't like the it's not X it's Y phrasing"]


def test_parse_taxonomy_suggestions_skips_invalid_and_duplicate_items():
    response = """
    {
      "suggestions": [
        {"name": "Missing Constraint", "description": "Ignores an explicit user constraint."},
        {"name": "missing constraint", "description": "Duplicate with different casing."},
        {"name": "", "description": "No name"},
        {"name": "No Description"}
      ]
    }
    """

    suggestions = parse_taxonomy_suggestions(response)

    assert len(suggestions) == 1
    assert suggestions[0]["name"] == "Missing Constraint"


def test_parse_taxonomy_suggestions_rejects_response_without_json():
    with pytest.raises(ValueError):
        parse_taxonomy_suggestions("No structured response here")
