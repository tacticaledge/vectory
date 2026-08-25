import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.provider_config import ProviderSettings


def test_provider_settings_load_and_hide_environment_secrets():
    settings = ProviderSettings(
        openai_api_key="openai-secret",
        anthropic_api_key="anthropic-secret",
    )

    assert settings.api_key("openai") == "openai-secret"
    assert settings.api_key("anthropic") == "anthropic-secret"
    assert settings.api_key("unknown") == ""
    assert "openai-secret" not in repr(settings)


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
