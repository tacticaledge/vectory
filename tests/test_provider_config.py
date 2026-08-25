from components.provider_config import ProviderSettings


def test_provider_settings_load_and_hide_environment_secrets():
    settings = ProviderSettings.from_environment(
        {"OPENAI_API_KEY": "openai-secret", "ANTHROPIC_API_KEY": "anthropic-secret"}
    )

    assert settings.api_key("openai") == "openai-secret"
    assert settings.api_key("anthropic") == "anthropic-secret"
    assert settings.api_key("unknown") == ""
    assert "openai-secret" not in repr(settings)
