import os
import pytest
from pathlib import Path


def test_load_providers_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-secret123")
    config_file = tmp_path / "providers.yaml"
    config_file.write_text("""
proxy:
  name: "test"
  base_url: "https://example.com"
  api_key: "${TEST_API_KEY}"
  openai_compat: "/v1/chat/completions"
  anthropic_compat: "/v1/messages"
  models_list: "/v1/models"
officials: {}
""")
    from apitest.config import load_providers
    providers = load_providers(config_file)
    assert providers["proxy"]["api_key"] == "sk-secret123"


def test_load_models(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  - id: "gpt-4o"
    protocol: openai_compat
    claimed_context: 128000
    has_official: true
    official_provider: openai
    official_model: "gpt-4o"
""")
    from apitest.config import load_models
    models = load_models(config_file)
    assert len(models) == 1
    assert models[0]["id"] == "gpt-4o"
    assert models[0]["has_official"] is True
