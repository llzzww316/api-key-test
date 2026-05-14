import pytest
from unittest.mock import patch, MagicMock
from apitest.clients.openai_compat import OpenAICompatClient
from apitest.clients.base import UnifiedResponse


def test_openai_client_chat_parses_response():
    client = OpenAICompatClient(
        base_url="https://fake.api.com",
        api_key="sk-fake",
        chat_path="/v1/chat/completions"
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [{"message": {"content": "Hello!", "tool_calls": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    }
    mock_response.elapsed.total_seconds.return_value = 0.5

    with patch("httpx.Client.post", return_value=mock_response):
        result = client.chat("gpt-4o", [{"role": "user", "content": "Hi"}])

    assert isinstance(result, UnifiedResponse)
    assert result.model == "gpt-4o"
    assert result.content == "Hello!"
    assert result.usage == {"prompt": 5, "completion": 2, "total": 7}
    assert result.finish_reason == "stop"
