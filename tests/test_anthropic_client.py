import pytest
from unittest.mock import patch, MagicMock
from apitest.clients.anthropic_compat import AnthropicCompatClient
from apitest.clients.base import UnifiedResponse


def test_anthropic_client_chat_parses_response():
    client = AnthropicCompatClient(
        base_url="https://fake.api.com",
        api_key="sk-fake",
        messages_path="/v1/messages"
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3}
    }
    mock_response.elapsed.total_seconds.return_value = 0.8

    with patch("httpx.Client.post", return_value=mock_response):
        result = client.chat(
            "claude-sonnet-4-20250514",
            [{"role": "user", "content": "Hi"}]
        )

    assert isinstance(result, UnifiedResponse)
    assert result.model == "claude-sonnet-4-20250514"
    assert result.content == "Hello!"
    assert result.usage == {"prompt": 10, "completion": 3, "total": 13}
    assert result.finish_reason == "end_turn"
