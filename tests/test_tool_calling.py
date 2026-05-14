import pytest
import json


def test_tool_calling(proxy_client, model_entry, judge):
    """Test if model can return a valid tool call."""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }]

    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": "What is the weather in Tokyo? Use get_weather."}],
        temperature=0, max_tokens=100,
        tools=tools,
        tool_choice="auto"
    )

    tool_calls = resp.tool_calls
    print(f"Tool calls: {tool_calls}")

    if tool_calls and len(tool_calls) > 0:
        first = tool_calls[0]
        has_name = "function" in first and "name" in first["function"]
        has_args = "function" in first and "arguments" in first["function"]

        if has_name and has_args:
            try:
                args_str = first["function"]["arguments"]
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                city_ok = "city" in args and "get_weather" in str(first["function"]["name"]).lower()
                print(f"Tool call: {first['function']['name']}({args})")
                if city_ok:
                    pass
                else:
                    pytest.fail(f"WARN: tool format correct but params questionable: {args}")
            except json.JSONDecodeError:
                pytest.fail("WARN: tool call args not valid JSON")
        else:
            pytest.fail("WARN: tool call present but format incomplete")
    else:
        pytest.fail("FAIL: no tool call returned when explicitly requested")
