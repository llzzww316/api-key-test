import pytest
import tiktoken


def test_billing_accuracy(proxy_client, model_entry):
    """Check if reported token counts are reasonable."""
    prompt = "Hello, world!"

    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": prompt}],
        temperature=0, max_tokens=10
    )

    reported = resp.usage.get("prompt", 0)

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    expected = len(enc.encode(prompt))

    ratio = reported / expected if expected > 0 else 1.0
    print(f"Billing: reported={reported}, expected_tiktoken={expected}, ratio={ratio:.2f}")

    if 0.5 <= ratio <= 3.0:
        pass
    elif ratio <= 5.0:
        pytest.fail(f"WARN: Reported token count {ratio:.1f}x higher than expected")
    else:
        pytest.fail(f"FAIL: Token count severely inflated: {ratio:.1f}x")
