import pytest


def test_temperature_zero_deterministic(proxy_client, model_entry):
    """temperature=0 should produce identical outputs."""
    prompt = "What is the capital of France? Answer in one word."
    outputs = []
    for _ in range(3):
        resp = proxy_client.chat(
            model_entry["id"],
            [{"role": "user", "content": prompt}],
            temperature=0, max_tokens=20
        )
        outputs.append(resp.content.strip())

    unique = len(set(outputs))
    print(f"temperature=0: {unique} unique outputs out of 3")

    if unique == 1:
        pass
    elif unique == 2:
        pytest.fail("WARN: temperature=0 produced 2 different outputs")
    else:
        pytest.fail("FAIL: temperature=0 completely ignored — 3 different outputs")


def test_max_tokens_enforced(proxy_client, model_entry):
    """max_tokens should be respected."""
    max_tok = 10
    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": "Count from 1 to 100, one per line."}],
        temperature=0, max_tokens=max_tok
    )
    reported_completion = resp.usage.get("completion", 0)
    print(f"max_tokens={max_tok}, reported completion_tokens={reported_completion}")

    if reported_completion <= max_tok + 5:
        pass
    else:
        pytest.fail(f"FAIL: max_tokens={max_tok} but got {reported_completion} completion tokens")


def test_stop_sequence(proxy_client, model_entry):
    """stop sequence should halt generation."""
    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": "Say: 'Hello STOP World' and then continue talking."}],
        temperature=0, max_tokens=100, stop="STOP"
    )
    content = resp.content
    print(f"Stop test output: {content[:100]}")

    if "STOP" not in content:
        pass  # Model stopped before STOP
    elif "World" in content.split("STOP")[-1] if "STOP" in content else False:
        pytest.fail("FAIL: stop sequence ignored — continued after STOP")
    else:
        pass  # Stopped at STOP
