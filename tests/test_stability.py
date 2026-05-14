import time
import pytest


def test_stability(proxy_client, model_entry):
    """5 requests, same prompt, record latency and success."""
    prompt = "What is 2+2? Answer with just the number."
    latencies = []
    failures = 0

    for i in range(5):
        try:
            resp = proxy_client.chat(
                model_entry["id"],
                [{"role": "user", "content": prompt}],
                temperature=0, max_tokens=100
            )
            latencies.append(resp.latency_ms)
            assert "4" in resp.content, f"Wrong answer: {resp.content[:200]}"
        except Exception as e:
            failures += 1
            print(f"  Request {i+1} FAILED: {e}")
        if i < 4:
            time.sleep(1)

    success_rate = (5 - failures) / 5
    avg_latency = sum(latencies) / len(latencies) if latencies else float("inf")

    print(f"Stability: rate={success_rate:.0%}, avg_latency={avg_latency:.0f}ms, failures={failures}")

    if success_rate >= 0.8 and avg_latency < 30000:
        pass
    elif success_rate >= 0.6:
        pytest.fail(f"WARN: success_rate={success_rate:.0%}, avg_latency={avg_latency:.0f}ms")
    else:
        pytest.fail(f"FAIL: success_rate={success_rate:.0%}, avg_latency={avg_latency:.0f}ms")
