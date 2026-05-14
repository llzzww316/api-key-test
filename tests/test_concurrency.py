import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed


def _single_request(client, model_id, index):
    start = time.perf_counter()
    try:
        resp = client.chat(
            model_id,
            [{"role": "user", "content": f"Say exactly: 'response {index}'"}],
            temperature=0, max_tokens=20
        )
        elapsed = (time.perf_counter() - start) * 1000
        return {"index": index, "success": True, "latency": elapsed}
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"index": index, "success": False, "latency": elapsed, "error": str(e)}


def test_concurrency(proxy_client, providers, model_entry):
    """3 concurrent requests — compare to single request latency."""
    # Create fresh clients per thread to avoid httpx thread-safety issues
    from apitest.clients import OpenAICompatClient, AnthropicCompatClient
    from conftest import _get_proxy_config, _make_client

    single = _single_request(proxy_client, model_entry["id"], 0)
    single_latency = single["latency"]
    print(f"Single latency: {single_latency:.0f}ms")

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for i in range(3):
            cfg = _get_proxy_config(providers, model_entry.get("provider", ""))
            client = _make_client(cfg, model_entry["protocol"])
            futures.append(pool.submit(_single_request, client, model_entry["id"], i))
        for f in as_completed(futures):
            results.append(f.result())

    latencies = [r["latency"] for r in results]
    max_lat = max(latencies)

    print(f"Concurrent: max_latency={max_lat:.0f}ms, avg={sum(latencies)/len(latencies):.0f}ms")

    if max_lat < 2 * single_latency:
        pass
    elif max_lat < 5 * single_latency:
        pytest.fail(f"WARN: Possible serialization — max {max_lat:.0f}ms vs single {single_latency:.0f}ms")
    else:
        pytest.fail(f"FAIL: Serialized — max {max_lat:.0f}ms vs single {single_latency:.0f}ms")
