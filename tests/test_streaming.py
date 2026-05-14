import pytest
import statistics


def test_streaming_is_real(proxy_client, model_entry):
    """Check if streaming actually streams (non-uniform chunk intervals)."""
    resp = proxy_client.chat_stream(
        model_entry["id"],
        [{"role": "user", "content": "Write a 50-word paragraph about spring."}],
        temperature=0.7, max_tokens=300
    )

    chunks = resp.chunks
    assert len(chunks) >= 2, f"Too few chunks: {len(chunks)} — likely fake streaming"

    intervals = []
    for i in range(1, len(chunks)):
        intervals.append(chunks[i].timestamp_ms - chunks[i - 1].timestamp_ms)

    print(f"Chunks: {len(chunks)}, intervals: min={min(intervals):.0f}ms, max={max(intervals):.0f}ms", end="")
    if len(intervals) > 1:
        print(f", std={statistics.stdev(intervals):.0f}ms")
    else:
        print()

    if max(intervals) < 50:
        pytest.fail("FAIL: All chunks arrived at once — fake streaming")
    elif len(intervals) > 1 and statistics.stdev(intervals) < 200:
        pass
    else:
        print(f"First chunk latency: {resp.latency_first_chunk_ms:.0f}ms, Total: {resp.latency_total_ms:.0f}ms")
