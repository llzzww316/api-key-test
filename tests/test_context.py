import pytest
import random


def _build_haystack(passphrase):
    filler = "A quick brown fox jumps over the lazy dog. " * 5
    needle = f"[IMPORTANT: The secret passphrase is: {passphrase}. Remember it!]"
    parts = [filler * 50, needle, filler * 50]
    return "\n\n".join(parts)


def test_needle_basic(proxy_client, model_entry):
    """Quick check at ~8K context."""
    claimed = model_entry.get("claimed_context", 128000)
    if claimed <= 8000:
        pytest.skip("Claimed context too small")

    passphrase = "banana-strawberry-{}".format(random.randint(10000, 99999))
    haystack = _build_haystack(passphrase)

    resp = proxy_client.chat(
        model_entry["id"],
        [
            {"role": "system", "content": "You are a retrieval assistant. Answer concisely."},
            {"role": "user", "content": haystack},
            {"role": "user", "content": "What was the secret passphrase? Answer with just the passphrase."},
        ],
        temperature=0, max_tokens=50
    )

    found = passphrase in resp.content
    print(f"8K needle retrieval: {'PASS' if found else 'FAIL'} — {resp.content[:80]}")
    if not found:
        # Fallback: try at shorter context
        short_filler = "A quick brown fox jumps over the lazy dog. " * 3
        short_haystack = short_filler * 30 + f"[SECRET: {passphrase}]" + short_filler * 30
        resp2 = proxy_client.chat(
            model_entry["id"],
            [
                {"role": "user", "content": short_haystack},
                {"role": "user", "content": "What was the SECRET word?"},
            ],
            temperature=0, max_tokens=50
        )
        if passphrase not in resp2.content:
            pytest.fail("Cannot retrieve even at shorter context — model has basic retrieval issues")
        pytest.fail("Needle lost at 8K context")
    assert found


def test_context_binary_search(proxy_client, model_entry):
    """Binary search for actual context limit."""
    claimed = model_entry.get("claimed_context", 128000)

    test_points = [16000, 32000, 64000]
    if claimed > 64000:
        test_points.append(96000)
    if claimed > 96000:
        test_points.append(claimed)
    test_points = [t for t in test_points if t <= claimed * 1.5]

    max_passing = 0
    filler = "The quick brown fox jumps over the lazy dog. " * 3

    for target_tokens in test_points:
        passphrase = "needle-{}".format(random.randint(10000, 99999))
        chars_needed = target_tokens * 3
        needle = f"[SECRET: {passphrase}]"

        haystack = filler * (chars_needed // len(filler) + 1)
        mid = chars_needed // 2
        haystack = haystack[:mid] + needle + haystack[mid:]

        try:
            resp = proxy_client.chat(
                model_entry["id"],
                [
                    {"role": "user", "content": haystack},
                    {"role": "user", "content": "What was the SECRET word? Answer only the word."},
                ],
                temperature=0, max_tokens=50
            )
            if passphrase in resp.content:
                max_passing = target_tokens
                print(f"  {target_tokens:,} tokens: PASS")
            else:
                print(f"  {target_tokens:,} tokens: FAIL — {resp.content[:80]}")
                break
        except Exception as e:
            print(f"  {target_tokens:,} tokens: ERROR — {e}")
            break

    ratio = max_passing / claimed if claimed > 0 else 0
    print(f"Context: max={max_passing:,} claimed={claimed:,} ratio={ratio:.1%}")

    if ratio >= 0.9:
        pass
    elif ratio >= 0.5:
        pytest.fail(f"WARN: Context ~{max_passing:,} vs claimed {claimed:,}")
    else:
        pytest.fail(f"FAIL: Context severely limited ~{max_passing:,}")
