import pytest
from difflib import SequenceMatcher
from apitest.probes import IDENTITY_PROBES, MODEL_FINGERPRINTS


def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


class TestIdentityWithOfficial:
    """A/B comparison tests — require official API key."""

    @pytest.mark.skip_if_no_official
    def test_ab_output_similarity(self, proxy_client, official_client, model_entry):
        prompt = "Explain the concept of recursion in one paragraph."
        proxy_resp = proxy_client.chat(
            model_entry["id"], [{"role": "user", "content": prompt}],
            temperature=0, max_tokens=200
        )
        official_resp = official_client.chat(
            model_entry["official_model"], [{"role": "user", "content": prompt}],
            temperature=0, max_tokens=200
        )
        sim = _similarity(proxy_resp.content, official_resp.content)
        assert sim > 0.3, f"Output similarity too low: {sim:.2f} — possible model swap"

    @pytest.mark.skip_if_no_official
    def test_model_field_matches(self, proxy_client, model_entry):
        resp = proxy_client.chat(
            model_entry["id"],
            [{"role": "user", "content": "Say hi."}],
            max_tokens=10
        )
        expected = model_entry["id"].lower()
        actual = resp.model.lower()
        assert expected in actual or actual in expected, \
            f"Model field mismatch: expected {model_entry['id']}, got {resp.model}"


class TestIdentityProbes:
    """Probe-based tests — works without official key."""

    def test_self_identity(self, proxy_client, model_entry, judge):
        probe = IDENTITY_PROBES[0]
        resp = proxy_client.chat(
            model_entry["id"],
            [{"role": "user", "content": probe["prompt"]}],
            temperature=0, max_tokens=100
        )
        fingerprint = MODEL_FINGERPRINTS.get(model_entry["id"], {})
        expected_company = fingerprint.get("company", "")

        has_other = any(
            c.lower() in resp.content.lower()
            for c in ["Anthropic", "OpenAI", "Google", "DeepSeek", "Meta"]
            if c.lower() != expected_company.lower()
        )

        result = judge.evaluate("identity_fingerprint", {
            "claimed_model": model_entry["id"],
            "expected_company": expected_company or "unknown",
            "output": resp.content,
        })
        print(f"Identity: {result['verdict']} — {result['reason']}")
        if has_other:
            assert result["verdict"].value == "FAIL", \
                f"Identity mismatch: {resp.content[:150]}"

    def test_knowledge_cutoff(self, proxy_client, model_entry, judge):
        probe = IDENTITY_PROBES[1]
        resp = proxy_client.chat(
            model_entry["id"],
            [{"role": "user", "content": probe["prompt"]}],
            temperature=0, max_tokens=100
        )
        fp = MODEL_FINGERPRINTS.get(model_entry["id"], {})
        expected_years = fp.get("knowledge_cutoff", [])
        found = any(y in resp.content for y in expected_years)
        print(f"Cutoff response: {resp.content[:150]}")
        print(f"Expected: {expected_years}, Found: {found}")

    def test_model_name_match(self, proxy_client, model_entry):
        resp = proxy_client.chat(
            model_entry["id"],
            [{"role": "user", "content": "Say hello."}],
            temperature=0, max_tokens=50
        )
        expected = model_entry["id"].lower()
        actual = resp.model.lower()
        assert expected in actual or actual in expected, \
            f"Field mismatch: expected {model_entry['id']}, got {resp.model}"
