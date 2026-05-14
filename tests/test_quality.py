import pytest
import re


QUALITY_PROMPTS = {
    "code": (
        "Write a Python function `merge_sorted(a, b)` that merges two sorted lists "
        "into one sorted list in O(n) time. Only the function, no explanation."
    ),
    "reasoning": (
        "Alice is older than Bob. Bob is younger than Carol. Charlie is older than Alice. "
        "Who is the youngest? Answer with just the name."
    ),
    "chinese": (
        "把'春风又绿江南岸，明月何时照我还'翻译成英文并简要赏析。"
    ),
}

EXPECTED_REASONING_ANSWER = "bob"


def test_quality_code(proxy_client, model_entry, judge):
    prompt = QUALITY_PROMPTS["code"]
    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": prompt}],
        temperature=0, max_tokens=300
    )
    code = resp.content

    func_match = re.search(r"def merge_sorted\s*\([^)]*\)\s*:.+", code, re.DOTALL)
    code_to_test = func_match.group(0) if func_match else code

    test_results = "execution skipped"
    try:
        ns = {}
        exec(code_to_test, ns)
        fn = ns.get("merge_sorted")
        if fn:
            cases = [
                ([], [], []),
                ([1, 3], [2, 4], [1, 2, 3, 4]),
                ([1, 2, 3], [], [1, 2, 3]),
            ]
            for a, b, exp in cases:
                got = fn(a[:], b[:])
                assert got == exp, f"merge_sorted({a}, {b}) = {got}, expected {exp}"
            test_results = "all exec tests passed"
        else:
            test_results = "no merge_sorted function found"
    except Exception as e:
        test_results = f"exec failed: {e}"

    result = judge.evaluate("quality_code", {
        "task": prompt,
        "output": code[:500],
        "test_results": test_results,
    })
    print(f"Code quality: {result['verdict']} — {result['reason']}")
    assert result["verdict"].value != "FAIL", f"Code FAIL: {result['reason']}"


def test_quality_reasoning(proxy_client, model_entry, judge):
    prompt = QUALITY_PROMPTS["reasoning"]
    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": prompt}],
        temperature=0, max_tokens=100
    )
    answer_lower = resp.content.lower().strip()
    correct = EXPECTED_REASONING_ANSWER in answer_lower

    result = judge.evaluate("quality_reasoning", {
        "question": prompt,
        "expected": EXPECTED_REASONING_ANSWER,
        "output": resp.content,
    })
    print(f"Reasoning: {result['verdict']} (correct={'yes' if correct else 'no'}) — {result['reason']}")
    if not correct:
        assert result["verdict"].value != "PASS", \
            f"Wrong answer marked PASS: {resp.content[:100]}"


def test_quality_chinese(proxy_client, model_entry, judge):
    prompt = QUALITY_PROMPTS["chinese"]
    resp = proxy_client.chat(
        model_entry["id"],
        [{"role": "user", "content": prompt}],
        temperature=0, max_tokens=300
    )
    result = judge.evaluate("quality_chinese", {
        "original": "春风又绿江南岸，明月何时照我还",
        "output": resp.content,
    })
    print(f"Chinese: {result['verdict']} — {result['reason']}")
    assert result["verdict"].value != "FAIL", f"Chinese FAIL: {result['reason']}"
