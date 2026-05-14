from apitest.judge import JudgeClient, Verdict


def test_parse_verdict_pass():
    judge = JudgeClient(api_key="sk-fake", base_url="https://fake.api.com", model="gpt-4o-mini")
    result = judge._parse_verdict("PASS: code is correct and efficient")
    assert result["verdict"] == Verdict.PASS
    assert "correct" in result["reason"]

def test_parse_verdict_warn():
    judge = JudgeClient(api_key="sk-fake", base_url="https://fake.api.com", model="gpt-4o-mini")
    result = judge._parse_verdict("WARN: mostly correct but minor bug")
    assert result["verdict"] == Verdict.WARN

def test_parse_verdict_fail():
    judge = JudgeClient(api_key="sk-fake", base_url="https://fake.api.com", model="gpt-4o-mini")
    result = judge._parse_verdict("FAIL: logic error in merge")
    assert result["verdict"] == Verdict.FAIL

def test_parse_verdict_chinese_colon():
    judge = JudgeClient(api_key="sk-fake", base_url="https://fake.api.com", model="gpt-4o-mini")
    result = judge._parse_verdict("PASS：翻译准确，赏析有深度")
    assert result["verdict"] == Verdict.PASS

def test_parse_verdict_no_colon_fallback():
    judge = JudgeClient(api_key="sk-fake", base_url="https://fake.api.com", model="gpt-4o-mini")
    result = judge._parse_verdict("PASS the implementation looks good")
    assert result["verdict"] == Verdict.PASS
