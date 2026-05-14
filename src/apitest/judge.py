import re
from enum import Enum
from .clients.openai_compat import OpenAICompatClient


class Verdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    def __str__(self):
        return self.value


JUDGE_PROMPTS = {
    "quality_code": """You are evaluating an LLM code output.
Task: {task}
Model output: {output}
Test execution result: {test_results}

Rate as PASS/WARN/FAIL with one-line reason.
- PASS: correct, clean, efficient
- WARN: mostly correct but has minor issues
- FAIL: incorrect logic or won't run
Format: PASS|WARN|FAIL: reason""",

    "quality_reasoning": """You are evaluating an LLM reasoning ability.
Question: {question}
Expected answer: {expected}
Model output: {output}

Rate as PASS/WARN/FAIL with one-line reason.
Format: PASS|WARN|FAIL: reason""",

    "quality_chinese": """你是一个翻译质量评审员。
原文: {original}
模型输出: {output}

评判: 翻译准确性、文化理解、赏析深度
输出: PASS|WARN|FAIL: 一句话理由""",

    "identity_fingerprint": """You are detecting LLM identity misrepresentation.
Model claims to be: {claimed_model}
Expected company: {expected_company}
When asked about identity, it responded: {output}

Does this match what the claimed model would say?
Rate: PASS (consistent) / WARN (ambiguous) / FAIL (different model)
Format: PASS|WARN|FAIL: evidence""",
}


class JudgeClient:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o-mini", protocol: str = "openai_compat"):
        if protocol == "anthropic_compat":
            from .clients.anthropic_compat import AnthropicCompatClient
            self.client = AnthropicCompatClient(
                base_url=base_url,
                api_key=api_key,
                messages_path="/v1/messages",
            )
        else:
            self.client = OpenAICompatClient(
                base_url=base_url,
                api_key=api_key,
                chat_path="/v1/chat/completions",
            )
        self.model = model

    def evaluate(self, category: str, context: dict) -> dict:
        template = JUDGE_PROMPTS.get(category, JUDGE_PROMPTS["quality_code"])
        prompt = template.format(**context)
        resp = self.client.chat(
            self.model,
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        return {
            **self._parse_verdict(resp.content),
            "judge_model": self.model,
            "judge_usage": resp.usage,
        }

    def _parse_verdict(self, text: str) -> dict:
        m = re.match(
            r"(PASS|WARN|FAIL)\s*[:：]\s*(.+)",
            text.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return {
                "verdict": Verdict(m.group(1).upper()),
                "reason": m.group(2).strip(),
            }
        upper = text.strip().upper()
        if upper.startswith("PASS"):
            return {"verdict": Verdict.PASS, "reason": text.strip()}
        if upper.startswith("WARN"):
            return {"verdict": Verdict.WARN, "reason": text.strip()}
        return {"verdict": Verdict.FAIL, "reason": text.strip()}
