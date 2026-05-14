import os

DEFAULT_PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.0 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4-turbo": {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
    "o1": {"input": 15.0 / 1_000_000, "output": 60.0 / 1_000_000},
    "claude-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    "claude-haiku": {"input": 0.80 / 1_000_000, "output": 4.0 / 1_000_000},
    "deepseek": {"input": 0.27 / 1_000_000, "output": 1.10 / 1_000_000},
    "gemini": {"input": 1.25 / 1_000_000, "output": 5.0 / 1_000_000},
}
FALLBACK_PRICING = {"input": 5.0 / 1_000_000, "output": 15.0 / 1_000_000}


class BudgetExceeded(Exception):
    pass


class BudgetGuard:
    def __init__(self, limit_usd: float | None = None, pricing: dict | None = None):
        self.limit = limit_usd or float(os.environ.get("BUDGET_LIMIT", "1.00"))
        self.pricing = pricing or DEFAULT_PRICING
        self.spent = 0.0

    @property
    def remaining(self) -> float:
        return max(0, self.limit - self.spent)

    def _get_pricing(self, model: str) -> dict:
        for key in self.pricing:
            if key in model.lower():
                return self.pricing[key]
        return FALLBACK_PRICING

    def estimate_cost(self, tokens: int, model: str) -> float:
        p = self._get_pricing(model)
        return tokens * p["input"] * 1.5

    def check(self, estimated_tokens: int, model: str):
        cost = self.estimate_cost(estimated_tokens, model)
        if self.spent + cost > self.limit:
            raise BudgetExceeded(
                f"Budget exceeded: spent ${self.spent:.4f}, limit ${self.limit:.4f}"
            )

    def record(self, usage: dict, model: str):
        p = self._get_pricing(model)
        self.spent += usage.get("prompt", 0) * p["input"]
        self.spent += usage.get("completion", 0) * p["output"]
