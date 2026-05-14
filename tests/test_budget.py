import os
import pytest
from apitest.budget import BudgetGuard, BudgetExceeded

PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.0 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}

def test_budget_records_and_tracks():
    guard = BudgetGuard(limit_usd=0.01, pricing=PRICING)
    guard.record({"prompt": 100, "completion": 50, "total": 150}, "gpt-4o")
    assert guard.spent > 0
    assert guard.remaining < guard.limit

def test_budget_raises_when_exceeded():
    guard = BudgetGuard(limit_usd=0.0001, pricing=PRICING)
    guard.record({"prompt": 10000, "completion": 5000, "total": 15000}, "gpt-4o")
    with pytest.raises(BudgetExceeded):
        guard.check(1000, "gpt-4o")

def test_budget_allows_when_under_limit():
    guard = BudgetGuard(limit_usd=10.0, pricing=PRICING)
    guard.check(1000, "gpt-4o")  # should not raise

def test_budget_reads_from_env(monkeypatch):
    monkeypatch.setenv("BUDGET_LIMIT", "0.50")
    guard = BudgetGuard(pricing=PRICING)
    assert guard.limit == 0.50

def test_budget_remaining():
    guard = BudgetGuard(limit_usd=1.00, pricing=PRICING)
    assert guard.remaining == 1.00
    guard.record({"prompt": 100000, "completion": 50000, "total": 150000}, "gpt-4o")
    assert guard.remaining < 1.00
