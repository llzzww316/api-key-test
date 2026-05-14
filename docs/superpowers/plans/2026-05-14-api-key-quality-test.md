# API Key 质量测试框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable pytest-driven test framework that detects model identity fraud, context window shrinkage, quality degradation, and other issues with proxy API providers.

**Architecture:** Config-driven (YAML) test matrix with unified client abstraction over OpenAI/Anthropic protocols. pytest parameterization auto-expands models × tests. BudgetGuard prevents overspend. LLM-as-Judge for subjective evaluations.

**Tech Stack:** Python 3.11+, uv, pytest, httpx, openai SDK, anthropic SDK, tiktoken, pyyaml

---

## File Structure

```
api-key-test/
├── pyproject.toml                  # Project config, dependencies
├── .gitignore
├── .env.example                    # Template for env vars
├── config/
│   ├── providers.example.yaml      # Provider config template
│   ├── models.example.yaml         # Model list template
│   └── judge.yaml                  # LLM Judge config
├── src/apitest/
│   ├── __init__.py
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── base.py                 # UnifiedResponse dataclass + BaseClient ABC
│   │   ├── openai_compat.py        # OpenAI-compatible client
│   │   └── anthropic_compat.py     # Anthropic-compatible client
│   ├── config.py                   # YAML loading + env var expansion
│   ├── budget.py                   # BudgetGuard
│   ├── judge.py                    # LLM-as-Judge
│   ├── probes.py                   # Identity probe definitions
│   └── reporter.py                 # Markdown report generator
├── tests/
│   ├── conftest.py                 # Fixtures + parameterization
│   ├── test_identity.py
│   ├── test_context.py
│   ├── test_quality.py
│   ├── test_stability.py
│   ├── test_streaming.py
│   ├── test_billing.py
│   ├── test_parameters.py
│   ├── test_tool_calling.py
│   └── test_concurrency.py
└── reports/                        # Generated reports (gitignored)
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/apitest/__init__.py`
- Create: `src/apitest/clients/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd C:/1/projects/api-key-test
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "apitest"
version = "0.1.0"
description = "API proxy quality testing framework"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "openai>=1.40",
    "anthropic>=0.40",
    "tiktoken>=0.7",
    "pyyaml>=6.0",
    "pytest>=8.0",
    "pytest-html>=4.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create .gitignore**

```
.env
config/providers.yaml
config/models.yaml
reports/
__pycache__/
*.pyc
.pytest_cache/
htmlcov/
.venv/
```

- [ ] **Step 4: Create .env.example**

```bash
AI18N_API_KEY=your-proxy-key-here
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
BUDGET_LIMIT=1.00
```

- [ ] **Step 5: Create package init files**

`src/apitest/__init__.py`:
```python
"""API proxy quality testing framework."""
```

`src/apitest/clients/__init__.py`:
```python
from .base import UnifiedResponse, BaseClient
from .openai_compat import OpenAICompatClient
from .anthropic_compat import AnthropicCompatClient

__all__ = ["UnifiedResponse", "BaseClient", "OpenAICompatClient", "AnthropicCompatClient"]
```

- [ ] **Step 6: Install dependencies**

```bash
uv sync
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/apitest/__init__.py src/apitest/clients/__init__.py
git commit -m "feat: 初始化项目结构和依赖配置"
```

---

## Task 2: Config Loading

**Files:**
- Create: 
- Create: 
- Create: 
- Create: 
- Test: 

- [ ] **Step 1: Write test for config loading**

`tests/test_config.py`:
```python
import os
import pytest
from pathlib import Path


def test_load_providers_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-secret123")
    config_file = tmp_path / "providers.yaml"
    config_file.write_text("""
proxy:
  name: "test"
  base_url: "https://example.com"
  api_key: "${TEST_API_KEY}"
  openai_compat: "/v1/chat/completions"
  anthropic_compat: "/v1/messages"
  models_list: "/v1/models"
officials: {}
""")
    from apitest.config import load_providers
    providers = load_providers(config_file)
    assert providers["proxy"]["api_key"] == "sk-secret123"


def test_load_models(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  - id: "gpt-4o"
    protocol: openai_compat
    claimed_context: 128000
    has_official: true
    official_provider: openai
    official_model: "gpt-4o"
""")
    from apitest.config import load_models
    models = load_models(config_file)
    assert len(models) == 1
    assert models[0]["id"] == "gpt-4o"
    assert models[0]["has_official"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement config.py**

`src/apitest/config.py`:
```python
import os
import re
from pathlib import Path
import yaml


def _expand_env_vars(obj):
    if isinstance(obj, str):
        return re.sub(
            r"$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_providers(path: Path | None = None) -> dict:
    if path is None:
        path = Path("config/providers.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _expand_env_vars(data)


def load_models(path: Path | None = None) -> list[dict]:
    if path is None:
        path = Path("config/models.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["models"]


def load_judge(path: Path | None = None) -> dict:
    if path is None:
        path = Path("config/judge.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["judge"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Create config example files**

`config/providers.example.yaml`:
```yaml
proxy:
  name: "ai18n"
  base_url: "https://api.ai18n.chat"
  api_key: "${AI18N_API_KEY}"
  openai_compat: "/v1/chat/completions"
  anthropic_compat: "/v1/messages"
  models_list: "/v1/models"

officials:
  openai:
    base_url: "https://api.openai.com"
    api_key: "${OPENAI_API_KEY}"
    compat: "openai"
  # anthropic:
  #   base_url: "https://api.anthropic.com"
  #   api_key: "${ANTHROPIC_API_KEY}"
  #   compat: "anthropic"
```

`config/models.example.yaml`:
```yaml
models:
  - id: "gpt-4o"
    protocol: openai_compat
    claimed_context: 128000
    has_official: true
    official_provider: openai
    official_model: "gpt-4o"

  - id: "claude-sonnet-4-20250514"
    protocol: anthropic_compat
    claimed_context: 200000
    has_official: false

  - id: "deepseek-v3"
    protocol: openai_compat
    claimed_context: 128000
    has_official: false
```

`config/judge.yaml`:
```yaml
judge:
  provider: "openai"
  model: "gpt-4o-mini"
  fallback: "proxy:gpt-4o-mini"
  max_tokens: 200
```

- [ ] **Step 6: Commit**

```bash
git add src/apitest/config.py config/ tests/test_config.py
git commit -m "feat: 配置加载和 YAML 模板"
```

---


## Task 3: Base Client Abstraction

**Files:**
- Create: `src/apitest/clients/base.py`

- [ ] **Step 1: Implement base.py**

`src/apitest/clients/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class UnifiedResponse:
    model: str
    content: str
    usage: dict
    latency_ms: float
    raw: dict
    finish_reason: str
    tool_calls: list | None = None


@dataclass
class StreamChunk:
    content: str
    timestamp_ms: float


@dataclass
class StreamResult:
    chunks: list[StreamChunk] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency_first_chunk_ms: float = 0.0
    latency_total_ms: float = 0.0


class BaseClient(ABC):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @abstractmethod
    def chat(self, model: str, messages: list[dict], **kwargs) -> UnifiedResponse:
        ...

    @abstractmethod
    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> StreamResult:
        ...
```

- [ ] **Step 2: Commit**

```bash
git add src/apitest/clients/base.py
git commit -m "feat: 统一响应格式和客户端基类"
```

---

## Task 4: OpenAI Compatible Client

**Files:**
- Create: `src/apitest/clients/openai_compat.py`
- Test: `tests/test_openai_client.py`

- [ ] **Step 1: Write test**

`tests/test_openai_client.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from apitest.clients.openai_compat import OpenAICompatClient
from apitest.clients.base import UnifiedResponse


def test_openai_client_chat_parses_response():
    client = OpenAICompatClient(
        base_url="https://fake.api.com",
        api_key="sk-fake",
        chat_path="/v1/chat/completions"
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [{"message": {"content": "Hello!", "tool_calls": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    }
    mock_response.elapsed.total_seconds.return_value = 0.5

    with patch("httpx.Client.post", return_value=mock_response):
        result = client.chat("gpt-4o", [{"role": "user", "content": "Hi"}])

    assert isinstance(result, UnifiedResponse)
    assert result.model == "gpt-4o"
    assert result.content == "Hello!"
    assert result.usage == {"prompt": 5, "completion": 2, "total": 7}
    assert result.finish_reason == "stop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OpenAI client**

`src/apitest/clients/openai_compat.py`:
```python
import time
import json
import httpx
from .base import BaseClient, UnifiedResponse, StreamResult, StreamChunk


class OpenAICompatClient(BaseClient):
    def __init__(self, base_url: str, api_key: str, chat_path: str = "/v1/chat/completions"):
        super().__init__(base_url, api_key)
        self.chat_path = chat_path
        self._http = httpx.Client(timeout=120.0)

    def chat(self, model: str, messages: list[dict], **kwargs) -> UnifiedResponse:
        payload = {"model": model, "messages": messages, **kwargs}
        resp = self._http.post(
            f"{self.base_url}{self.chat_path}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return UnifiedResponse(
            model=data.get("model", model),
            content=choice["message"]["content"],
            usage={
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            },
            latency_ms=resp.elapsed.total_seconds() * 1000,
            raw=data,
            finish_reason=choice.get("finish_reason", "unknown"),
            tool_calls=choice["message"].get("tool_calls"),
        )

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> StreamResult:
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        chunks = []
        start = time.perf_counter()
        first_chunk_time = None

        with self._http.stream(
            "POST",
            f"{self.base_url}{self.chat_path}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = now
                    chunks.append(StreamChunk(content=content, timestamp_ms=(now - start) * 1000))

        total_time = (time.perf_counter() - start) * 1000
        return StreamResult(
            chunks=chunks,
            model=model,
            latency_first_chunk_ms=(first_chunk_time - start) * 1000 if first_chunk_time else total_time,
            latency_total_ms=total_time,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/apitest/clients/openai_compat.py tests/test_openai_client.py
git commit -m "feat: OpenAI 兼容协议客户端"
```

---

## Task 5: Anthropic Compatible Client

**Files:**
- Create: `src/apitest/clients/anthropic_compat.py`
- Test: `tests/test_anthropic_client.py`

- [ ] **Step 1: Write test**

`tests/test_anthropic_client.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from apitest.clients.anthropic_compat import AnthropicCompatClient
from apitest.clients.base import UnifiedResponse


def test_anthropic_client_chat_parses_response():
    client = AnthropicCompatClient(
        base_url="https://fake.api.com",
        api_key="sk-fake",
        messages_path="/v1/messages"
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3}
    }
    mock_response.elapsed.total_seconds.return_value = 0.8

    with patch("httpx.Client.post", return_value=mock_response):
        result = client.chat(
            "claude-sonnet-4-20250514",
            [{"role": "user", "content": "Hi"}]
        )

    assert isinstance(result, UnifiedResponse)
    assert result.model == "claude-sonnet-4-20250514"
    assert result.content == "Hello!"
    assert result.usage == {"prompt": 10, "completion": 3, "total": 13}
    assert result.finish_reason == "end_turn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_anthropic_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Anthropic client**

`src/apitest/clients/anthropic_compat.py`:
```python
import time
import json
import httpx
from .base import BaseClient, UnifiedResponse, StreamResult, StreamChunk


class AnthropicCompatClient(BaseClient):
    def __init__(self, base_url: str, api_key: str, messages_path: str = "/v1/messages"):
        super().__init__(base_url, api_key)
        self.messages_path = messages_path
        self._http = httpx.Client(timeout=120.0)

    def chat(self, model: str, messages: list[dict], **kwargs) -> UnifiedResponse:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }
        resp = self._http.post(
            f"{self.base_url}{self.messages_path}",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        text = "".join(b["text"] for b in content_blocks if b["type"] == "text")
        usage = data.get("usage", {})
        input_t = usage.get("input_tokens", 0)
        output_t = usage.get("output_tokens", 0)
        return UnifiedResponse(
            model=data.get("model", model),
            content=text,
            usage={"prompt": input_t, "completion": output_t, "total": input_t + output_t},
            latency_ms=resp.elapsed.total_seconds() * 1000,
            raw=data,
            finish_reason=data.get("stop_reason", "unknown"),
            tool_calls=None,
        )

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> StreamResult:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "stream": True,
            **kwargs,
        }
        chunks = []
        start = time.perf_counter()
        first_chunk_time = None

        with self._http.stream(
            "POST",
            f"{self.base_url}{self.messages_path}",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["type"] == "content_block_delta":
                    text = data["delta"].get("text", "")
                    if text:
                        now = time.perf_counter()
                        if first_chunk_time is None:
                            first_chunk_time = now
                        chunks.append(StreamChunk(content=text, timestamp_ms=(now - start) * 1000))

        total_time = (time.perf_counter() - start) * 1000
        return StreamResult(
            chunks=chunks,
            model=model,
            latency_first_chunk_ms=(first_chunk_time - start) * 1000 if first_chunk_time else total_time,
            latency_total_ms=total_time,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_anthropic_client.py -v`
Expected: PASS

- [ ] **Step 5: Update clients/__init__.py**

`src/apitest/clients/__init__.py`:
```python
from .base import UnifiedResponse, BaseClient, StreamResult, StreamChunk
from .openai_compat import OpenAICompatClient
from .anthropic_compat import AnthropicCompatClient

__all__ = [
    "UnifiedResponse", "BaseClient", "StreamResult", "StreamChunk",
    "OpenAICompatClient", "AnthropicCompatClient",
]
```

- [ ] **Step 6: Commit**

```bash
git add src/apitest/clients/ tests/test_anthropic_client.py
git commit -m "feat: Anthropic 兼容协议客户端"
```

---

