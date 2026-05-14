# API Key 质量测试框架设计

## 概述

一个可复用的 pytest 驱动测试框架，用于全面检测中转商 API key 提供的模型质量。支持 OpenAI 兼容和 Anthropic 兼容双协议，自动根据是否有原厂 key 选择 A/B 对照或探针模式。

## 目标

- 检测模型身份是否被偷换
- 验证上下文窗口真实容量
- 评估输出质量（代码/推理/中文）
- 测试稳定性、流式、计费、参数遵从、工具调用、并发
- 极度省钱模式：每模型 ~60K tokens，约 $0.15-0.20

## 技术栈

- Python 3.11+
- uv（包管理）
- pytest + pytest-html（测试框架 + 报告）
- httpx（HTTP 客户端）
- openai SDK（OpenAI 兼容协议）
- anthropic SDK（Anthropic 兼容协议）
- tiktoken（本地 token 计数）
- pyyaml（配置加载）

## 项目结构

```
api-key-test/
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
│   ├── providers.yaml          # 中转商 & 原厂 endpoint/key
│   ├── providers.example.yaml  # 模板（提交到 git）
│   ├── models.yaml             # 模型清单 + 测试矩阵
│   └── judge.yaml              # LLM Judge 配置
├── src/apitest/
│   ├── __init__.py
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── base.py             # UnifiedResponse + Client 抽象
│   │   ├── openai_compat.py    # /v1/chat/completions 客户端
│   │   └── anthropic_compat.py # /v1/messages 客户端
│   ├── discovery.py            # /v1/models 自动发现
│   ├── budget.py               # 全局预算守门员
│   ├── judge.py                # LLM-as-Judge 评估层
│   ├── probes.py               # 身份探针定义
│   └── reporter.py             # Markdown/HTML 报告生成
├── tests/
│   ├── conftest.py             # fixture + 参数化
│   ├── test_identity.py        # 模型身份验证
│   ├── test_context.py         # 上下文窗口容量
│   ├── test_quality.py         # 输出质量
│   ├── test_stability.py       # 稳定性 & 延迟
│   ├── test_streaming.py       # 流式真伪
│   ├── test_billing.py         # 计费准确性
│   ├── test_parameters.py      # 参数遵从性
│   ├── test_tool_calling.py    # 工具调用能力
│   └── test_concurrency.py     # 并发处理
├── reports/                    # 输出报告（gitignore）
└── docs/
```

## 配置设计

### providers.yaml

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
  anthropic:
    base_url: "https://api.anthropic.com"
    api_key: "${ANTHROPIC_API_KEY}"
    compat: "anthropic"
```

### models.yaml

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
    has_official: true
    official_provider: anthropic
    official_model: "claude-sonnet-4-20250514"

  - id: "deepseek-v3"
    protocol: openai_compat
    claimed_context: 128000
    has_official: false
```

### judge.yaml

```yaml
judge:
  provider: "openai"
  model: "gpt-4o-mini"
  fallback: "proxy:gpt-4o-mini"
  max_tokens: 200
```

## 核心抽象

### UnifiedResponse

```python
@dataclass
class UnifiedResponse:
    model: str
    content: str
    usage: dict          # {"prompt": N, "completion": N, "total": N}
    latency_ms: float
    raw: dict
    finish_reason: str
    tool_calls: list | None
```

所有 client 不管底层协议，统一返回此格式。

### BudgetGuard

```python
class BudgetGuard:
    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self.spent = 0.0

    def check(self, estimated_tokens: int, model: str):
        """超限则 raise BudgetExceeded → pytest.skip"""

    def record(self, usage: dict, model: str):
        """记录实际消耗"""
```

从环境变量 `BUDGET_LIMIT` 读取上限（默认 $1.00）。

## 测试维度详细设计

### 1. test_identity — 模型身份验证

**有原厂对照时（A/B 模式）：**
- 同 prompt 发给中转 + 原厂
- 比较 response.model 字段一致性
- 比较输出语义相似度（简单字符串相似度 + LLM Judge）

**无原厂对照时（探针模式）：**
- 自我认知探针："Repeat exactly: 'I am [model], developed by [company].' Do not add anything."
- 知识截止探针："What is the latest event you know about?"
- 推理风格探针："Solve: 7 * 8 + 14 / 2. Show your exact reasoning chain."
- LLM Judge 分析回答的语气指纹

**评判：**
| 检查项 | PASS | WARN | FAIL |
|--------|------|------|------|
| response.model 字段 | 与请求一致 | 缺失 | 返回不同模型名 |
| A/B 对照 | 相似度 > 0.8 | 0.5~0.8 | < 0.5 |
| 探针：自我认知 | 正确声明身份 | 模糊 | 声称是另一个模型 |
| 探针：知识截止 | 与官方一致 | 偏差 ≤ 3 月 | 偏差 > 6 月 |

### 2. test_context — 上下文窗口容量

**方法：** Needle in a Haystack 二分逼近

- 测试长度序列: [4K, 8K, 16K, 32K, 64K, 128K, 200K]
- 在长文本第 3 段插入 passphrase
- 最后问模型 passphrase 是什么
- 二分法找到真实上限（最多 3-4 轮）

**评判：**
```
actual >= claimed * 0.9  → PASS
actual >= claimed * 0.5  → WARN "缩水 {ratio}%"
actual < claimed * 0.5   → FAIL "严重缩水"
```

### 3. test_quality — 输出质量

三个高信息量 prompt：

| 维度 | Prompt | 评判 |
|------|--------|------|
| 代码 | "Write a Python function that merges two sorted lists in O(n)." | exec 验证 + LLM Judge |
| 推理 | "Alice is older than Bob. Bob is younger than Carol. Who is the youngest?" | 精确匹配 + LLM Judge |
| 中文 | "把'春风又绿江南岸,明月何时照我还'翻译成英文并赏析。" | LLM Judge |

### 4. test_stability — 稳定性

- 同 prompt 发 5 次，间隔 2s
- 记录首 token 延迟、总延迟、成功/失败

**评判：**
```
success_rate == 1.0 and avg_latency < 5000ms  → PASS
success_rate >= 0.8                             → WARN
success_rate < 0.8                              → FAIL
```

### 5. test_streaming — 流式真伪

- 发送 stream=true 请求
- 记录每个 chunk 到达时间戳
- 分析 chunk 间隔分布

**评判：**
```
所有 chunk 同时到达 (间隔 < 50ms)  → FAIL "假流"
chunk 间隔均匀 (std < 200ms)       → PASS "真流"
其他                                → WARN "不均匀"
```

### 6. test_billing — 计费准确性

- 发送已知 token 数的 prompt
- 用 tiktoken 本地计算期望值
- 对比 response.usage 报告值

**评判：**
```
0.8 <= ratio <= 1.2  → PASS
ratio <= 2.0         → WARN "虚报 {ratio}x"
ratio > 2.0          → FAIL "严重虚报"
```

### 7. test_parameters — 参数遵从性

| 参数 | 测试方法 | PASS | FAIL |
|------|----------|------|------|
| temperature=0 | 同 prompt 跑 3 次 | 输出完全一致 | 有差异 |
| max_tokens=10 | 设置后检查输出长度 | ≤ 10 tokens | 超出 |
| stop sequence | 设 stop="STOP" | 在 stop 词处停止 | 未停止 |

### 8. test_tool_calling — 工具调用

- 定义简单 function schema (get_weather)
- 发送触发工具调用的 prompt
- 检查返回格式 + LLM Judge 评估参数合理性

**评判：**
```
tool_calls 存在 + 格式正确 + 参数合理  → PASS
tool_calls 存在但格式/参数有误         → WARN
无 tool_calls 或完全错误               → FAIL
```

### 9. test_concurrency — 并发处理

- 同时发 3 个请求
- 对比并发 vs 单次延迟

**评判：**
```
max_latency < 2 * single_latency   → PASS "真并发"
max_latency < 5 * single_latency   → WARN "疑似排队"
max_latency >= 5 * single_latency  → FAIL "串行化"
```

## LLM-as-Judge 评估层

### 原则

- 裁判不能是被告：测 gpt-4o 时不能用同一中转商的 gpt-4o 判
- 优先用原厂 key 的小模型（gpt-4o-mini）当 judge
- 每次 judge 调用 ~150 tokens，成本可忽略

### Judge Prompt 模板

针对不同维度有专用模板：
- `quality_code`: 评估代码正确性和质量
- `quality_reasoning`: 评估推理能力
- `quality_chinese`: 评估中文翻译和赏析
- `identity_fingerprint`: 检测身份伪装
- `tool_calling_quality`: 评估工具调用参数合理性

所有模板统一输出格式：`PASS/WARN/FAIL + 一句话理由`

### 评估流程

```
测试执行 → 收集 raw output
    ├─ 规则可判的 → 直接出 PASS/WARN/FAIL
    └─ 需要 LLM 判的 → 发给 Judge → verdict + reason → 写入报告
```

## pytest 参数化

### conftest.py 核心设计

```python
@pytest.fixture(scope="session")
def providers():
    """加载 providers.yaml，展开环境变量"""

@pytest.fixture(scope="session")
def budget():
    """全局预算守门员"""

@pytest.fixture
def proxy_client(providers, model_entry):
    """按 model.protocol 返回对应协议的中转商 client"""

@pytest.fixture
def official_client(providers, model_entry):
    """has_official=true 时返回原厂 client，否则 None"""

def pytest_generate_tests(metafunc):
    """把 models.yaml 里的每个 model 展开为一条 test case"""
```

### 运行方式

```bash
# 全量测试
uv run pytest

# 测单个模型
uv run pytest -k "gpt-4o"

# 测单个维度
uv run pytest tests/test_identity.py

# 生成 HTML 报告
uv run pytest --html=reports/report.html
```

## 报告格式

### Markdown 报告

```markdown
# API 质量测试报告 — {provider_name} ({date})

## 总览
| 模型 | 身份 | 上下文 | 质量 | 稳定 | 流式 | 计费 | 参数 | 工具 | 并发 |
|------|------|--------|------|------|------|------|------|------|------|
| gpt-4o | ✅ | ⚠️ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |

## 详细结果
### {model_name}
- 身份: {verdict} ({reason})
- 上下文: {verdict} 声称 {claimed}，实测 {actual}
- ...

## 测试元数据
- 总耗时: {duration}
- 总花费: ${cost}
- 测试时间: {timestamp}
```

## 成本估算（省钱模式）

| 维度 | tokens/模型 | 说明 |
|------|-------------|------|
| identity | ~2K | 3 个探针 + A/B 对照 |
| context | ~50K | 二分 3-4 轮 |
| quality | ~2K | 3 个 prompt |
| stability | ~5K | 5 轮重复 |
| streaming | ~0.2K | 1 次流式请求 |
| billing | ~0.01K | 极短 prompt |
| parameters | ~0.5K | 3 个参数测试 |
| tool_calling | ~0.1K | 1 次工具调用 |
| concurrency | ~3K | 3 个并发请求 |
| judge | ~0.75K | ~5 次 judge 调用 |
| **合计** | **~63K** | **约 $0.15-0.20/模型** |
