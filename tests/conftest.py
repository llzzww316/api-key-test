import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from apitest.config import load_providers, load_models, load_judge
from apitest.clients import OpenAICompatClient, AnthropicCompatClient
from apitest.judge import JudgeClient


def _get_proxy_config(providers: dict, name: str) -> dict:
    """Look up a proxy by name in proxies dict or officials dict."""
    proxies = providers.get("proxies", {})
    if name in proxies:
        return proxies[name]
    officials = providers.get("officials", {})
    if name in officials:
        return officials[name]
    # Backward compat: old format with single "proxy"
    if "proxy" in providers:
        return providers["proxy"]
    pytest.exit(f"Provider '{name}' not found in config. Check providers.yaml")


def _make_client(cfg: dict, protocol: str):
    if protocol == "anthropic_compat":
        return AnthropicCompatClient(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            messages_path=cfg.get("anthropic_compat", "/v1/messages"),
        )
    else:
        return OpenAICompatClient(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            chat_path=cfg.get("openai_compat", "/v1/chat/completions"),
        )


# ── Session-scoped fixtures ──

@pytest.fixture(scope="session")
def providers():
    path = Path("config/providers.yaml")
    if not path.exists():
        pytest.exit("config/providers.yaml not found. Copy from config/providers.example.yaml")
    return load_providers(path)


@pytest.fixture(scope="session")
def model_list():
    path = Path("config/models.yaml")
    if not path.exists():
        pytest.exit("config/models.yaml not found. Copy from config/models.example.yaml")
    return load_models(path)


@pytest.fixture(scope="session")
def judge_default(providers):
    """Default judge config from judge.yaml — shared session fixture."""
    cfg = load_judge()
    default_provider = cfg.get("default_provider", "")
    model = cfg["model"]
    cfg_found = _get_proxy_config(providers, default_provider) if default_provider else None
    if cfg_found:
        return {"cfg": cfg_found, "model": model}
    proxies = providers.get("proxies", {})
    first = next(iter(proxies.values()), providers.get("proxy", {}))
    return {"cfg": first, "model": model}


@pytest.fixture
def proxy_client(providers, model_entry):
    """Client for the model being tested — selected by model_entry.provider."""
    proxy_name = model_entry.get("provider", "")
    cfg = _get_proxy_config(providers, proxy_name) if proxy_name else _get_proxy_config(providers, "")
    return _make_client(cfg, model_entry["protocol"])


@pytest.fixture
def judge(providers, model_entry, judge_default):
    """Judge client — per-model, can use a different proxy for cross-evaluation."""
    judge_provider = model_entry.get("judge_provider", "")
    model = judge_default["model"]

    if judge_provider:
        cfg = _get_proxy_config(providers, judge_provider)
        # Pick protocol: use openai_compat if available, else anthropic_compat
        if cfg.get("openai_compat"):
            protocol = "openai_compat"
        elif cfg.get("anthorpic_compat"):
            protocol = "anthropic_compat"
        else:
            protocol = "openai_compat"
        return JudgeClient(
            api_key=cfg["api_key"], base_url=cfg["base_url"], model=model, protocol=protocol
        )

    # Per-model judge not specified — try to auto-cross: pick ANY other proxy
    proxies = providers.get("proxies", {})
    tested_provider = model_entry.get("provider", "")
    others = [k for k in proxies if k != tested_provider]
    if others:
        cfg = proxies[others[0]]
        protocol = "openai_compat" if cfg.get("openai_compat") else "anthropic_compat"
        return JudgeClient(
            api_key=cfg["api_key"], base_url=cfg["base_url"], model=model, protocol=protocol
        )

    # Only one proxy available — use default judge (same as proxy, not ideal but works)
    cfg = judge_default["cfg"]
    protocol = "openai_compat" if cfg.get("openai_compat") else "anthropic_compat"
    return JudgeClient(
        api_key=cfg["api_key"], base_url=cfg["base_url"], model=model, protocol=protocol
    )


@pytest.fixture
def official_client(providers, model_entry):
    if not model_entry.get("has_official"):
        return None
    official_provider = model_entry.get("official_provider")
    if not official_provider:
        return None
    official = providers.get("officials", {}).get(official_provider)
    if not official:
        return None
    compat = official.get("compat", "openai")
    if compat == "anthropic":
        return AnthropicCompatClient(
            base_url=official["base_url"],
            api_key=official["api_key"],
        )
    else:
        return OpenAICompatClient(
            base_url=official["base_url"],
            api_key=official["api_key"],
        )


# ── Parameterization ──

def pytest_generate_tests(metafunc):
    if "model_entry" in metafunc.fixturenames:
        path = Path("config/models.yaml")
        if path.exists():
            models = load_models(path)
            # Unique ID: "provider/model_id" to avoid duplicate model IDs across providers
            metafunc.parametrize(
                "model_entry", models,
                ids=lambda m: f"{m.get('provider', '?')}/{m['id']}"
            )


# ── Custom markers ──

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_if_no_official: skip test when model has no official counterpart"
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("skip_if_no_official"):
        model_entry = item.callspec.params.get("model_entry", {})
        if not model_entry.get("has_official"):
            pytest.skip("No official API key for this model")


# ── Report collection & generation ──

from apitest.reporter import generate_report

_results = []


@pytest.fixture(autouse=True)
def _collect_results(request, model_entry):
    """Auto-collect test results for reporting."""
    # Capture model_entry while it's still alive
    _entry = {"id": model_entry["id"], "provider": model_entry.get("provider", "")}
    yield
    if "model_entry" not in request.fixturenames:
        return  # skip unit tests that don't use model parameterization

    outcome = "PASS"
    reason = ""
    if hasattr(request.node, "rep_call"):
        if request.node.rep_call.failed:
            outcome = "FAIL"
            if request.node.rep_call.longrepr:
                reason = str(request.node.rep_call.longrepr)[:200]
        elif request.node.rep_call.skipped:
            reason = "skipped (no official key or budget)"

    _results.append({
        "model": _entry["id"],
        "provider": _entry.get("provider", ""),
        "test_name": request.node.originalname if hasattr(request.node, "originalname") else request.node.name,
        "verdict": outcome,
        "reason": reason,
    })


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


def pytest_sessionfinish(session, exitstatus):
    if _results:
        # Use provider name from first model entry
        provider = _results[0].get("provider", "unknown") if _results else "unknown"
        report_path = generate_report(_results, provider)
        print(f"\nReport: {report_path}")
