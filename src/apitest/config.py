import os
import re
from pathlib import Path
import yaml


def _expand_env_vars(obj):
    if isinstance(obj, str):
        return re.sub(
            r'\$\{(\w+)\}',
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
