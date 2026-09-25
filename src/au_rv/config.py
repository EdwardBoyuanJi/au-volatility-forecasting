from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    load_dotenv = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    config_path = Path(path) if path else root / "config.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    if load_dotenv is not None:
        load_dotenv(root / ".env", override=False)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(root)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path


def ensure_project_directories(config: dict[str, Any]) -> None:
    root = Path(config["_project_root"])
    for relative in (
        "data/raw",
        "data/interim",
        "data/processed",
        "data/outputs",
        "data/examples",
        "models",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def require_environment(names: list[str]) -> dict[str, str]:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {joined}. "
            "Copy .env.example to .env and add the real credentials."
        )
    return {name: os.environ[name] for name in names}


def config_fingerprint(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def git_commit(root: str | Path | None = None) -> str:
    working = Path(root) if root else project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # Archive deliveries may not contain .git. In that case preserve a
        # deterministic fingerprint of the executable source and core config.
        digest = hashlib.sha256()
        candidates = [
            working / "config.yaml",
            working / "requirements.txt",
            working / "requirements-lock.txt",
            *sorted((working / "src").rglob("*.py")),
            *sorted((working / "scripts").rglob("*.py")),
        ]
        for path in candidates:
            if not path.is_file():
                continue
            digest.update(path.relative_to(working).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return f"source-sha256:{digest.hexdigest()[:16]}"
