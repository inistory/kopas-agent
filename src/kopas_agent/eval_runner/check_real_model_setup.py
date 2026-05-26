from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import yaml

PROVIDER_ENV_KEYS = {
    "openai/": ["OPENAI_API_KEY"],
    "anthropic/": ["ANTHROPIC_API_KEY"],
    "gemini/": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google/": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "together_ai/": ["TOGETHERAI_API_KEY"],
    "openrouter/": ["OPENROUTER_API_KEY"],
    "deepseek/": ["DEEPSEEK_API_KEY"],
    "mistral/": ["MISTRAL_API_KEY"],
}


def _provider_env_keys(model_name: str) -> list[str]:
    for prefix, env_keys in PROVIDER_ENV_KEYS.items():
        if model_name.startswith(prefix):
            return env_keys
    return []


def check_setup(config_path: Path) -> list[str]:
    messages: list[str] = []
    _load_env_file(Path(".env"))

    if importlib.util.find_spec("litellm") is None:
        messages.append("warning: litellm is not installed; adapter='litellm_json' will not run")
    else:
        messages.append("ok: litellm is installed")

    if not config_path.exists():
        messages.append(f"missing_config: {config_path} does not exist")
        messages.append(
            "hint: copy configs/real_models.example.yaml "
            "to configs/real_models.yaml"
        )
        return messages

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_configs = config.get("models", [])
    if not model_configs:
        messages.append("missing_models: config has no models")
        return messages

    for model_config in model_configs:
        messages.extend(_check_model_config(model_config))

    return messages


def _check_model_config(model_config: dict[str, Any]) -> list[str]:
    model_id = str(model_config.get("model_id", "<missing-model-id>"))
    adapter = model_config.get("adapter")
    model_name = str(model_config.get("model", ""))
    messages = []

    if adapter not in {"litellm_json", "openrouter_json"}:
        messages.append(f"{model_id}: skipped non-real adapter {adapter!r}")
        return messages

    if not model_name or "<replace-with-model>" in model_name:
        messages.append(f"{model_id}: missing_model_name")
        return messages

    if adapter == "openrouter_json":
        env_key = str(model_config.get("api_key_env", "OPENROUTER_API_KEY"))
        if os.environ.get(env_key):
            messages.append(
                f"{model_id}: ok credentials present for OpenRouter model {model_name!r}"
            )
        else:
            messages.append(f"{model_id}: missing_credentials; set {env_key}")
        return messages

    env_keys = _provider_env_keys(model_name)
    if not env_keys:
        messages.append(
            f"{model_id}: unknown_provider_env_for {model_name!r}; "
            "LiteLLM may still work if provider credentials are configured elsewhere"
        )
        return messages

    if any(os.environ.get(key) for key in env_keys):
        messages.append(f"{model_id}: ok credentials present for {model_name!r}")
    else:
        messages.append(
            f"{model_id}: missing_credentials; set one of {', '.join(env_keys)}"
        )
    return messages


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check real-model experiment readiness.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/real_models.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    messages = check_setup(args.model_config)
    for message in messages:
        print(message)
    blocking = [message for message in messages if message.startswith(("missing_",))]
    blocking.extend(
        message for message in messages if ": missing_" in message
    )
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
