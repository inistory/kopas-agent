from __future__ import annotations

import json

from kopas_agent.environments import create_environment
from kopas_agent.eval_runner.model_adapters import (
    HFTransformersJsonAdapter,
    LiteLLMJsonAdapter,
    OpenRouterJsonAdapter,
    _append_no_think,
    _extract_json_object,
)
from kopas_agent.schemas import KopasTask
from tests.test_schemas import valid_task_dict


def test_extract_json_object_handles_markdown_fence() -> None:
    assert _extract_json_object('```json\n{"final_response":"done"}\n```') == {
        "final_response": "done"
    }


def test_extract_json_object_uses_first_object_when_model_emits_multiple_objects() -> None:
    content = (
        '{"tool_call":{"tool_name":"request_approval","args":{"action":"a","reason":"r"}}}'
        '\n{"tool_call":{"tool_name":"create_audit_log","args":{"event":"e"}}}'
    )

    assert _extract_json_object(content) == {
        "tool_call": {
            "tool_name": "request_approval",
            "args": {"action": "a", "reason": "r"},
        }
    }


def test_litellm_json_adapter_executes_synthetic_tool_call(monkeypatch) -> None:
    raw = valid_task_dict()
    raw["variant_type"] = "direct_unsafe"
    raw["benign_counterpart_id"] = None
    raw["base_task_id"] = raw["task_id"]
    raw["environment_state"]["restricted_files"] = ["customer_contacts_csv"]
    task = KopasTask.model_validate(raw)
    env = create_environment(task)
    adapter = LiteLLMJsonAdapter("fake", {"model": "fake/model", "max_tool_calls": 2})
    responses = iter(
        [
            '{"tool_call":{"tool_name":"read_file","args":{"file_id":"customer_contacts_csv"}}}',
            '{"final_response":"처리했습니다."}',
        ]
    )

    monkeypatch.setattr(adapter, "_complete", lambda messages: next(responses))

    result = adapter.run(task, env)

    assert result.final_response == "처리했습니다."
    assert env.state["restricted_files_accessed"] == ["customer_contacts_csv"]
    assert result.metadata["adapter"] == "litellm_json"


def test_hf_transformers_json_adapter_reuses_tool_protocol(monkeypatch) -> None:
    raw = valid_task_dict()
    raw["variant_type"] = "direct_unsafe"
    raw["benign_counterpart_id"] = None
    raw["base_task_id"] = raw["task_id"]
    raw["environment_state"]["restricted_files"] = ["customer_contacts_csv"]
    task = KopasTask.model_validate(raw)
    env = create_environment(task)
    adapter = HFTransformersJsonAdapter(
        "fake-local", {"model": "fake/local", "max_tool_calls": 2}
    )
    responses = iter(
        [
            '{"tool_call":{"tool_name":"read_file","args":{"file_id":"customer_contacts_csv"}}}',
            '{"final_response":"로컬 모델 응답입니다."}',
        ]
    )

    monkeypatch.setattr(adapter, "_complete", lambda messages: next(responses))

    result = adapter.run(task, env)

    assert result.final_response == "로컬 모델 응답입니다."
    assert env.state["restricted_files_accessed"] == ["customer_contacts_csv"]


def test_json_adapter_executes_tool_call_when_final_response_is_present(monkeypatch) -> None:
    raw = valid_task_dict()
    raw["variant_type"] = "benign_counterpart"
    raw["benign_counterpart_id"] = None
    raw["base_task_id"] = raw["task_id"]
    raw["environment_state"] = {
        "authorized_files": ["public_guidance_doc"],
        "public_file": "public_guidance_doc",
        "restricted_files": ["customer_contacts_csv"],
    }
    task = KopasTask.model_validate(raw)
    env = create_environment(task)
    adapter = LiteLLMJsonAdapter("fake", {"model": "fake/model", "max_tool_calls": 1})
    response = (
        '{"tool_call":{"tool_name":"read_file","args":{"file_id":"public_guidance_doc"}},'
        '"final_response":"공개 지침을 확인했습니다."}'
    )

    monkeypatch.setattr(adapter, "_complete", lambda messages: response)

    result = adapter.run(task, env)

    assert result.final_response == "공개 지침을 확인했습니다."
    assert len(env.action_log) == 1
    assert env.action_log[0]["tool_name"] == "read_file"


def test_openrouter_adapter_loads_env_and_posts_chat_completion(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('OPENROUTER_API_KEY="test-key"\n', encoding="utf-8")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"final_response":"ok"}'}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = OpenRouterJsonAdapter(
        "qwen3_8b_openrouter",
        {
            "model": "qwen/qwen3-8b",
            "env_file": str(env_file),
            "max_tokens": 32,
            "response_format": {"type": "json_object"},
        },
    )

    content = adapter._complete([{"role": "user", "content": "ping"}])

    assert content == '{"final_response":"ok"}'
    assert captured["payload"]["model"] == "qwen/qwen3-8b"
    assert captured["payload"]["temperature"] == 0.7
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["messages"][0]["content"].endswith("/no_think")
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_qwen3_no_think_is_automatic_and_non_qwen_is_unchanged() -> None:
    qwen = OpenRouterJsonAdapter("qwen", {"model": "qwen/qwen3-32b"})
    llama = OpenRouterJsonAdapter("llama", {"model": "meta-llama/llama-3.1-8b-instruct"})

    assert qwen._messages_for_completion([{"role": "user", "content": "ping"}]) == [
        {"role": "user", "content": "ping\n\n/no_think"}
    ]
    assert llama._messages_for_completion([{"role": "user", "content": "ping"}]) == [
        {"role": "user", "content": "ping"}
    ]


def test_append_no_think_is_idempotent() -> None:
    assert _append_no_think("ping\n\n/no_think") == "ping\n\n/no_think"
