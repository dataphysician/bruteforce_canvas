import json

import pytest

from bruteforce_canvas.llm_clients import (
    OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL,
    OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL,
    OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS,
    OpenAICompatibleJsonLLMClient,
    OpenAICompatibleServerJsonLLMClient,
    _chat_completions_url,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeTimeoutResponse:
    def __enter__(self) -> "FakeTimeoutResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        raise TimeoutError("timed out")


def test_chat_completions_url_accepts_server_root_or_v1_base_url() -> None:
    assert _chat_completions_url("https://llm.example.test") == "https://llm.example.test/v1/chat/completions"
    assert _chat_completions_url("https://llm.example.test/v1") == "https://llm.example.test/v1/chat/completions"
    assert (
        _chat_completions_url("https://llm.example.test/v1/chat/completions")
        == "https://llm.example.test/v1/chat/completions"
    )


def test_openai_compatible_json_client_posts_structured_decoding_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"approved": true, "issues": []}\n```',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bruteforce_canvas.llm_clients.urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleServerJsonLLMClient(
        base_url=OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL,
        api_key="secret",
        timeout_seconds=7,
    )
    result = client.generate_json(system="verify", user={"document": {}}, schema_name="VerificationReport")

    assert result == {"approved": True, "issues": []}
    assert captured["url"] == f"{OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL}/chat/completions"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer secret"
    body = captured["body"]
    assert body["model"] == OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL
    assert body["max_completion_tokens"] == 2048
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "VerificationReport",
            "schema": body["response_format"]["json_schema"]["schema"],
            "strict": True,
        },
    }
    assert body["response_format"]["json_schema"]["schema"]["title"] == "VerificationReport"
    assert "guided_json" not in body
    assert "json_schema" not in json.loads(body["messages"][1]["content"])


def test_openai_compatible_json_client_includes_schema_in_prompt_when_structured_decoding_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"approved": true, "issues": []}',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("bruteforce_canvas.llm_clients.urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleServerJsonLLMClient(
        base_url=OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL,
        structured_decoding=False,
    )
    client.generate_json(system="verify", user={"document": {}}, schema_name="VerificationReport")

    body = captured["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert "guided_json" not in body
    assert json.loads(body["messages"][1]["content"])["json_schema"]["title"] == "VerificationReport"


def test_openai_compatible_json_client_rejects_unknown_schema() -> None:
    client = OpenAICompatibleServerJsonLLMClient(base_url=OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL)

    with pytest.raises(ValueError, match="unsupported LLM JSON schema"):
        client.generate_json(system="x", user={}, schema_name="UnknownSchema")


def test_openai_compatible_json_client_default_timeout_allows_cloud_gpu_cold_start() -> None:
    client = OpenAICompatibleServerJsonLLMClient(base_url=OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL)

    assert client.timeout_seconds == OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS
    assert client.timeout_seconds >= 600


def test_openai_compatible_json_client_reports_read_timeout_with_env_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeTimeoutResponse:
        return FakeTimeoutResponse()

    monkeypatch.setattr("bruteforce_canvas.llm_clients.urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleServerJsonLLMClient(
        base_url=OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL,
        timeout_seconds=3,
    )

    with pytest.raises(RuntimeError, match="timed out after 3s"):
        client.generate_json(system="verify", user={"document": {}}, schema_name="VerificationReport")


def test_openai_compatible_json_client_alias_points_to_server_adapter() -> None:
    assert OpenAICompatibleJsonLLMClient is OpenAICompatibleServerJsonLLMClient
