from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from bruteforce_canvas.prompt import (
    ActionLane,
    CanonicalEnum,
    ConstraintLane,
    ObjectLane,
    VerificationReport,
)
from bruteforce_canvas.prompt_models import CinematographyLane, PromptDocumentSpec


OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL = "https://dataphysician--mellum2-vllm-openai.modal.run/v1"
OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL = "mellum2-thinking"
OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS = 600.0


_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "PromptDocumentSpec": PromptDocumentSpec,
    "PromptDocumentSpecRepair": PromptDocumentSpec,
    "CanonicalEnum": CanonicalEnum,
    "VerificationReport": VerificationReport,
    "ObjectLane": ObjectLane,
    "ActionLane": ActionLane,
    "CinematographyLane": CinematographyLane,
    "ConstraintLane": ConstraintLane,
}


@dataclass(frozen=True)
class OpenAICompatibleServerJsonLLMClient:
    """JSON adapter for OpenAI-compatible chat-completion servers.

    This adapter is intentionally server-scoped: local model adapters
    should be implemented directly in the repository, while arbitrary
    cloud/server deployments can be exercised by pointing this adapter
    at their ``/v1`` endpoint. The adapter sends the target Pydantic
    schema through OpenAI-style ``response_format.type=json_schema``
    structured decoding, then validates the returned JSON object locally.
    """

    base_url: str
    model: str = OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL
    api_key: str | None = None
    timeout_seconds: float = OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS
    max_completion_tokens: int = 2048
    temperature: float = 0.0
    structured_decoding: bool = True

    def generate_json(self, *, system: str, user: dict, schema_name: str) -> dict:
        schema = _schema_for(schema_name)
        user_payload: dict[str, Any] = {
            "schema_name": schema_name,
            "input": user,
        }
        if not self.structured_decoding:
            user_payload["json_schema"] = schema
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system}\n\n"
                        f"Return only a JSON object matching {schema_name}. "
                        "Do not include markdown, commentary, or hidden reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": _response_format(schema_name, schema, structured=self.structured_decoding),
        }

        response = _post_json(
            _chat_completions_url(self.base_url),
            payload,
            api_key=self.api_key or os.environ.get("BC_LLM_API_KEY"),
            timeout_seconds=self.timeout_seconds,
        )
        content = _choice_message_content(response)
        parsed = _parse_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"{schema_name} response must be a JSON object")
        return parsed


def _schema_for(schema_name: str) -> dict[str, Any]:
    model = _SCHEMA_MODELS.get(schema_name)
    if model is None:
        raise ValueError(f"unsupported LLM JSON schema: {schema_name}")
    return model.model_json_schema()


def _response_format(schema_name: str, schema: dict[str, Any], *, structured: bool) -> dict[str, Any]:
    if not structured:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }


def _chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM server returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach LLM server at {url}: {error.reason}") from error
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeError(
            f"LLM server timed out after {timeout_seconds:g}s waiting for {url}. "
            "Increase BC_LLM_TIMEOUT_SECONDS or BC_VLM_TIMEOUT_SECONDS for cloud GPU cold starts."
        ) from error

    try:
        parsed = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("LLM server returned non-JSON response") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM server response must be a JSON object")
    return parsed


def _choice_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not contain choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM response choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response choice did not contain a message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ]
        return "".join(text_parts)
    raise RuntimeError("LLM response message content must be text")


def _parse_json_object(content: str) -> object:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                value, _end = decoder.raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
        raise ValueError("LLM response did not contain a JSON object")


OpenAICompatibleJsonLLMClient = OpenAICompatibleServerJsonLLMClient
