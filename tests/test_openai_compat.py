"""Offline unit tests for the OpenAI-compatible wire translation layer.

Covers ``to_internal_chat_request`` (developer->system; tool/function ->
ValueError; multimodal list -> ValueError; n>1 -> ValueError;
``max_completion_tokens`` fallback), the ``build_*`` response/chunk shapes, and
``wants_stream_usage``. Pure functions, no I/O.
"""

import pytest

from app.schemas.chat import ChatResponse, Usage
from app.schemas.openai_compat import (
    OAIChatCompletionRequest,
    build_chunk,
    build_completion_response,
    build_role_chunk,
    build_usage_chunk,
    to_internal_chat_request,
)


def _req(**kw) -> OAIChatCompletionRequest:
    base = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(kw)
    return OAIChatCompletionRequest(**base)


# --- to_internal_chat_request ----------------------------------------------


def test_developer_role_folds_to_system():
    internal = to_internal_chat_request(
        _req(messages=[{"role": "developer", "content": "rules"}, {"role": "user", "content": "x"}])
    )
    assert internal.messages[0].role == "system"
    assert internal.messages[0].content == "rules"
    assert internal.messages[1].role == "user"


@pytest.mark.parametrize("role", ["tool", "function"])
def test_tool_function_roles_raise(role):
    with pytest.raises(ValueError):
        to_internal_chat_request(_req(messages=[{"role": role, "content": "x"}]))


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        to_internal_chat_request(_req(messages=[{"role": "wizard", "content": "x"}]))


def test_multimodal_list_content_raises():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with pytest.raises(ValueError):
        to_internal_chat_request(_req(messages=msgs))


def test_n_greater_than_one_raises():
    with pytest.raises(ValueError):
        to_internal_chat_request(_req(n=2))


def test_n_one_ok():
    internal = to_internal_chat_request(_req(n=1))
    assert len(internal.messages) == 1


def test_max_completion_tokens_fallback():
    internal = to_internal_chat_request(_req(max_completion_tokens=42))
    assert internal.max_tokens == 42


def test_max_tokens_takes_precedence_over_completion():
    internal = to_internal_chat_request(_req(max_tokens=10, max_completion_tokens=42))
    assert internal.max_tokens == 10


def test_none_content_becomes_empty_string():
    internal = to_internal_chat_request(_req(messages=[{"role": "user", "content": None}]))
    assert internal.messages[0].content == ""


def test_passthrough_sampling_params():
    internal = to_internal_chat_request(
        _req(temperature=0.3, top_p=0.7, stop=["X"], seed=9, stream=True)
    )
    assert internal.temperature == 0.3
    assert internal.top_p == 0.7
    assert internal.stop == ["X"]
    assert internal.seed == 9
    assert internal.stream is True


# --- wants_stream_usage -----------------------------------------------------


def test_wants_stream_usage_true():
    assert _req(stream_options={"include_usage": True}).wants_stream_usage() is True


def test_wants_stream_usage_false_variants():
    assert _req().wants_stream_usage() is False
    assert _req(stream_options={"include_usage": False}).wants_stream_usage() is False
    assert _req(stream_options={}).wants_stream_usage() is False


# --- build_completion_response ----------------------------------------------


def _resp() -> ChatResponse:
    return ChatResponse(
        id="cmpl-9",
        provider="openai",
        model="gpt-4o-mini",
        content="hello",
        usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        finish_reason="stop",
    )


def test_build_completion_response_shape():
    out = build_completion_response(_resp(), created=1700000000)
    assert out["id"] == "cmpl-9"
    assert out["object"] == "chat.completion"
    assert out["created"] == 1700000000
    assert out["model"] == "gpt-4o-mini"
    assert out["choices"][0]["index"] == 0
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_build_completion_response_defaults_finish_reason():
    resp = _resp()
    resp.finish_reason = None
    out = build_completion_response(resp, created=0)
    assert out["choices"][0]["finish_reason"] == "stop"


# --- build_chunk / build_role_chunk / build_usage_chunk ---------------------


def test_build_chunk_with_content():
    out = build_chunk("hi", id="c1", created=1, model="gpt-4o-mini")
    assert out["object"] == "chat.completion.chunk"
    assert out["choices"][0]["delta"] == {"content": "hi"}
    assert out["choices"][0]["finish_reason"] is None


def test_build_chunk_empty_text_gives_empty_delta():
    out = build_chunk("", id="c1", created=1, model="gpt-4o-mini", finish_reason="stop")
    assert out["choices"][0]["delta"] == {}
    assert out["choices"][0]["finish_reason"] == "stop"


def test_build_role_chunk_shape():
    out = build_role_chunk(id="c1", created=2, model="gpt-4o-mini")
    assert out["object"] == "chat.completion.chunk"
    assert out["choices"][0]["delta"] == {"role": "assistant"}
    assert out["choices"][0]["finish_reason"] is None


def test_build_usage_chunk_shape():
    out = build_usage_chunk(
        id="c1",
        created=3,
        model="gpt-4o-mini",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    assert out["object"] == "chat.completion.chunk"
    assert out["choices"] == []
    assert out["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
