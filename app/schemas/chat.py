"""Unified chat request/response schemas (provider-agnostic).

The request shape mirrors the OpenAI chat-completions format; provider adapters
translate to/from their native formats.

All additions here are *additive*: the original ``Usage``/``ChatResponse`` field
names are unchanged so stored rows and existing clients stay compatible.
"""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    model: str = Field(
        ..., description="e.g. 'gpt-4o-mini', 'claude-3-5-sonnet-latest', "
        "'gemini-1.5-flash', or 'azure/<deployment>'"
    )
    messages: list[Message] = Field(..., min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False

    # --- Optional sampling passthrough (mapped per-provider; ignored where unsupported) ---
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = Field(default=None)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = Field(default=None)

    # --- Gateway features (additive) ---
    fallback_models: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Models to try, in order, if the primary fails.",
    )
    cache: bool | None = Field(
        default=None,
        description="Opt in/out of the response cache for this request "
        "(only effective for deterministic, non-streaming calls).",
    )

    def stop_sequences(self) -> list[str]:
        """Normalise ``stop`` into a list (possibly empty)."""
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    provider: str
    model: str
    content: str
    usage: Usage
    cost_usd: float = 0.0

    # --- Additive metadata ---
    finish_reason: str | None = None
    cached: bool = False
    fallback_used: bool = False
    latency_ms: int = 0


class StreamChunk(BaseModel):
    """One streamed delta from a provider.

    Adapters yield content chunks (``text`` set) followed by exactly one
    terminal chunk carrying the final :class:`Usage` (its ``text`` may be ``""``).
    Usage values are absolute / last-wins — never re-summed by consumers.
    """

    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    model: str | None = None
