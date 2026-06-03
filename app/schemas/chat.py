"""Unified chat request/response schemas (provider-agnostic).

The request shape mirrors the OpenAI chat-completions format; provider adapters
translate to/from their native formats.
"""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    model: str = Field(..., description="e.g. 'gpt-4o-mini' or 'claude-3-5-sonnet-latest'")
    messages: list[Message] = Field(..., min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False


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
