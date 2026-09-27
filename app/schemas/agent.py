from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    model: str | None = None


class AgentStep(BaseModel):
    step: int
    action: Literal["tool", "final", "error"]
    tool_name: str | None = None
    summary: str


class AgentToolCall(BaseModel):
    step: int
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    output: Any = None
    error: str | None = None


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
