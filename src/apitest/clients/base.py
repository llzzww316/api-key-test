from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class UnifiedResponse:
    model: str
    content: str
    usage: dict
    latency_ms: float
    raw: dict
    finish_reason: str
    tool_calls: list | None = None


@dataclass
class StreamChunk:
    content: str
    timestamp_ms: float


@dataclass
class StreamResult:
    chunks: list[StreamChunk] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency_first_chunk_ms: float = 0.0
    latency_total_ms: float = 0.0


class BaseClient(ABC):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @abstractmethod
    def chat(self, model: str, messages: list[dict], **kwargs) -> UnifiedResponse:
        ...

    @abstractmethod
    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> StreamResult:
        ...
