import time
import json
import httpx
from .base import BaseClient, UnifiedResponse, StreamResult, StreamChunk


class AnthropicCompatClient(BaseClient):
    def __init__(self, base_url: str, api_key: str, messages_path: str = "/v1/messages"):
        super().__init__(base_url, api_key)
        self.messages_path = messages_path
        self._http = httpx.Client(timeout=120.0)

    def chat(self, model: str, messages: list[dict], **kwargs) -> UnifiedResponse:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }
        resp = self._http.post(
            f"{self.base_url}{self.messages_path}",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        text = "".join(b["text"] for b in content_blocks if b["type"] == "text")
        usage = data.get("usage", {})
        input_t = usage.get("input_tokens", 0)
        output_t = usage.get("output_tokens", 0)
        return UnifiedResponse(
            model=data.get("model", model),
            content=text,
            usage={"prompt": input_t, "completion": output_t, "total": input_t + output_t},
            latency_ms=resp.elapsed.total_seconds() * 1000,
            raw=data,
            finish_reason=data.get("stop_reason", "unknown"),
            tool_calls=None,
        )

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> StreamResult:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "stream": True,
            **kwargs,
        }
        chunks = []
        start = time.perf_counter()
        first_chunk_time = None

        with self._http.stream(
            "POST",
            f"{self.base_url}{self.messages_path}",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["type"] == "content_block_delta":
                    text = data["delta"].get("text", "")
                    if text:
                        now = time.perf_counter()
                        if first_chunk_time is None:
                            first_chunk_time = now
                        chunks.append(StreamChunk(content=text, timestamp_ms=(now - start) * 1000))

        total_time = (time.perf_counter() - start) * 1000
        return StreamResult(
            chunks=chunks,
            model=model,
            latency_first_chunk_ms=(first_chunk_time - start) * 1000 if first_chunk_time else total_time,
            latency_total_ms=total_time,
        )
