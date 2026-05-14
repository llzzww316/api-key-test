import time
import json
import httpx
from .base import BaseClient, UnifiedResponse, StreamResult, StreamChunk


class OpenAICompatClient(BaseClient):
    def __init__(self, base_url: str, api_key: str, chat_path: str = "/v1/chat/completions"):
        super().__init__(base_url, api_key)
        self.chat_path = chat_path
        self._http = httpx.Client(timeout=120.0)

    def chat(self, model: str, messages: list[dict], **kwargs) -> UnifiedResponse:
        payload = {"model": model, "messages": messages, **kwargs}
        resp = self._http.post(
            f"{self.base_url}{self.chat_path}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return UnifiedResponse(
            model=data.get("model", model),
            content=choice["message"]["content"],
            usage={
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            },
            latency_ms=resp.elapsed.total_seconds() * 1000,
            raw=data,
            finish_reason=choice.get("finish_reason", "unknown"),
            tool_calls=choice["message"].get("tool_calls"),
        )

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> StreamResult:
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        chunks = []
        start = time.perf_counter()
        first_chunk_time = None

        with self._http.stream(
            "POST",
            f"{self.base_url}{self.chat_path}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = now
                    chunks.append(StreamChunk(content=content, timestamp_ms=(now - start) * 1000))

        total_time = (time.perf_counter() - start) * 1000
        return StreamResult(
            chunks=chunks,
            model=model,
            latency_first_chunk_ms=(first_chunk_time - start) * 1000 if first_chunk_time else total_time,
            latency_total_ms=total_time,
        )
