"""OpenAI-compatible HTTP server (engine=hf, in-process Transformers backend).

Endpoints (OpenAI protocol + platform extras):

- ``GET  /v1/models``
- ``POST /v1/completions``          (stream / non-stream)
- ``POST /v1/chat/completions``     (stream / non-stream)
- ``GET  /health``
- ``GET  /metrics``                 (Prometheus text format)

This server is the platform's own Transformers baseline: it lets the full
OpenAI protocol be exercised without a GPU, and gives the Benchmark system
(M3) a fair "no-optimization" comparison target that speaks the same API as
vLLM / SGLang.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel

from minillm.serve.config import ServeConfig
from minillm.serve.engine.hf import HFAdapter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- metrics

REQUESTS = Counter("minillm_requests_total", "Requests received", ["endpoint", "engine"])
TOKENS = Counter("minillm_tokens_generated_total", "Tokens generated", ["engine"])
DURATION = Histogram(
    "minillm_request_duration_seconds", "End-to-end request latency",
    ["endpoint"], buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
TTFT = Histogram(
    "minillm_ttft_seconds", "Time to first token",
    ["endpoint"], buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
GPU_MEM_USED = Gauge("minillm_gpu_mem_used_bytes", "GPU memory used", ["engine"])
GPU_MEM_TOTAL = Gauge("minillm_gpu_mem_total_bytes", "GPU memory total", ["engine"])
MODEL_LOADED = Gauge("minillm_model_loaded", "1 if the serving model is loaded", ["engine", "model"])


# ---------------------------------------------------------------- schemas

class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


# ---------------------------------------------------------------- helpers

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _openai_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error"}})


class OpenAICompatServer:
    """Builds the FastAPI app around a given engine adapter."""

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.engine = HFAdapter(
            cfg.model.name_or_path,
            dtype=cfg.model.dtype,
            attn_impl=cfg.model.attn_impl,
            tokenizer_name=cfg.model.tokenizer_name,
            trust_remote_code=cfg.model.trust_remote_code,
            max_model_len=cfg.model.max_model_len,
        )
        MODEL_LOADED.labels(engine="hf", model=cfg.model.name_or_path).set(1)

    # ------------------------------------------------------------ endpoints

    def _check_model(self, model: str) -> None:
        if model != self.cfg.model.name_or_path:
            raise HTTPException(
                status_code=400,
                detail=f"model {model!r} not served here (serving {self.cfg.model.name_or_path!r})",
            )

    def list_models(self) -> dict:
        return {
            "object": "list",
            "data": [{"id": self.cfg.model.name_or_path, "object": "model", "owned_by": "minillm"}],
        }

    def completions(self, req: CompletionRequest) -> Response:
        self._check_model(req.model)
        if req.stream:
            return self._stream_completions(req)
        return self._non_stream(self._run_completions, req)

    def chat(self, req: ChatRequest) -> Response:
        self._check_model(req.model)
        if req.stream:
            return self._stream_chat(req)
        return self._non_stream(self._run_chat, req)

    def health(self) -> dict:
        return {"status": "ok", "engine": "hf", "model": self.cfg.model.name_or_path}

    def metrics(self) -> Response:
        from prometheus_client import generate_latest

        return Response(generate_latest(), media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------ internals

    def _run_completions(self, req: CompletionRequest) -> dict:
        result = self.engine.completions(
            req.prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p
        )
        return self._completion_response(req.prompt, result)

    def _run_chat(self, req: ChatRequest) -> dict:
        result = self.engine.chat(
            [m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p,
        )
        return self._chat_response(result)

    def _completion_response(self, prompt: str, result) -> dict:
        created = int(time.time())
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": created,
            "model": self.cfg.model.name_or_path,
            "choices": [{"index": 0, "text": result.text, "finish_reason": result.finish_reason}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    def _chat_response(self, result) -> dict:
        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": self.cfg.model.name_or_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    def _non_stream(self, runner, req) -> Response:
        endpoint = "chat" if isinstance(req, ChatRequest) else "completions"
        REQUESTS.labels(endpoint=endpoint, engine="hf").inc()
        with DURATION.labels(endpoint=endpoint).time():
            try:
                body = runner(req)
            except Exception as e:  # noqa: BLE001
                log.exception("generation failed")
                raise HTTPException(status_code=500, detail=str(e)) from e
        TOKENS.labels(engine="hf").inc(body["usage"]["completion_tokens"])
        return JSONResponse(body)

    def _stream_completions(self, req: CompletionRequest) -> StreamingResponse:
        endpoint = "completions"
        REQUESTS.labels(endpoint=endpoint, engine="hf").inc()
        streamer = self.engine.completions(
            req.prompt, max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, stream=True,
        )
        created = int(time.time())
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        first = {"sent": False}
        total_tokens = 0

        def gen():
            nonlocal total_tokens
            with TTFT.labels(endpoint=endpoint).time():
                try:
                    for piece in streamer:
                        if not first["sent"]:
                            first["sent"] = True
                        total_tokens += 1
                        yield _sse({
                            "id": req_id, "object": "text_completion", "created": created,
                            "model": self.cfg.model.name_or_path,
                            "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                        })
                except Exception as e:  # noqa: BLE001
                    log.exception("streaming failed")
                    yield _sse({"error": {"message": str(e)}})
                    return
            yield _sse({
                "id": req_id, "object": "text_completion", "created": created,
                "model": self.cfg.model.name_or_path,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            })
            yield "data: [DONE]\n\n"
            TOKENS.labels(engine="hf").inc(total_tokens)
            DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - t0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    def _stream_chat(self, req: ChatRequest) -> StreamingResponse:
        endpoint = "chat"
        REQUESTS.labels(endpoint=endpoint, engine="hf").inc()
        streamer = self.engine.chat(
            [m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p, stream=True,
        )
        created = int(time.time())
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        total_tokens = 0

        def gen():
            nonlocal total_tokens
            with TTFT.labels(endpoint=endpoint).time():
                try:
                    for piece in streamer:
                        total_tokens += 1
                        yield _sse({
                            "id": req_id, "object": "chat.completion.chunk", "created": created,
                            "model": self.cfg.model.name_or_path,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        })
                except Exception as e:  # noqa: BLE001
                    log.exception("streaming failed")
                    yield _sse({"error": {"message": str(e)}})
                    return
            yield _sse({
                "id": req_id, "object": "chat.completion.chunk", "created": created,
                "model": self.cfg.model.name_or_path,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            yield "data: [DONE]\n\n"
            TOKENS.labels(engine="hf").inc(total_tokens)
            DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - t0)

        return StreamingResponse(gen(), media_type="text/event-stream")


def build_app(cfg: ServeConfig) -> FastAPI:
    """Create the FastAPI app for the given serving config."""
    server = OpenAICompatServer(cfg)
    app = FastAPI(title=f"minillm {cfg.engine} server", version="0.1.0")

    @app.get("/v1/models")
    def models():  # pragma: no cover - thin wrapper
        return server.list_models()

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        return server.completions(req)

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        return server.chat(req)

    @app.get("/health")
    def health():
        return server.health()

    @app.get("/metrics")
    def metrics():
        return server.metrics()

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException):  # noqa: ARG001
        return _openai_error(exc.status_code, str(exc.detail))

    return app
