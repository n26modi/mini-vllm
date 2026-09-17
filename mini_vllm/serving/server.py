"""
HTTP server for mini-vllm. Exposes:
  POST /generate          — single request, returns full completion
  POST /generate/stream   — single request, server-sent events stream
  GET  /health            — liveness check

Run:
    python -m mini_vllm.serving.server
"""
from __future__ import annotations

import json
import time

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from mini_vllm.configs.flags import EngineConfig
from mini_vllm.engine.model_runner import CachedEngine
from mini_vllm.engine.sampler import SampleParams


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="mini-vllm")
    _engine: CachedEngine | None = None

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 128
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = 0

    @app.on_event("startup")
    def _startup():
        global _engine
        config = EngineConfig()
        _engine = CachedEngine(config)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        assert _engine is not None
        params = SampleParams(
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k
        )
        metrics = _engine.generate(req.prompt, req.max_new_tokens, params)
        text = _engine.tokenizer.decode(
            _engine.tokenizer.encode(req.prompt + metrics.prompt)[len(_engine.tokenizer.encode(req.prompt)):],
            skip_special_tokens=True,
        )
        return JSONResponse({
            "text": _engine.tokenizer.decode(
                _engine.tokenizer.encode(req.prompt, add_special_tokens=False),
                skip_special_tokens=True,
            ),
            "ttft_ms": metrics.ttft_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "throughput_tok_s": metrics.throughput_tok_s,
            "total_tokens": metrics.total_tokens,
        })

    @app.post("/generate/stream")
    def generate_stream(req: GenerateRequest):
        assert _engine is not None
        params = SampleParams(
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k
        )

        def _event_stream():
            for tok in _engine.serve_request(req.prompt, req.max_new_tokens, params):
                yield f"data: {json.dumps({'token': tok})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    def run(host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(app, host=host, port=port)

else:
    def run(host="0.0.0.0", port=8000):
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")


if __name__ == "__main__":
    run()
