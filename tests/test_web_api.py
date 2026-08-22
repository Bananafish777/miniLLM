"""Web admin console tests: API surface + real-engine integration."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from minillm.serve.config import ServeConfig
from minillm.serve.server import build_app as build_serve_app
from minillm.web.app import build_web_app
from minillm.web.config import WebConfig

MODEL = "data/models/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------- fixtures

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_report(dirpath: Path, experiment: str) -> Path:
    out = dirpath / experiment
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "timestamp": "2026-08-20 12:00:00",
        "metrics": [
            {"engine": "vllm", "model": "m", "concurrency": 8, "input_tokens": 128,
             "output_tokens": 128, "throughput_tps": 400.0, "ttft_p99": 0.05,
             "success_rate": 1.0},
            {"engine": "hf", "model": "m", "concurrency": 8, "input_tokens": 128,
             "output_tokens": 128, "throughput_tps": 100.0, "ttft_p99": 0.5,
             "success_rate": 1.0},
        ],
        "findings": [{"severity": "info", "message": "vllm 吞吐是 hf 的 4.0×"}],
    }
    path = out / "bench_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _make_train_run(dirpath: Path, name: str) -> Path:
    out = dirpath / name
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": name,
        "finetune_mode": "lora",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "device": "cuda",
        "train_metrics": {"eval_loss": 1.234, "train_tokens_per_second": 1234.5},
        "export": {"export_dir": str(out / "export")},
    }
    path = out / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return out


@pytest.fixture()
def data_dirs(tmp_path):
    bench = tmp_path / "bench"
    train = tmp_path / "runs"
    _make_report(bench, "bench-e2e")
    _make_train_run(train, "lora-run-1")
    # 训练目录下的 bench 报告不应混入训练列表
    _make_report(train, "bench-nested")
    return bench, train


@pytest.fixture()
def app(data_dirs):
    bench, train = data_dirs
    cfg = WebConfig(
        serve_name="minillm-test", host="127.0.0.1", port=8080,
        engines=[], bench_dir=str(bench), train_dir=str(train),
    )
    return build_web_app(cfg)


# ---------------------------------------------------------------- API tests

def test_index_page(app):
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "miniLLM Admin Console" in resp.text
    assert "engine-cards" in resp.text


def test_static_assets(app):
    client = TestClient(app)
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_health(app):
    client = TestClient(app)
    data = client.get("/api/health").json()
    assert data["status"] == "ok"


def test_engines_empty_config(app):
    client = TestClient(app)
    assert client.get("/api/engines").json() == []


def test_status_with_no_engines(app):
    client = TestClient(app)
    data = client.get("/api/status").json()
    assert data["engines"] == []
    assert data["serve_name"] == "minillm-test"


def test_bench_list(app, data_dirs):
    client = TestClient(app)
    data = client.get("/api/bench").json()
    assert len(data["runs"]) == 1  # 仅 bench 目录
    run = data["runs"][0]
    assert run["experiment"] == "bench-e2e"
    assert run["n_cases"] == 2
    assert len(run["findings"]) == 1


def test_train_list_excludes_bench(app, data_dirs):
    client = TestClient(app)
    data = client.get("/api/train").json()
    names = [r["experiment"] for r in data["runs"]]
    assert names == ["lora-run-1"]
    assert data["runs"][0]["eval_loss"] == 1.234


def test_missing_dirs_graceful(tmp_path):
    cfg = WebConfig(serve_name="x", engines=[], bench_dir=str(tmp_path / "nope"),
                    train_dir=str(tmp_path / "nope2"))
    client = TestClient(build_web_app(cfg))
    assert client.get("/api/bench").json()["runs"] == []
    assert client.get("/api/train").json()["runs"] == []


# ---------------------------------------------------------------- integration

@pytest.fixture(scope="module")
def serve_url():
    """Real HF OpenAI-compatible server in a thread."""
    port = _free_port()
    cfg = ServeConfig(
        engine="hf", serve_name="minillm-test", host="127.0.0.1", port=port,
        model={"name_or_path": MODEL, "dtype": "fp32", "attn_impl": "eager", "max_model_len": 256},
    )
    server = uvicorn.Server(uvicorn.Config(build_serve_app(cfg), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                # 制造一点流量，让指标非零
                httpx.post(
                    f"{url}/v1/chat/completions",
                    json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 8, "temperature": 0.0},
                    timeout=30,
                )
                yield url
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:  # pragma: no cover
        raise RuntimeError("serve server did not start")
    server.should_exit = True
    thread.join(timeout=10)


def test_status_with_live_engine(serve_url):
    cfg = WebConfig(
        serve_name="minillm-test", engines=[{"name": "hf-live", "type": "hf", "base_url": serve_url}],
    )
    client = TestClient(build_web_app(cfg))
    data = client.get("/api/status").json()
    assert len(data["engines"]) == 1
    engine = data["engines"][0]
    assert engine["up"] is True
    assert engine["model"] == MODEL
    assert engine["metrics"]["minillm_requests_total"] >= 1
    assert engine["metrics"]["minillm_tokens_generated_total"] >= 1


# ---------------------------------------------------------------- /api/chat

def _chat_cfg(serve_url) -> WebConfig:
    return WebConfig(
        serve_name="minillm-test",
        engines=[{"name": "hf-live", "type": "hf", "base_url": serve_url}],
    )


def test_chat_non_stream(serve_url):
    client = TestClient(build_web_app(_chat_cfg(serve_url)))
    resp = client.post("/api/chat", json={
        "engine": "hf-live",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "temperature": 0.0, "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"]
    assert data["usage"]["completion_tokens"] == 8


def test_chat_stream_sse(serve_url):
    client = TestClient(build_web_app(_chat_cfg(serve_url)))
    with client.stream("POST", "/api/chat", json={
        "engine": "hf-live",
        "messages": [{"role": "user", "content": "stream me"}],
        "max_tokens": 8, "temperature": 0.0, "stream": True,
    }) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())
    assert "data:" in body
    assert "[DONE]" in body


def test_chat_unknown_engine():
    client = TestClient(build_web_app(_chat_cfg("http://127.0.0.1:9")))
    resp = client.post("/api/chat", json={
        "engine": "nope", "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 404


def test_chat_engine_unreachable():
    client = TestClient(build_web_app(_chat_cfg("http://127.0.0.1:1")))
    resp = client.post("/api/chat", json={
        "engine": "hf-live", "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 503


def test_admin_config_three_engines():
    from minillm.common.utils import load_model_config

    cfg = load_model_config(WebConfig, "configs/web/admin.yaml")
    names = [e.name for e in cfg.engines]
    assert names == ["vllm-mlx", "hf-local", "sglang-mlx"]
    assert any("8010" in e.base_url for e in cfg.engines)


def test_status_with_down_engine_graceful():
    cfg = WebConfig(
        serve_name="minillm-test",
        engines=[{"name": "down", "type": "vllm", "base_url": "http://127.0.0.1:1"}],
    )
    client = TestClient(build_web_app(cfg))
    data = client.get("/api/status").json()
    engine = data["engines"][0]
    assert engine["up"] is False
    assert engine["error"] is not None
