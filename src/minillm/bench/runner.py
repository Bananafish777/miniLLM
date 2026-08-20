"""Benchmark case runner: warmup → concurrent load → per-case measurements."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from minillm.bench.cases import BenchCase
from minillm.bench.client import AsyncHFClient, AsyncOpenAIClient, RequestSample, make_prompt
from minillm.bench.config import BenchRunConfig

log = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """All measurements for one load case."""

    case: BenchCase
    samples: list[RequestSample]
    wall_s: float
    total_output_tokens: int
    total_input_tokens: int
    server_tokens_before: int
    server_tokens_after: int
    server_requests_before: int
    server_requests_after: int
    gpu_mem_samples: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        ok = sum(1 for s in self.samples if s.error is None)
        return ok / len(self.samples) if self.samples else 0.0


async def _worker(
    client: AsyncOpenAIClient | AsyncHFClient,
    sem: asyncio.Semaphore,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> RequestSample:
    async with sem:
        return await client.stream_completions(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )


async def _gpu_mem_sampler(client, stop: asyncio.Event, out: list[float]) -> None:
    """Sample GPU memory (when available) every 0.5 s during a case."""
    while not stop.is_set():
        try:
            stats = client.adapter.system_stats() if isinstance(client, AsyncHFClient) else None
            if stats is not None and stats.gpu_mem_used_bytes:
                out.append(float(stats.gpu_mem_used_bytes))
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass


async def run_case(
    cfg: BenchRunConfig, case: BenchCase, client: AsyncOpenAIClient | AsyncHFClient
) -> CaseResult:
    """Run one case: warmup, then `concurrency × requests_per_concurrency` requests."""
    prompt = make_prompt(case.input_tokens)
    sampling = cfg.sampling
    run_cfg = cfg.run

    # warmup (engine caches, CUDA graphs, prefix cache warm)
    for _ in range(run_cfg.warmup_requests):
        await client.stream_completions(
            prompt, max_tokens=case.output_tokens,
            temperature=sampling.temperature, top_p=sampling.top_p,
        )

    server_before = await client.metrics()

    sem = asyncio.Semaphore(case.concurrency)
    total = case.concurrency * run_cfg.requests_per_concurrency
    stop = asyncio.Event()
    gpu_samples: list[float] = []
    sampler = asyncio.create_task(_gpu_mem_sampler(client, stop, gpu_samples))

    t0 = time.perf_counter()
    samples = await asyncio.gather(
        *(
            _worker(client, sem, prompt, case.output_tokens, sampling.temperature, sampling.top_p)
            for _ in range(total)
        )
    )
    wall = time.perf_counter() - t0
    stop.set()
    await sampler

    server_after = await client.metrics()

    total_out = sum(s.output_tokens for s in samples)
    total_in = sum(s.input_tokens for s in samples if s.input_tokens)
    result = CaseResult(
        case=case,
        samples=samples,
        wall_s=wall,
        total_output_tokens=total_out,
        total_input_tokens=total_in,
        server_tokens_before=int(server_before.get("minillm_tokens_generated_total", 0)),
        server_tokens_after=int(server_after.get("minillm_tokens_generated_total", 0)),
        server_requests_before=int(server_before.get("minillm_requests_total", 0)),
        server_requests_after=int(server_after.get("minillm_requests_total", 0)),
        gpu_mem_samples=gpu_samples,
    )
    log.info(
        "case done: %s | wall=%.1fs ok=%d/%d out_tokens=%d (server delta=%d) tps=%.1f",
        case.label(), wall, sum(1 for s in samples if s.error is None), total,
        total_out, result.server_tokens_after - result.server_tokens_before,
        total_out / wall if wall > 0 else 0.0,
    )
    return result
