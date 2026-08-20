"""Benchmark case expansion: matrix → flat list of concrete load cases."""

from __future__ import annotations

from dataclasses import dataclass

from minillm.bench.config import BenchRunConfig


@dataclass(frozen=True)
class BenchCase:
    """One load case: a single engine under one concurrency/length combo."""

    engine: str
    model_label: str
    concurrency: int
    input_tokens: int
    output_tokens: int

    def key(self) -> str:
        return (
            f"{self.engine}|{self.model_label}|c{self.concurrency}|"
            f"in{self.input_tokens}|out{self.output_tokens}"
        )

    def label(self) -> str:
        return (
            f"{self.engine}/{self.model_label} c={self.concurrency} "
            f"in={self.input_tokens} out={self.output_tokens}"
        )


def expand_cases(cfg: BenchRunConfig) -> list[BenchCase]:
    """Cartesian product of the configured matrix."""
    cases = [
        BenchCase(
            engine=engine,
            model_label=model,
            concurrency=conc,
            input_tokens=in_len,
            output_tokens=out_len,
        )
        for engine in cfg.matrix.engines
        for model in cfg.matrix.models
        for conc in cfg.matrix.concurrency
        for in_len in cfg.matrix.input_tokens
        for out_len in cfg.matrix.output_tokens
    ]
    return cases
