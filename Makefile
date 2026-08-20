SHELL := /bin/bash
PY := .venv/bin/python
UV := uv

.PHONY: setup test smoke smoke-hub lint train help

## setup: create venv (Python 3.12) and install package + dev deps
setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install -e ".[dev]"

## test: run fast offline unit tests
test:
	$(PY) -m pytest -m "not hub" -q

## smoke: CPU smoke test with scratch tiny GPT-2 (fully offline)
smoke:
	$(PY) -m pytest -m smoke -q

## smoke-hub: hub-path smoke test via HuggingFace mirror (set HF_ENDPOINT if needed)
smoke-hub:
	HF_ENDPOINT=$${HF_ENDPOINT:-https://hf-mirror.com} $(PY) -m pytest -m hub -q

## lint: ruff check
lint:
	$(UV) run ruff check src tests

## train: run finetuning pipeline with a YAML config
train:
	$(PY) -m minillm.cli train --config $(CONFIG)

## serve: vLLM OpenAI-compatible server (M2)
serve:
	$(PY) -m minillm.cli serve --config $(CONFIG)

## bench: run benchmark matrix (M3)
bench:
	$(PY) -m minillm.cli bench --config $(CONFIG)

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'
