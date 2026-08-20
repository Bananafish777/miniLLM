SHELL := /bin/bash
PY := .venv/bin/python
UV := uv
# 沙箱/受限环境: 把 uv 的缓存与 Python 安装目录放在工作区内
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python
export UV_STATE_DIR := $(CURDIR)/.uv-state
export HF_HOME := $(CURDIR)/data/cache/huggingface

.PHONY: setup test smoke smoke-hub lint train help

## setup: create venv (Python 3.12) and install package + dev deps
setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install -e ".[dev]"

## test: run fast offline unit tests
test:
	$(PY) -m pytest -m "not hub and not bench" -q

## bench-test: run the real benchmark pipeline on the local HF engine
bench-test:
	$(PY) -m pytest -m bench -q

## smoke: CPU smoke test with scratch tiny GPT-2 (fully offline)
smoke:
	$(PY) -m pytest -m smoke -q

## smoke-hub: hub-path smoke test (fetch real tiny checkpoint via curl mirror, then test)
smoke-hub:
	@test -f data/models/tiny-random-LlamaForCausalLM/model.safetensors || scripts/fetch_model.sh
	$(PY) -m pytest -m hub -q

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
