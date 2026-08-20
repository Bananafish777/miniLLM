#!/usr/bin/env bash
# Fetch a HuggingFace repo into a local directory using curl.
#
# Why curl: in some networks (e.g. behind certain firewalls / sandboxes),
# hf-mirror.com serves Python HTTP clients (httpx/urllib/requests) a 308
# redirect to huggingface.co (blocked), while curl is proxied normally.
# This script lets the hub-path smoke test run with a *real* pretrained
# checkpoint without depending on the Python client's hub download path.
#
# Usage: scripts/fetch_model.sh [REPO] [DEST]
#   REPO  huggingface repo id            (default: hf-internal-testing/tiny-random-LlamaForCausalLM)
#   DEST  local destination directory    (default: data/models/<repo basename>)
set -euo pipefail

REPO="${1:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
DEST="${2:-data/models/$(basename "$REPO")}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "fetching $REPO -> $DEST (endpoint: $ENDPOINT)"
mkdir -p "$DEST"

FILES="$(curl -s --max-time 15 "$ENDPOINT/api/models/$REPO" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(f['rfilename'] for f in d.get('siblings',[]) if not f['rfilename'].startswith(('onnx','README','.gitattributes'))))")"

if [ -z "$FILES" ]; then
  echo "ERROR: no files listed for $REPO (endpoint unreachable?)" >&2
  exit 1
fi

for f in $FILES; do
  mkdir -p "$(dirname "$DEST/$f")"
  curl -sL --retry 2 --max-time 180 -o "$DEST/$f" "$ENDPOINT/$REPO/resolve/main/$f" &
done
wait

echo "done: $(ls "$DEST" | tr '\n' ' ')"
