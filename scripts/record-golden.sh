#!/usr/bin/env bash
# Record golden fixtures from llama.cpp — the oracle this engine must match.
#
#   ./scripts/record-golden.sh tokenizer   # tests/fixtures/tokenizer_golden.json
#   ./scripts/record-golden.sh logits      # tests/fixtures/logits_golden.{json,bin}
#   ./scripts/record-golden.sh greedy      # tests/fixtures/greedy_golden.json
#
# Fixtures are committed; the tests never call llama.cpp themselves.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${LLM_MODEL:-$ROOT/models/MiniCPM5-2B-Q8_0.gguf}"
WHAT="${1:-tokenizer}"

[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

case "$WHAT" in
  tokenizer) python3 "$ROOT/scripts/record_tokenizer.py" "$MODEL" "$ROOT" ;;
  logits)    python3 "$ROOT/scripts/record_logits.py" "$MODEL" "$ROOT" ;;
  greedy)    python3 "$ROOT/scripts/record_greedy.py" "$MODEL" "$ROOT" ;;
  *) echo "unknown fixture: $WHAT" >&2; exit 1 ;;
esac
