#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MANIFEST="${MANIFEST:-results/csr_sweep_manifest.csv}"
DELTAS="${DELTAS:-1,2,4,8,16,32,64}"
REPEATS="${REPEATS:-3}"
NET_LIMIT="${NET_LIMIT:-10}"
PAIR_LIMIT="${PAIR_LIMIT:-50}"
OUT="${OUT:-results/net_delta_sweep.csv}"
SUMMARY="${SUMMARY:-results/net_delta_summary.csv}"

if [ ! -x build/net_delta_runner ]; then
  echo "Missing build/net_delta_runner. Build it first; see the command below:" >&2
  echo "  hipcc -std=c++17 -O3 -DROUTING_PATHFINDER_NO_MAIN ..." >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST. Generate CSR first." >&2
  exit 1
fi

command=(
  "build/net_delta_runner"
  "--csr" "{csr}"
  "--metadata" "{metadata}"
  "--delta" "{delta}"
  "--net-limit" "$NET_LIMIT"
  "--pair-limit" "$PAIR_LIMIT"
  "--repeat" "$REPEATS"
)

if [ -n "${MAX_SSSP_ITERS:-}" ]; then
  command+=("--max-sssp-iters" "$MAX_SSSP_ITERS")
fi

python3 tools/delta_sweep.py \
  --manifest "$MANIFEST" \
  --deltas "$DELTAS" \
  --repeats 1 \
  --out "$OUT" \
  --command "${command[*]}"

python3 tools/analyze_delta_results.py \
  --sweep "$OUT" \
  --summary "$SUMMARY"

echo "Done. See $SUMMARY"
