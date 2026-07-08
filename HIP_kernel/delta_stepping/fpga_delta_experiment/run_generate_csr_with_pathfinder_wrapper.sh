#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <benchmark-name> [more-benchmark-names...]" >&2
  echo "Example: $0 vtr_mcml logicnets_jscl" >&2
  exit 2
fi

cd "$(dirname "$0")"

: "${SETUP_ROOT:?Set SETUP_ROOT to the workspace where setup.sh installed rips2026-amd and fpga24_routing_contest}"
: "${DEVICE:?Set DEVICE to the xcvu3p.device path}"

AMD_REPO="${AMD_REPO:-$SETUP_ROOT/rips2026-amd}"
INTERCHANGE_TO_CSR="${INTERCHANGE_TO_CSR:-$AMD_REPO/interchange_to_csr}"
PATHFINDER_BIN="${PATHFINDER_BIN:-$AMD_REPO/pathfinder}"
SCHEMA_DIR="${SCHEMA_DIR:-$SETUP_ROOT/fpga24_routing_contest/fpga-interchange-schema/interchange}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmarks}"
NET_LIMIT="${NET_LIMIT:-1}"

for required in "$AMD_REPO/Routing/pathfinder_benchmark.py" "$INTERCHANGE_TO_CSR" "$PATHFINDER_BIN" "$SCHEMA_DIR/PhysicalNetlist.capnp" "$DEVICE" "$BENCHMARK_DIR"; do
  if [ ! -e "$required" ]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

mkdir -p results/pathfinder_csr
manifest="results/csr_sweep_manifest.csv"
printf 'benchmark,csr,metadata,netlist,phys\n' > "$manifest"

for name in "$@"; do
  phys="$BENCHMARK_DIR/${name}_unrouted.phys"
  netlist="$BENCHMARK_DIR/${name}.netlist"
  out_dir="results/pathfinder_csr/$name"
  out_phys="$out_dir/${name}_pathfinder.phys"

  if [ ! -f "$phys" ]; then
    echo "Missing physical benchmark: $phys" >&2
    exit 1
  fi
  if [ ! -f "$netlist" ]; then
    echo "Missing logical benchmark: $netlist" >&2
    exit 1
  fi

  mkdir -p "$out_dir"
  echo "Generating CSR for $name"
  python3 "$AMD_REPO/Routing/pathfinder_benchmark.py" \
    "$phys" \
    "$out_phys" \
    --logical-netlist "$netlist" \
    --device "$DEVICE" \
    --schema-dir "$SCHEMA_DIR" \
    --interchange-to-csr "$INTERCHANGE_TO_CSR" \
    --pathfinder "$PATHFINDER_BIN" \
    --work-dir "$out_dir" \
    --keep-work-dir \
    --net-limit "$NET_LIMIT" \
    --allow-unrouted-stubs

  csr="$(find "$out_dir" -maxdepth 1 -name '*.csrbin' -type f | head -n 1)"
  metadata="$(find "$out_dir" -maxdepth 1 -name '*.ifmeta.bin' -type f | head -n 1)"
  if [ -z "$csr" ] || [ -z "$metadata" ]; then
    echo "CSR or metadata output missing for $name" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s\n' "$name" "$csr" "$metadata" "$netlist" "$phys" >> "$manifest"
done

echo "Done. Wrote $manifest"
