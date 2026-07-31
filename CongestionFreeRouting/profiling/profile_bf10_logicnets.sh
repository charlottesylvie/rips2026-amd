#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Profile BF10 while routing logicnets_jscl with rocprofv3.

Usage:
  bash CongestionFreeRouting/profiling/profile_bf10_logicnets.sh
  make profile-bf10-logicnets

Environment variables:
  BF10_PROFILE_NET_LIMIT=100       Nets to route; use 0 for the complete design.
  BF10_PROFILE_WORKERS=1           Parallel BF10 worker streams.
  BF10_PROFILE_RUN=<timestamp>     Profile run directory name.
  BF10_PROFILE_ROOT=pathfinder-profiles
  BF10_PROFILE_ARGS="..."          Extra rocprofv3 arguments.
  BF10_PROFILE_PATHFINDER_ARGS="..." Extra pathfinder arguments.
  PATHFINDER_DEVICE_GRAPH=<path>   Resident device graph artifact.

Examples:
  # Fast attribution trace (default: 100 nets, one worker)
  make profile-bf10-logicnets

  # Full LogicNets trace
  BF10_PROFILE_NET_LIMIT=0 make profile-bf10-logicnets

  # Check overlap with four independent BF10 workspaces
  BF10_PROFILE_WORKERS=4 BF10_PROFILE_NET_LIMIT=1000 \
    make profile-bf10-logicnets
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

for command_name in hipcc make python3 rocprofv3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: required command '$command_name' is not available" >&2
    exit 1
  fi
done

net_limit="${BF10_PROFILE_NET_LIMIT:-100}"
workers="${BF10_PROFILE_WORKERS:-1}"
run_name="${BF10_PROFILE_RUN:-bf10-logicnets-$(date +%Y%m%d-%H%M%S)}"
profile_root="${BF10_PROFILE_ROOT:-pathfinder-profiles}"
profiler_args="${BF10_PROFILE_ARGS:-}"
extra_pathfinder_args="${BF10_PROFILE_PATHFINDER_ARGS:-}"
device_graph="${PATHFINDER_DEVICE_GRAPH:-xcvu3p.full-poc-base-wire.devicegraph}"
hip_flags="${PATHFINDER_HIP_FLAGS:--std=c++17 -O3 -x hip}"
hip_libs="${PATHFINDER_HIP_LIBS:-}"

if [[ ! "$net_limit" =~ ^[0-9]+$ ]]; then
  echo "error: BF10_PROFILE_NET_LIMIT must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: BF10_PROFILE_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: BF10_PROFILE_RUN may contain only letters, digits, '.', '_', and '-'" >&2
  exit 2
fi

for required_file in \
  logicnets_jscl_unrouted.phys \
  logicnets_jscl.netlist \
  "$device_graph" \
  ./interchange_to_csr \
  ./routes_to_phys; do
  if [[ ! -e "$required_file" ]]; then
    echo "error: required LogicNets input/tool '$required_file' is missing" >&2
    echo "Run the repository setup and device-graph preprocessing steps first." >&2
    exit 1
  fi
done

mkdir -p "$profile_root"
profile_root="$(cd "$profile_root" && pwd)"
profile_dir="$profile_root/$run_name/logicnets_jscl"
if [[ -e "$profile_dir" ]]; then
  echo "error: profile output already exists: $profile_dir" >&2
  echo "Choose a new BF10_PROFILE_RUN name to avoid mixing traces." >&2
  exit 1
fi
work_dir="$repo_root/bf10-profile-work/$run_name"
mkdir -p "$work_dir"

echo "[bf10-profile] rebuilding pathfinder with ROCTx ranges"
make -B ./pathfinder \
  PATHFINDER_HIP_FLAGS="$hip_flags -DPATHFINDER_ENABLE_ROCTX" \
  PATHFINDER_HIP_LIBS="$hip_libs -lrocprofiler-sdk-roctx"
make ./PathFinderFile

pathfinder_args="--parallel-net-workers $workers --work-dir $work_dir --keep-work-dir"
if [[ "$net_limit" != "0" ]]; then
  pathfinder_args+=" --net-limit $net_limit"
fi
if [[ -n "$extra_pathfinder_args" ]]; then
  pathfinder_args+=" $extra_pathfinder_args"
fi

echo "[bf10-profile] run=$run_name workers=$workers net_limit=$net_limit"
make \
  ROUTER=PathFinderFile \
  BENCHMARKS="logicnets_jscl" \
  VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=bellman-ford \
  PATHFINDER_ARGS="$pathfinder_args" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN="$run_name" \
  PATHFINDER_PROFILE_ROOT="$profile_root" \
  PATHFINDER_PROFILE_ARGS="$profiler_args" \
  PATHFINDER_DEVICE_GRAPH="$device_graph" \
  logicnets_jscl_PathFinderFile.phys

database="$(find "$profile_dir" -type f -name '*_results.db' -print -quit 2>/dev/null || true)"
if [[ -z "$database" ]]; then
  echo "error: rocprofv3 completed but no '*_results.db' was found under $profile_dir" >&2
  exit 1
fi

analysis_dir="$profile_dir/bf10-analysis"
python3 "$script_dir/analyze_bf10_rocpd.py" \
  "$database" \
  --output-dir "$analysis_dir"

echo "[bf10-profile] database: $database"
echo "[bf10-profile] report:   $analysis_dir/summary.md"
echo "[bf10-profile] json:     $analysis_dir/summary.json"
