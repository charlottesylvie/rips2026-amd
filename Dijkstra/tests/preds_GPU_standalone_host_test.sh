#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
test_tmp="$(mktemp -d "${TMPDIR:-/tmp}/preds-gpu-host-test.XXXXXX")"
trap 'rm -rf "$test_tmp"' EXIT

cxx="${CXX:-c++}"
common_flags=(
  -std=c++17
  -O0
  -Wall
  -Wextra
  -Wpedantic
  # The host shim intentionally discards launch-only values and turns HIP
  # shared storage into ordinary placeholders.
  -Wno-uninitialized
  -Wno-unused-function
  -Wno-unused-parameter
  -Wno-unused-variable
  -I"$script_dir/fake_hip"
  -I"$repo_root/Dijkstra"
)

"$cxx" "${common_flags[@]}" -DPREDS_GPU_STANDALONE_MAIN \
  "$repo_root/Dijkstra/preds_GPU.cpp" \
  -o "$test_tmp/preds_GPU"

expect_success() {
  local expected="$1"
  shift
  "$@" >"$test_tmp/stdout" 2>"$test_tmp/stderr"
  grep -F -- "$expected" "$test_tmp/stdout" >/dev/null
  test ! -s "$test_tmp/stderr"
}

expect_failure() {
  local expected="$1"
  shift
  if "$@" >"$test_tmp/stdout" 2>"$test_tmp/stderr"; then
    echo "expected command to fail: $*" >&2
    return 1
  fi
  grep -F -- "$expected" "$test_tmp/stderr" >/dev/null
}

expect_success "Usage:" "$test_tmp/preds_GPU" --help
expect_success "Predicate modes:" "$test_tmp/preds_GPU" -h
expect_failure \
  "expected a RIPS CSR file and a source node" \
  "$test_tmp/preds_GPU"
expect_failure \
  "source node must be a nonnegative 32-bit integer" \
  "$test_tmp/preds_GPU" graph.csrbin -1 --predicate IN_SIMPLE
expect_failure \
  "--predicate <MODE> is required" \
  "$test_tmp/preds_GPU" graph.csrbin 0
expect_failure \
  "invalid predicate mode 'NOT_A_MODE'" \
  "$test_tmp/preds_GPU" graph.csrbin 0 --predicate NOT_A_MODE
expect_failure \
  "--predicate may be specified only once" \
  "$test_tmp/preds_GPU" graph.csrbin 0 \
    --predicate IN_SIMPLE --predicate OUT_SIMPLE

"$cxx" "${common_flags[@]}" \
  -DPREDS_GPU_STANDALONE_MAIN -DPREDS_GPU_NO_MAIN \
  -c "$repo_root/Dijkstra/preds_GPU.cpp" \
  -o "$test_tmp/preds_GPU_no_main.o"
"$cxx" "${common_flags[@]}" \
  "$script_dir/preds_GPU_guarded_host_smoke.cpp" \
  "$test_tmp/preds_GPU_no_main.o" \
  -o "$test_tmp/preds_GPU_guarded_host_smoke"
"$test_tmp/preds_GPU_guarded_host_smoke"

echo "preds_GPU standalone host regression passed"
