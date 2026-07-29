#!/usr/bin/env bash
# AMD-server validation harness for the generic Delta-Stepping controllers.

set -Eeuo pipefail

IFS=$'\n\t'
umask 022

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
CALLER_DIR=$(pwd -P)

usage() {
  cat <<'EOF'
Usage:
  CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh [options]

Run this command from the repository root. The harness refuses a different
working directory so relative contest classpath entries and evidence are clear.

Phases:
  --phase 1|2|3|all       Run preflight/build/smoke, full correctness, full
                           timing/profile, or every phase (default: all).
  --resume RUN_DIR         Resume an existing harness run. All immutable
                           configuration is loaded from RUN_DIR/state/config.sh.
  --confirm-gpu-idle       Required before phase 3. This is the operator's
                           assertion that unrelated GPU work has been stopped.

Inputs (fresh runs only):
  --run-root DIR           Parent for unique run directories.
  --benchmark NAME         Benchmark stem (default: logicnets_jscl).
  --input-phys FILE        Unrouted physical netlist.
  --logical-netlist FILE   Logical netlist used by the checker.
  --device-graph FILE      Preprocessed full-device routing graph.
  --interchange-to-csr EXE Existing converter binary; never rebuilt or replaced.
  --routes-to-phys EXE     Existing reconstructor binary; never rebuilt/replaced.
  --java-classpath FILE    Existing java-classpath.txt for CheckPhysNetlist.

Tools and thresholds (fresh runs only):
  --host-cxx COMMAND       Host compiler (default: g++).
  --hipcc COMMAND          HIP compiler (default: hipcc).
  --python COMMAND         Python 3 interpreter (default: python3).
  --java COMMAND           Java launcher (default: java).
  --rocprofv3 COMMAND      ROCm profiler (default: rocprofv3).
  --jvm-heap-mib N         Checker initial/maximum heap in MiB (default: 32736).
  --expected-routes N      Expected full-run route records (default: 27960).
  --expected-depth N       Required full-run maximum route depth (default: 214).
  --stress-runs N          Sequential/multi-queue HIP stress runs (default: 1200).
  --test-timeout-minutes N Low-level test timeout (default: 60).
  --smoke-timeout-minutes N
                           Each 500-route smoke timeout (default: 120).
  --full-timeout-minutes N Full correctness/timing timeout (default: 720).
  --profile-timeout-minutes N
                           Full profile/analysis timeout (default: 1440).
  --timing-repetitions N   Full timing repetitions; must be >=3 (default: 5).
  --max-timing-regression-percent P
                           Maximum reduced b4 median regression versus host
                           (default: 5). Use a nonnegative decimal.
  --min-free-gib N         Free disk required before phase 3 (default: 20).
  -h, --help               Show this help.

The harness never removes a run, reuses a partial attempt directory, overwrites
repository binaries, downloads assets, or invokes Make clean/distclean. A failed
attempt remains in place and resume creates a new attempt beside it.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '[delta-validation] %s\n' "$*"
}

((BASH_VERSINFO[0] >= 4)) || die "Bash 4 or newer is required"
[[ $CALLER_DIR == "$REPO_ROOT" ]] ||
  die "invoke this harness from the repository root: $REPO_ROOT (current: $CALLER_DIR)"

for argument in "$@"; do
  case "$argument" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

PHASE=all
RESUME_DIR=
CONFIRM_GPU_IDLE=0
# Keep default evidence outside the Git worktree so the preflight status is not
# contaminated by the harness's own append-only artifacts.
RUN_ROOT="$REPO_ROOT/../rips2026-amd-validation/runs"
BENCHMARK=logicnets_jscl
INPUT_PHYS="$REPO_ROOT/logicnets_jscl_unrouted.phys"
LOGICAL_NETLIST="$REPO_ROOT/logicnets_jscl.netlist"
DEVICE_GRAPH="$REPO_ROOT/xcvu3p.full-poc-base-wire.devicegraph"
INTERCHANGE_TO_CSR="$REPO_ROOT/interchange_to_csr"
ROUTES_TO_PHYS="$REPO_ROOT/routes_to_phys"
JAVA_CLASSPATH="$REPO_ROOT/java-classpath.txt"
HOST_CXX=g++
HIPCC=hipcc
PYTHON=python3
JAVA=java
ROCPROFV3=rocprofv3
JVM_HEAP_MIB=32736
EXPECTED_ROUTES=27960
EXPECTED_DEPTH=214
STRESS_RUNS=1200
TEST_TIMEOUT_MINUTES=60
SMOKE_TIMEOUT_MINUTES=120
FULL_TIMEOUT_MINUTES=720
PROFILE_TIMEOUT_MINUTES=1440
TIMING_REPETITIONS=5
MAX_TIMING_REGRESSION_PERCENT=5
MIN_FREE_GIB=20
SMOKE_NET_LIMIT=500
CONFIG_VERSION=1
CONFIG_REPO_ROOT=
RESUMING=0

# Find --resume before normal parsing so its immutable configuration can be
# loaded before any phase begins. Only config.sh files created by this harness
# and accompanied by the run sentinel are accepted.
ORIGINAL_ARGS=("$@")
for ((arg_index = 0; arg_index < ${#ORIGINAL_ARGS[@]}; ++arg_index)); do
  if [[ ${ORIGINAL_ARGS[$arg_index]} == --resume ]]; then
    ((arg_index + 1 < ${#ORIGINAL_ARGS[@]})) || die "--resume requires a value"
    [[ -z $RESUME_DIR ]] || die "--resume may be specified only once"
    RESUME_DIR=${ORIGINAL_ARGS[$((arg_index + 1))]}
  fi
done

if [[ -n $RESUME_DIR ]]; then
  [[ -d $RESUME_DIR ]] || die "resume directory does not exist: $RESUME_DIR"
  RUN_DIR=$(cd -- "$RESUME_DIR" && pwd -P)
  [[ -f $RUN_DIR/state/harness-run-v1 ]] ||
    die "not a delta-controller harness run: $RUN_DIR"
  [[ -f $RUN_DIR/state/config.sh ]] || die "resume config is missing: $RUN_DIR/state/config.sh"
  # shellcheck disable=SC1090
  source "$RUN_DIR/state/config.sh"
  [[ ${CONFIG_VERSION:-} == 1 ]] || die "unsupported resume config version"
  [[ ${CONFIG_REPO_ROOT:-} == "$REPO_ROOT" ]] ||
    die "resume run belongs to a different repository: ${CONFIG_REPO_ROOT:-unknown}"
  RESUMING=1
fi

fresh_only() {
  ((RESUMING == 0)) || die "$1 cannot be changed while resuming; start a fresh run"
}

while (($#)); do
  case "$1" in
    --phase)
      (($# >= 2)) || die "--phase requires a value"
      PHASE=$2
      shift 2
      ;;
    --resume)
      (($# >= 2)) || die "--resume requires a value"
      shift 2
      ;;
    --confirm-gpu-idle)
      CONFIRM_GPU_IDLE=1
      shift
      ;;
    --run-root|--benchmark|--input-phys|--logical-netlist|--device-graph|\
    --interchange-to-csr|--routes-to-phys|--java-classpath|--host-cxx|--hipcc|\
    --python|--java|--rocprofv3|--jvm-heap-mib|--expected-routes|\
    --expected-depth|--stress-runs|--timing-repetitions|\
    --test-timeout-minutes|--smoke-timeout-minutes|--full-timeout-minutes|\
    --profile-timeout-minutes|\
    --max-timing-regression-percent|--min-free-gib)
      (($# >= 2)) || die "$1 requires a value"
      fresh_only "$1"
      option=$1
      value=$2
      case "$option" in
        --run-root) RUN_ROOT=$value ;;
        --benchmark) BENCHMARK=$value ;;
        --input-phys) INPUT_PHYS=$value ;;
        --logical-netlist) LOGICAL_NETLIST=$value ;;
        --device-graph) DEVICE_GRAPH=$value ;;
        --interchange-to-csr) INTERCHANGE_TO_CSR=$value ;;
        --routes-to-phys) ROUTES_TO_PHYS=$value ;;
        --java-classpath) JAVA_CLASSPATH=$value ;;
        --host-cxx) HOST_CXX=$value ;;
        --hipcc) HIPCC=$value ;;
        --python) PYTHON=$value ;;
        --java) JAVA=$value ;;
        --rocprofv3) ROCPROFV3=$value ;;
        --jvm-heap-mib) JVM_HEAP_MIB=$value ;;
        --expected-routes) EXPECTED_ROUTES=$value ;;
        --expected-depth) EXPECTED_DEPTH=$value ;;
        --stress-runs) STRESS_RUNS=$value ;;
        --test-timeout-minutes) TEST_TIMEOUT_MINUTES=$value ;;
        --smoke-timeout-minutes) SMOKE_TIMEOUT_MINUTES=$value ;;
        --full-timeout-minutes) FULL_TIMEOUT_MINUTES=$value ;;
        --profile-timeout-minutes) PROFILE_TIMEOUT_MINUTES=$value ;;
        --timing-repetitions) TIMING_REPETITIONS=$value ;;
        --max-timing-regression-percent) MAX_TIMING_REGRESSION_PERCENT=$value ;;
        --min-free-gib) MIN_FREE_GIB=$value ;;
      esac
      shift 2
      ;;
    -h|--help)
      shift
      ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$PHASE" in 1|2|3|all) ;; *) die "--phase must be 1, 2, 3, or all" ;; esac
[[ $BENCHMARK =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe benchmark name: $BENCHMARK"
[[ $JVM_HEAP_MIB =~ ^[1-9][0-9]*$ ]] || die "--jvm-heap-mib must be positive"
[[ $EXPECTED_ROUTES =~ ^[1-9][0-9]*$ ]] || die "--expected-routes must be positive"
[[ $EXPECTED_DEPTH =~ ^[0-9]+$ ]] || die "--expected-depth must be nonnegative"
[[ $STRESS_RUNS =~ ^[1-9][0-9]*$ ]] || die "--stress-runs must be positive"
[[ $TEST_TIMEOUT_MINUTES =~ ^[1-9][0-9]*$ ]] || die "--test-timeout-minutes must be positive"
[[ $SMOKE_TIMEOUT_MINUTES =~ ^[1-9][0-9]*$ ]] || die "--smoke-timeout-minutes must be positive"
[[ $FULL_TIMEOUT_MINUTES =~ ^[1-9][0-9]*$ ]] || die "--full-timeout-minutes must be positive"
[[ $PROFILE_TIMEOUT_MINUTES =~ ^[1-9][0-9]*$ ]] || die "--profile-timeout-minutes must be positive"
[[ $TIMING_REPETITIONS =~ ^[1-9][0-9]*$ ]] || die "--timing-repetitions must be positive"
((TIMING_REPETITIONS >= 3)) || die "--timing-repetitions must be at least 3"
[[ $MAX_TIMING_REGRESSION_PERCENT =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "--max-timing-regression-percent must be a nonnegative decimal"
[[ $MIN_FREE_GIB =~ ^[1-9][0-9]*$ ]] || die "--min-free-gib must be positive"

absolute_from_caller() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$CALLER_DIR" "$1" ;;
  esac
}

if ((RESUMING == 0)); then
  RUN_ROOT=$(absolute_from_caller "$RUN_ROOT")
  INPUT_PHYS=$(absolute_from_caller "$INPUT_PHYS")
  LOGICAL_NETLIST=$(absolute_from_caller "$LOGICAL_NETLIST")
  DEVICE_GRAPH=$(absolute_from_caller "$DEVICE_GRAPH")
  INTERCHANGE_TO_CSR=$(absolute_from_caller "$INTERCHANGE_TO_CSR")
  ROUTES_TO_PHYS=$(absolute_from_caller "$ROUTES_TO_PHYS")
  JAVA_CLASSPATH=$(absolute_from_caller "$JAVA_CLASSPATH")
fi

allocate_run_dir() {
  mkdir -p -- "$RUN_ROOT"
  local stamp base candidate suffix
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  base="$RUN_ROOT/delta-controller-${stamp}-$$"
  for suffix in {0..99}; do
    candidate=$base
    ((suffix == 0)) || candidate="${base}-${suffix}"
    if mkdir -- "$candidate" 2>/dev/null; then
      RUN_DIR=$(cd -- "$candidate" && pwd -P)
      return 0
    fi
  done
  die "could not allocate a unique run directory below $RUN_ROOT"
}

write_assignment() {
  printf '%s=%q\n' "$1" "$2"
}

write_config() {
  local config=$RUN_DIR/state/config.sh
  [[ ! -e $config ]] || die "refusing to replace existing config: $config"
  {
    write_assignment CONFIG_VERSION "$CONFIG_VERSION"
    write_assignment CONFIG_REPO_ROOT "$REPO_ROOT"
    write_assignment RUN_ROOT "$RUN_ROOT"
    write_assignment BENCHMARK "$BENCHMARK"
    write_assignment INPUT_PHYS "$INPUT_PHYS"
    write_assignment LOGICAL_NETLIST "$LOGICAL_NETLIST"
    write_assignment DEVICE_GRAPH "$DEVICE_GRAPH"
    write_assignment INTERCHANGE_TO_CSR "$INTERCHANGE_TO_CSR"
    write_assignment ROUTES_TO_PHYS "$ROUTES_TO_PHYS"
    write_assignment JAVA_CLASSPATH "$JAVA_CLASSPATH"
    write_assignment HOST_CXX "$HOST_CXX"
    write_assignment HIPCC "$HIPCC"
    write_assignment PYTHON "$PYTHON"
    write_assignment JAVA "$JAVA"
    write_assignment ROCPROFV3 "$ROCPROFV3"
    write_assignment JVM_HEAP_MIB "$JVM_HEAP_MIB"
    write_assignment EXPECTED_ROUTES "$EXPECTED_ROUTES"
    write_assignment EXPECTED_DEPTH "$EXPECTED_DEPTH"
    write_assignment STRESS_RUNS "$STRESS_RUNS"
    write_assignment TEST_TIMEOUT_MINUTES "$TEST_TIMEOUT_MINUTES"
    write_assignment SMOKE_TIMEOUT_MINUTES "$SMOKE_TIMEOUT_MINUTES"
    write_assignment FULL_TIMEOUT_MINUTES "$FULL_TIMEOUT_MINUTES"
    write_assignment PROFILE_TIMEOUT_MINUTES "$PROFILE_TIMEOUT_MINUTES"
    write_assignment TIMING_REPETITIONS "$TIMING_REPETITIONS"
    write_assignment MAX_TIMING_REGRESSION_PERCENT "$MAX_TIMING_REGRESSION_PERCENT"
    write_assignment MIN_FREE_GIB "$MIN_FREE_GIB"
    write_assignment SMOKE_NET_LIMIT "$SMOKE_NET_LIMIT"
  } >"$config"
  chmod a-w -- "$config" 2>/dev/null || true
}

if ((RESUMING == 0)); then
  allocate_run_dir
  mkdir -p -- "$RUN_DIR/state/steps" "$RUN_DIR/state/cases" \
    "$RUN_DIR/state/profile-analysis" "$RUN_DIR/attempts" "$RUN_DIR/cases" \
    "$RUN_DIR/profile-analysis" "$RUN_DIR/reports"
  printf 'delta-controller-server-validation-v1\n' >"$RUN_DIR/state/harness-run-v1"
  CONFIG_REPO_ROOT=$REPO_ROOT
  write_config
fi

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

CURRENT_CONTEXT=initialization
on_exit() {
  local status=$?
  set +e
  printf '%s\texit=%s\tcontext=%s\n' "$(timestamp_utc)" "$status" "$CURRENT_CONTEXT" \
    >>"$RUN_DIR/state/events.log"
  if ((status != 0)); then
    printf '[delta-validation] FAILED in %s; all evidence retained in %s\n' \
      "$CURRENT_CONTEXT" "$RUN_DIR" >&2
  fi
}
trap on_exit EXIT

note "run directory: $RUN_DIR"
printf '%s\tstart\tphase=%s\tresume=%s\n' "$(timestamp_utc)" "$PHASE" "$RESUMING" \
  >>"$RUN_DIR/state/events.log"

require_command() {
  command -v -- "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f $1 ]] || die "missing $2: $1"
}

require_executable() {
  [[ -f $1 && -x $1 ]] || die "missing or non-executable $2: $1"
}

new_attempt() {
  local category=$1
  local label=$2
  local parent stamp candidate suffix
  parent="$RUN_DIR/$category/$label"
  mkdir -p -- "$parent"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  for suffix in {0..99}; do
    candidate="$parent/attempt-${stamp}-$$"
    ((suffix == 0)) || candidate="${candidate}-${suffix}"
    if mkdir -- "$candidate" 2>/dev/null; then
      ATTEMPT_DIR=$candidate
      return 0
    fi
  done
  die "could not allocate attempt directory below $parent"
}

verify_success_manifest() {
  local value=$1
  local manifest=$value/.harness-success-v1
  [[ -f $manifest ]] || die "completed result has no success manifest: $value"
  local header
  IFS= read -r header <"$manifest" || true
  [[ $header == delta-controller-success-manifest-v1 ]] ||
    die "completed result has an invalid success manifest: $value"
  local expected_size expected_sha relative observed_size observed_sha
  local listed_count=0
  while IFS=$'\t' read -r expected_size expected_sha relative; do
    [[ -n $expected_size && -n $expected_sha && -n $relative && \
       $expected_size =~ ^[0-9]+$ && $expected_sha =~ ^[0-9a-f]{64}$ ]] ||
      die "malformed success-manifest entry in $manifest"
    [[ $relative != /* && $relative != ../* && $relative != */../* ]] ||
      die "unsafe success-manifest path in $manifest: $relative"
    [[ -f $value/$relative ]] ||
      die "completed result is missing a recorded artifact: $value/$relative"
    observed_size=$(stat -c %s -- "$value/$relative")
    [[ $observed_size == "$expected_size" ]] ||
      die "completed artifact size changed: $value/$relative"
    observed_sha=$(sha256sum -- "$value/$relative")
    observed_sha=${observed_sha%% *}
    [[ $observed_sha == "$expected_sha" ]] ||
      die "completed artifact content changed: $value/$relative"
    ((++listed_count))
  done < <(tail -n +2 -- "$manifest")
  local actual_count
  actual_count=$(find "$value" -type f ! -name .harness-success-v1 -printf . | wc -c)
  [[ $actual_count == "$listed_count" ]] ||
    die "completed result file inventory changed: $value"

  local dependency_manifest=$value/.harness-dependencies-v1
  if [[ -f $dependency_manifest ]]; then
    IFS= read -r header <"$dependency_manifest" || true
    [[ $header == delta-controller-dependencies-v1 ]] ||
      die "completed result has an invalid dependency manifest: $value"
    local expected_marker_sha dependency_marker observed_marker_sha
    local dependency_count=0
    while IFS=$'\t' read -r expected_marker_sha dependency_marker; do
      [[ $expected_marker_sha =~ ^[0-9a-f]{64}$ && \
         -n $dependency_marker && $dependency_marker == "$RUN_DIR"/* ]] ||
        die "unsafe dependency marker in $dependency_manifest"
      [[ -f $dependency_marker ]] ||
        die "completed result dependency is missing: $dependency_marker"
      observed_marker_sha=$(sha256sum -- "$dependency_marker")
      observed_marker_sha=${observed_marker_sha%% *}
      [[ $observed_marker_sha == "$expected_marker_sha" ]] ||
        die "completed result dependency marker changed: $dependency_marker"
      done_value "$dependency_marker" ||
        die "completed result dependency is missing: $dependency_marker"
      ((++dependency_count))
    done < <(tail -n +2 -- "$dependency_manifest")
    ((dependency_count > 0)) ||
      die "completed result has an empty dependency manifest: $value"
  fi
}

declare -A VERIFIED_DONE_MARKERS=()

done_value() {
  local marker=$1
  if [[ -n ${VERIFIED_DONE_MARKERS[$marker]+present} ]]; then
    DONE_VALUE=${VERIFIED_DONE_MARKERS[$marker]}
    return 0
  fi
  [[ -f $marker ]] || return 1
  local target expected_manifest_sha observed_manifest_sha
  {
    IFS= read -r target || true
    IFS= read -r expected_manifest_sha || true
  } <"$marker"
  [[ -n ${target:-} && -d $target && \
     ${expected_manifest_sha:-} =~ ^[0-9a-f]{64}$ ]] ||
    die "done marker is stale or malformed: $marker"
  [[ -f $target/.harness-success-v1 ]] ||
    die "done marker success manifest is missing: $marker"
  observed_manifest_sha=$(sha256sum -- "$target/.harness-success-v1")
  observed_manifest_sha=${observed_manifest_sha%% *}
  [[ $observed_manifest_sha == "$expected_manifest_sha" ]] ||
    die "done marker success-manifest digest changed: $marker"
  verify_success_manifest "$target"
  VERIFIED_DONE_MARKERS["$marker"]=$target
  DONE_VALUE=$target
}

mark_done() {
  local marker=$1
  local value=$2
  [[ -d $value ]] || die "cannot mark a missing result done: $value"
  mkdir -p -- "$(dirname -- "$marker")"
  [[ ! -e $marker ]] || die "refusing to replace done marker: $marker"
  local success_manifest=$value/.harness-success-v1
  [[ ! -e $success_manifest ]] ||
    die "refusing to replace success manifest: $success_manifest"
  {
    printf 'delta-controller-success-manifest-v1\n'
    while IFS= read -r -d '' artifact; do
      local artifact_size artifact_sha relative
      artifact_size=$(stat -c %s -- "$artifact")
      artifact_sha=$(sha256sum -- "$artifact")
      artifact_sha=${artifact_sha%% *}
      relative=${artifact#"$value"/}
      printf '%s\t%s\t%s\n' "$artifact_size" "$artifact_sha" "$relative"
    done < <(find "$value" -type f ! -name .harness-success-v1 -print0 | \
      LC_ALL=C sort -z)
  } >"$success_manifest"
  chmod a-w -- "$success_manifest" 2>/dev/null || true
  local success_manifest_sha
  success_manifest_sha=$(sha256sum -- "$success_manifest")
  success_manifest_sha=${success_manifest_sha%% *}
  printf '%s\n%s\n' "$value" "$success_manifest_sha" >"$marker"
  chmod a-w -- "$marker" 2>/dev/null || true
}

write_dependency_manifest() {
  local output=$1
  shift
  (($# > 0)) || die "dependency manifest requires at least one marker"
  [[ ! -e $output ]] || die "refusing to replace dependency manifest: $output"
  {
    printf 'delta-controller-dependencies-v1\n'
    local marker marker_sha
    for marker in "$@"; do
      [[ -f $marker ]] || die "dependency marker is missing: $marker"
      marker_sha=$(sha256sum -- "$marker")
      marker_sha=${marker_sha%% *}
      printf '%s\t%s\n' "$marker_sha" "$marker"
    done
  } >"$output"
  chmod a-w -- "$output" 2>/dev/null || true
}

format_command() {
  local rendered= arg
  for arg in "$@"; do
    printf -v rendered '%s%q ' "$rendered" "$arg"
  done
  printf '%s\n' "${rendered% }"
}

run_logged_in_dir() {
  local log=$1
  local directory=$2
  shift 2
  {
    printf '\n[%s] cwd=%q command=' "$(timestamp_utc)" "$directory"
    format_command "$@"
  } >>"$log"
  local status
  set +e
  (cd -- "$directory" && "$@") 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  set -e
  printf '[%s] exit=%s\n' "$(timestamp_utc)" "$status" >>"$log"
  return "$status"
}

run_logged() {
  local log=$1
  shift
  run_logged_in_dir "$log" "$REPO_ROOT" "$@"
}

source_fingerprint() {
  "$PYTHON" - "$REPO_ROOT" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
include_roots = (
    root / "CongestionFreeRouting",
    root / "HIP_kernel" / "bellman_ford" / "src",
    root / "Routing" / "tests" / "fake_hip",
)
suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".py", ".sh"}
files = []
for include_root in include_roots:
    if not include_root.exists():
        continue
    for path in include_root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if "__pycache__" in relative.parts or "39155_analysis" in relative.parts:
            continue
        files.append(path)
digest = hashlib.sha256()
for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix().encode()
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    payload = path.read_bytes()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
print(digest.hexdigest())
PY
}

record_or_verify_source_fingerprint() {
  require_command "$PYTHON"
  local fingerprint_file=$RUN_DIR/state/source-fingerprint.sha256
  local observed expected
  observed=$(source_fingerprint)
  if [[ -f $fingerprint_file ]]; then
    IFS= read -r expected <"$fingerprint_file"
    [[ $observed == "$expected" ]] ||
      die "validation sources changed since this run began; start a fresh run"
  else
    printf '%s\n' "$observed" >"$fingerprint_file"
    chmod a-w -- "$fingerprint_file" 2>/dev/null || true
  fi
}

FATAL_PATTERN='(^[[:space:]]*error:)|(^|[^[:alnum:]_])(hipError[A-Z]|HIP error at|HSA_STATUS_ERROR|GPU (page )?fault|Segmentation fault|core dumped|terminate called|uncaught exception|Delta-Stepping cooperative controller detected|stale controller publication|HIP (launch|asynchronous|runtime) error)'
PROFILE_LOSS_PATTERN='(dropped[^0-9]*[1-9][0-9]*|[1-9][0-9]*[[:space:]_-]*(records?|events?|samples?)[[:space:]_-]*dropped|[1-9][0-9]*[[:space:]_-]*dropped[[:space:]_-]*(records?|events?|samples?)|buffer[[:space:]_-]*(overflow|overrun)|trace[[:space:]_-]*(data[[:space:]_-]*)?(loss|incomplete)|data[[:space:]_-]*loss)'

scan_fatal_log() {
  local log=$1
  local matches=$2
  local status
  set +e
  LC_ALL=C grep -Ein -- "$FATAL_PATTERN" "$log" >"$matches"
  status=$?
  set -e
  if ((status == 0)); then
    printf 'Fatal diagnostics found in %s; see %s\n' "$log" "$matches" >&2
    return 1
  fi
  ((status == 1)) || die "fatal-log scan failed for $log"
}

scan_profile_loss_log() {
  local log=$1
  local matches=$2
  local status
  set +e
  LC_ALL=C grep -Ein -- "$PROFILE_LOSS_PATTERN" "$log" >"$matches"
  status=$?
  set -e
  if ((status == 0)); then
    printf 'Profiler loss/overflow diagnostics found in %s; see %s\n' \
      "$log" "$matches" >&2
    return 1
  fi
  ((status == 1)) || die "profile-loss scan failed for $log"
}

verify_profile_loss_pattern() {
  local sample
  for sample in \
    'dropped records: 5' \
    '5 records dropped' \
    '5 dropped records' \
    'trace data loss' \
    'buffer overflow'; do
    printf '%s\n' "$sample" | LC_ALL=C grep -Eiq -- "$PROFILE_LOSS_PATTERN" ||
      die "profile-loss pattern missed its self-test sample: $sample"
  done
  for sample in \
    'dropped records: 0' \
    '0 records dropped' \
    '0 dropped records'; do
    if printf '%s\n' "$sample" | LC_ALL=C grep -Eiq -- "$PROFILE_LOSS_PATTERN"; then
      die "profile-loss pattern rejected its zero-count self-test sample: $sample"
    fi
  done
}

write_input_manifest() {
  local output=$1
  "$PYTHON" - "$output" "$REPO_ROOT" \
    input_phys "$INPUT_PHYS" \
    logical_netlist "$LOGICAL_NETLIST" \
    device_graph "$DEVICE_GRAPH" \
    interchange_to_csr "$INTERCHANGE_TO_CSR" \
    routes_to_phys "$ROUTES_TO_PHYS" \
    java_classpath "$JAVA_CLASSPATH" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
repository = pathlib.Path(sys.argv[2]).resolve()
arguments = sys.argv[3:]
if len(arguments) % 2:
    raise RuntimeError("manifest arguments must be label/path pairs")
manifest = {"schema_version": 1, "files": {}}

def hash_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}

def hash_directory(path):
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        total_bytes += size
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }

def resolve_entry(raw):
    if not raw:
        raise RuntimeError("empty Java classpath components are not supported")
    path = pathlib.Path(raw)
    resolved = (path if path.is_absolute() else repository / path).resolve()
    if resolved == repository:
        raise RuntimeError("the repository root cannot be a Java classpath entry")
    return resolved

def classpath_snapshot(classpath_file):
    raw = classpath_file.read_text(encoding="utf-8").strip()
    if not raw or "\n" in raw or "\r" in raw:
        raise RuntimeError("java-classpath.txt must contain one nonempty classpath line")
    entries = []
    for index, token in enumerate(raw.split(os.pathsep)):
        if "*" in token:
            if token.count("*") != 1 or not token.endswith("*"):
                raise RuntimeError(f"unsupported Java classpath wildcard: {token!r}")
            directory = resolve_entry(token[:-1])
            if not directory.is_dir():
                raise RuntimeError(f"Java classpath wildcard directory is missing: {directory}")
            jars = sorted(
                item.resolve()
                for item in directory.iterdir()
                if item.is_file() and item.suffix.lower() == ".jar"
            )
            if not jars:
                raise RuntimeError(f"Java classpath wildcard has no JARs: {directory}")
            entries.append({
                "index": index,
                "raw": token,
                "kind": "jar_wildcard",
                "directory": str(directory),
                "artifacts": [hash_file(item) for item in jars],
            })
            continue
        path = resolve_entry(token)
        if path.is_file():
            snapshot = hash_file(path)
            snapshot.update({"index": index, "raw": token, "kind": "file"})
        elif path.is_dir():
            snapshot = hash_directory(path)
            snapshot.update({"index": index, "raw": token, "kind": "directory"})
        else:
            raise RuntimeError(f"Java classpath entry is missing: {path}")
        entries.append(snapshot)
    return {"raw": raw, "entries": entries}

for label, raw_path in zip(arguments[0::2], arguments[1::2]):
    path = pathlib.Path(raw_path).resolve()
    manifest["files"][label] = hash_file(path)
manifest["checker_classpath"] = classpath_snapshot(
    pathlib.Path(manifest["files"]["java_classpath"]["path"])
)
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

verify_input_manifest() {
  local manifest=$1
  "$PYTHON" - "$manifest" "$REPO_ROOT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
repository = pathlib.Path(sys.argv[2]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
failures = []

def hash_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}

def hash_directory(path):
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        total_bytes += size
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }

def resolve_entry(raw):
    if not raw:
        raise RuntimeError("empty Java classpath components are not supported")
    path = pathlib.Path(raw)
    resolved = (path if path.is_absolute() else repository / path).resolve()
    if resolved == repository:
        raise RuntimeError("the repository root cannot be a Java classpath entry")
    return resolved

def classpath_snapshot(classpath_file):
    raw = classpath_file.read_text(encoding="utf-8").strip()
    if not raw or "\n" in raw or "\r" in raw:
        raise RuntimeError("java-classpath.txt must contain one nonempty classpath line")
    entries = []
    for index, token in enumerate(raw.split(os.pathsep)):
        if "*" in token:
            if token.count("*") != 1 or not token.endswith("*"):
                raise RuntimeError(f"unsupported Java classpath wildcard: {token!r}")
            directory = resolve_entry(token[:-1])
            if not directory.is_dir():
                raise RuntimeError(f"Java classpath wildcard directory is missing: {directory}")
            jars = sorted(
                item.resolve()
                for item in directory.iterdir()
                if item.is_file() and item.suffix.lower() == ".jar"
            )
            if not jars:
                raise RuntimeError(f"Java classpath wildcard has no JARs: {directory}")
            entries.append({
                "index": index,
                "raw": token,
                "kind": "jar_wildcard",
                "directory": str(directory),
                "artifacts": [hash_file(item) for item in jars],
            })
            continue
        path = resolve_entry(token)
        if path.is_file():
            snapshot = hash_file(path)
            snapshot.update({"index": index, "raw": token, "kind": "file"})
        elif path.is_dir():
            snapshot = hash_directory(path)
            snapshot.update({"index": index, "raw": token, "kind": "directory"})
        else:
            raise RuntimeError(f"Java classpath entry is missing: {path}")
        entries.append(snapshot)
    return {"raw": raw, "entries": entries}

for label, expected in manifest.get("files", {}).items():
    path = pathlib.Path(expected["path"])
    if not path.is_file():
        failures.append(f"{label}: missing {path}")
        continue
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        failures.append(f"{label}: content changed: {path}")
try:
    classpath_file = pathlib.Path(manifest["files"]["java_classpath"]["path"])
    observed_classpath = classpath_snapshot(classpath_file)
    if observed_classpath != manifest.get("checker_classpath"):
        failures.append("checker_classpath: resolved JAR/directory contents changed")
except Exception as error:
    failures.append(f"checker_classpath: {error}")
if failures:
    raise RuntimeError("configured inputs changed since phase 1:\n- " + "\n- ".join(failures))
print(
    f"verified {len(manifest.get('files', {}))} immutable inputs "
    "plus resolved checker classpath artifacts"
)
PY
}

preflight() {
  local marker=$RUN_DIR/state/steps/phase1-preflight.done
  if done_value "$marker"; then
    note "skip completed preflight: $DONE_VALUE"
    verify_input_manifest "$DONE_VALUE/input-manifest.json"
    return 0
  fi

  CURRENT_CONTEXT='phase 1 preflight'
  new_attempt attempts phase1/preflight
  local attempt=$ATTEMPT_DIR
  local log=$attempt/preflight.log
  note "preflight attempt: $attempt"

  local command_name
  for command_name in bash date hostname git "$HOST_CXX" "$HIPCC" "$PYTHON" "$JAVA" \
    tee grep sed awk sort find sha256sum rocminfo cp stat wc tail timeout du; do
    require_command "$command_name"
  done
  verify_profile_loss_pattern
  require_file "$INPUT_PHYS" "unrouted physical netlist"
  require_file "$LOGICAL_NETLIST" "logical netlist"
  require_file "$DEVICE_GRAPH" "preprocessed device graph"
  require_executable "$INTERCHANGE_TO_CSR" "interchange_to_csr helper"
  require_executable "$ROUTES_TO_PHYS" "routes_to_phys helper"
  require_file "$JAVA_CLASSPATH" "Java classpath file"
  require_file "$REPO_ROOT/Makefile" "repository Makefile"
  require_file "$REPO_ROOT/CongestionFreeRouting/pathfinder.cpp" "PathFinder source"
  require_file "$REPO_ROOT/CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp" \
    "Delta-Stepping source"
  local validation_file
  for validation_file in \
    "$REPO_ROOT/CongestionFreeRouting/profiling/summarize_routes_jsonl.py" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/validate_delta_telemetry.py" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/validate_route_depth.py" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/splice_routes_jsonl.py" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/analyze_rocpd.py" \
    "$REPO_ROOT/CongestionFreeRouting/tests/profiling_validation_test.py"; do
    require_file "$validation_file" "validation support file"
  done
  [[ -s $JAVA_CLASSPATH ]] || die "Java classpath file is empty: $JAVA_CLASSPATH"
  [[ -e /dev/kfd ]] || die "/dev/kfd is unavailable; run on a ROCm-enabled AMD server"

  local git_top
  git_top=$(git -C "$REPO_ROOT" rev-parse --show-toplevel)
  git_top=$(cd -- "$git_top" && pwd -P)
  [[ $git_top == "$REPO_ROOT" ]] ||
    die "resolved Git top-level '$git_top' does not match repository '$REPO_ROOT'"

  {
    printf 'run_dir=%s\nrepo_root=%s\nbenchmark=%s\n' "$RUN_DIR" "$REPO_ROOT" "$BENCHMARK"
    printf 'expected_routes=%s\nexpected_depth=%s\nsmoke_net_limit=%s\n' \
      "$EXPECTED_ROUTES" "$EXPECTED_DEPTH" "$SMOKE_NET_LIMIT"
    printf 'stress_runs=%s\ntiming_repetitions=%s\n' "$STRESS_RUNS" "$TIMING_REPETITIONS"
    printf 'test_timeout_minutes=%s\nsmoke_timeout_minutes=%s\n' \
      "$TEST_TIMEOUT_MINUTES" "$SMOKE_TIMEOUT_MINUTES"
    printf 'full_timeout_minutes=%s\nprofile_timeout_minutes=%s\n' \
      "$FULL_TIMEOUT_MINUTES" "$PROFILE_TIMEOUT_MINUTES"
  } >"$attempt/configuration.txt"

  git -C "$REPO_ROOT" rev-parse HEAD >"$attempt/git-head.txt"
  git -C "$REPO_ROOT" status --short >"$attempt/git-status-short.txt"
  git -C "$REPO_ROOT" diff --binary >"$attempt/git-diff.patch"
  git -C "$REPO_ROOT" diff --cached --binary >"$attempt/git-diff-cached.patch"
  git -C "$REPO_ROOT" ls-files --others --exclude-standard >"$attempt/git-untracked.txt"

  {
    printf 'timestamp_utc=%s\n' "$(timestamp_utc)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'git_head=%s\n' "$(<"$attempt/git-head.txt")"
    printf 'git_status_short_lines=%s\n' "$(wc -l <"$attempt/git-status-short.txt")"
  } >"$attempt/preflight-summary.txt"

  run_logged "$log" hostname
  run_logged "$log" uname -a
  if command -v lscpu >/dev/null 2>&1; then run_logged "$log" lscpu; fi
  run_logged "$log" "$HOST_CXX" --version
  run_logged "$log" "$HIPCC" --version
  run_logged "$log" "$PYTHON" --version
  run_logged "$log" "$JAVA" -version
  run_logged "$log" timeout --version
  if command -v -- "$ROCPROFV3" >/dev/null 2>&1; then
    if ! run_logged "$attempt/rocprofv3-probe.log" "$ROCPROFV3" --version; then
      printf 'rocprofv3 is present but its version probe failed; phase 3 will require a working profiler\n' \
        >>"$log"
    fi
  else
    printf 'rocprofv3: NOT AVAILABLE (non-blocking until phase 3)\n' >>"$log"
  fi
  run_logged "$log" rocminfo
  if command -v rocm-smi >/dev/null 2>&1; then
    run_logged "$log" rocm-smi --showproductname --showdriverversion --showuse --showmemuse
  else
    printf 'rocm-smi not installed; rocminfo completed successfully\n' >>"$log"
  fi
  run_logged "$log" df -h "$RUN_DIR"
  scan_fatal_log "$log" "$attempt/fatal-matches.txt"

  write_input_manifest "$attempt/input-manifest.json"
  record_or_verify_source_fingerprint
  mark_done "$marker" "$attempt"
}

preserve_preexisting_binaries() {
  local marker=$RUN_DIR/state/steps/phase1-preexisting-binaries.done
  if done_value "$marker"; then
    note "skip completed pre-existing binary snapshot: $DONE_VALUE"
    return 0
  fi
  CURRENT_CONTEXT='phase 1 pre-existing binary snapshot'
  new_attempt attempts phase1/preexisting-binaries
  local attempt=$ATTEMPT_DIR
  local manifest=$attempt/manifest.txt
  local name source destination
  for name in PathFinderFile pathfinder; do
    source=$REPO_ROOT/$name
    destination=$attempt/${name}.preexisting
    if [[ -f $source ]]; then
      cp -p -- "$source" "$destination"
      {
        printf 'name=%s\nsource=%s\npreserved_copy=%s\n' "$name" "$source" "$destination"
        stat -- "$source"
        sha256sum -- "$source" "$destination"
      } >>"$manifest"
    else
      printf 'name=%s\nsource=%s\nstatus=NOT_PRESENT\n' "$name" "$source" >>"$manifest"
    fi
    printf '\n' >>"$manifest"
  done
  mark_done "$marker" "$attempt"
}

verify_phase1_inputs() {
  local marker=$RUN_DIR/state/steps/phase1-preflight.done
  done_value "$marker" || die "phase 1 preflight has not completed"
  verify_input_manifest "$DONE_VALUE/input-manifest.json"
  record_or_verify_source_fingerprint
}

build_candidates() {
  local marker=$RUN_DIR/state/steps/phase1-build.done
  if done_value "$marker"; then
    NORMAL_BUILD_DIR=$DONE_VALUE
    note "skip completed candidate build: $NORMAL_BUILD_DIR"
    return 0
  fi

  CURRENT_CONTEXT='phase 1 candidate build'
  new_attempt attempts phase1/build
  local attempt=$ATTEMPT_DIR
  local bin=$attempt/bin
  local log=$attempt/build.log
  mkdir -p -- "$bin"
  note "candidate build attempt: $attempt"

  run_logged "$log" "$HOST_CXX" -std=c++17 -O2 \
    "$REPO_ROOT/CongestionFreeRouting/pathfinder_router.cpp" \
    -o "$bin/PathFinderFile-candidate"

  run_logged "$log" "$HIPCC" -std=c++17 -O3 -x hip -DBF10_NO_MAIN \
    -I "$REPO_ROOT/HIP_kernel/bellman_ford/src" \
    -I "$REPO_ROOT/CongestionFreeRouting/bellman_ford" \
    -I "$REPO_ROOT/CongestionFreeRouting/delta_stepping" \
    -I "$REPO_ROOT/CongestionFreeRouting/unit_bfs" \
    "$REPO_ROOT/CongestionFreeRouting/pathfinder.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/bellman_ford/bf10.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp" \
    -pthread -o "$bin/pathfinder-candidate"

  run_logged "$log" "$HOST_CXX" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
    "$REPO_ROOT/CongestionFreeRouting/tests/delta_stepping_policy_test.cpp" \
    -o "$bin/delta_stepping_policy_test"

  run_logged "$log" "$HOST_CXX" -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    "$REPO_ROOT/CongestionFreeRouting/tests/pathfinder_router_args_test.cpp" \
    -o "$bin/pathfinder_router_args_test"

  run_logged "$log" "$HOST_CXX" -std=c++17 -O2 -pthread \
    -I "$REPO_ROOT/Routing/tests/fake_hip" \
    -I "$REPO_ROOT/HIP_kernel/bellman_ford/src" \
    -I "$REPO_ROOT/CongestionFreeRouting/bellman_ford" \
    -I "$REPO_ROOT/CongestionFreeRouting/delta_stepping" \
    -I "$REPO_ROOT/CongestionFreeRouting/unit_bfs" \
    "$REPO_ROOT/CongestionFreeRouting/tests/pathfinder_bf10_cpu_stub_test.cpp" \
    -o "$bin/pathfinder_bf10_cpu_stub_test"

  run_logged "$log" "$HIPCC" -std=c++17 -O2 -pthread -x hip \
    -I "$REPO_ROOT/HIP_kernel/bellman_ford/src" \
    -I "$REPO_ROOT/CongestionFreeRouting/unit_bfs" \
    "$REPO_ROOT/CongestionFreeRouting/tests/unit_bfs_hip_test.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp" \
    -o "$bin/unit_bfs_hip_test"

  run_logged "$log" "$HIPCC" -std=c++17 -O2 -pthread -x hip \
    -I "$REPO_ROOT/HIP_kernel/bellman_ford/src" \
    -I "$REPO_ROOT/CongestionFreeRouting/delta_stepping" \
    "$REPO_ROOT/CongestionFreeRouting/tests/delta_stepping_hip_test.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp" \
    -o "$bin/delta_stepping_hip_test"

  sha256sum -- "$bin"/* >"$attempt/binary-sha256.txt"
  scan_fatal_log "$log" "$attempt/fatal-matches.txt"
  mark_done "$marker" "$attempt"
  NORMAL_BUILD_DIR=$attempt
}

load_normal_build() {
  local marker=$RUN_DIR/state/steps/phase1-build.done
  done_value "$marker" || die "normal candidate build has not completed"
  NORMAL_BUILD_DIR=$DONE_VALUE
  CANDIDATE_WRAPPER=$NORMAL_BUILD_DIR/bin/PathFinderFile-candidate
  CANDIDATE_PATHFINDER=$NORMAL_BUILD_DIR/bin/pathfinder-candidate
  require_executable "$CANDIDATE_WRAPPER" "candidate PathFinderFile"
  require_executable "$CANDIDATE_PATHFINDER" "candidate pathfinder"
}

run_resumable_test() {
  local test_id=$1
  shift
  local marker=$RUN_DIR/state/steps/$test_id.done
  if done_value "$marker"; then
    note "skip completed test $test_id: $DONE_VALUE"
    return 0
  fi
  CURRENT_CONTEXT="test $test_id"
  new_attempt attempts "$test_id"
  local attempt=$ATTEMPT_DIR
  local log=$attempt/test.log
  note "run $test_id: $attempt"
  run_logged "$log" timeout --kill-after=2m \
    "${TEST_TIMEOUT_MINUTES}m" "$@"
  scan_fatal_log "$log" "$attempt/fatal-matches.txt"
  mark_done "$marker" "$attempt"
}

run_low_level_tests() {
  load_normal_build
  local bin=$NORMAL_BUILD_DIR/bin
  run_resumable_test phase1/tests/python-argument-forwarding \
    env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/tests/pathfinder_benchmark_args_test.py"
  run_resumable_test phase1/tests/profiling-validation \
    env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/tests/profiling_validation_test.py"
  run_resumable_test phase1/tests/delta-policy "$bin/delta_stepping_policy_test"
  run_resumable_test phase1/tests/router-args "$bin/pathfinder_router_args_test"
  run_resumable_test phase1/tests/pathfinder-cpu-stub "$bin/pathfinder_bf10_cpu_stub_test"
  run_resumable_test phase1/tests/unit-bfs-hip "$bin/unit_bfs_hip_test"
  run_resumable_test phase1/tests/unit-bfs-reuse-stress \
    env UNIT_BFS_REUSE_STRESS_RUNS="$STRESS_RUNS" "$bin/unit_bfs_hip_test"
  run_resumable_test phase1/tests/delta-reduced-hip \
    env DELTA_REQUIRE_REDUCED_CONTROLLER=1 "$bin/delta_stepping_hip_test"
  run_resumable_test phase1/tests/delta-reduced-multi-queue-stress \
    env DELTA_REQUIRE_REDUCED_CONTROLLER=1 DELTA_MULTI_QUEUE_STRESS_RUNS="$STRESS_RUNS" \
    "$bin/delta_stepping_hip_test"
}

assert_route_summary() {
  local summary=$1
  local expected=$2
  "$PYTHON" - "$summary" "$expected" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
failures = []
for key in ("route_requests", "routed"):
    if summary.get(key) != expected:
        failures.append(f"{key}: expected {expected}, observed {summary.get(key)!r}")
if summary.get("unrouted") != 0:
    failures.append(f"unrouted: expected 0, observed {summary.get('unrouted')!r}")
if summary.get("reached_sinks") != summary.get("sinks"):
    failures.append(
        f"reached_sinks: expected {summary.get('sinks')!r}, "
        f"observed {summary.get('reached_sinks')!r}"
    )
if failures:
    raise RuntimeError("route summary validation failed:\n- " + "\n- ".join(failures))
print(f"route summary gate passed: {expected} routed requests")
PY
}

assert_telemetry_completed() {
  local summary=$1
  "$PYTHON" - "$summary" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
queries = summary.get("queries")
completed = summary.get("completed_queries")
if type(queries) is not int or queries <= 0 or completed != queries:
    raise RuntimeError(
        "telemetry completion mismatch: "
        f"queries={queries!r}, completed={completed!r}"
    )
print(f"telemetry completion gate passed: {queries} queries")
PY
}

compare_route_endpoints() {
  local reference=$1
  local candidate=$2
  local output=$3
  "$PYTHON" - "$reference" "$candidate" "$output" <<'PY'
import hashlib
import json
import pathlib
import sys

reference_path = pathlib.Path(sys.argv[1])
candidate_path = pathlib.Path(sys.argv[2])
output_path = pathlib.Path(sys.argv[3])

def endpoint(value, context):
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}: expected object, observed {value!r}")
    node, site, pin = value.get("node"), value.get("site"), value.get("pin")
    if type(node) is not int or node < 0 or not isinstance(site, str) or not isinstance(pin, str):
        raise RuntimeError(f"{context}: invalid node/site/pin identity")
    return (node, site, pin)

def load(path):
    records = {}
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number}: expected object")
            net = record.get("net")
            if not isinstance(net, str) or not net:
                raise RuntimeError(f"{path}:{line_number}: invalid net name")
            if net in records:
                raise RuntimeError(f"{path}:{line_number}: duplicate net {net!r}")
            routed = record.get("routed")
            if type(routed) is not bool:
                raise RuntimeError(f"{path}:{line_number}: routed must be boolean")
            sources_value = record.get("sources")
            sinks_value = record.get("sinks")
            if not isinstance(sources_value, list) or not isinstance(sinks_value, list):
                raise RuntimeError(f"{path}:{line_number}: sources/sinks must be arrays")
            sources = sorted(
                endpoint(value, f"{path}:{line_number}:sources[{index}]")
                for index, value in enumerate(sources_value)
            )
            sinks = []
            for index, value in enumerate(sinks_value):
                identity = endpoint(value, f"{path}:{line_number}:sinks[{index}]")
                reached = value.get("reached")
                if type(reached) is not bool:
                    raise RuntimeError(
                        f"{path}:{line_number}:sinks[{index}].reached must be boolean"
                    )
                sinks.append((*identity, reached))
            records[net] = {
                "routed": routed,
                "sources": sources,
                "sinks": sorted(sinks),
            }
    if not records:
        raise RuntimeError(f"{path}: no route records")
    return records

reference = load(reference_path)
candidate = load(candidate_path)
failures = []
missing = sorted(set(reference) - set(candidate))
extra = sorted(set(candidate) - set(reference))
if missing:
    failures.append(f"missing nets ({len(missing)}): {missing[:10]!r}")
if extra:
    failures.append(f"extra nets ({len(extra)}): {extra[:10]!r}")
matched_sinks = 0
for net in sorted(set(reference).intersection(candidate)):
    if reference[net] != candidate[net]:
        failures.append(f"endpoint mismatch for net {net!r}")
        if len(failures) >= 21:
            break
    matched_sinks += len(reference[net]["sinks"])

reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
result = {
    "schema_version": 1,
    "reference": str(reference_path.resolve()),
    "candidate": str(candidate_path.resolve()),
    "reference_sha256": reference_sha,
    "candidate_sha256": candidate_sha,
    "byte_identical": reference_sha == candidate_sha,
    "matched_route_requests": len(reference) if not failures else None,
    "matched_sinks": matched_sinks if not failures else None,
    "endpoint_fields_compared": ["net", "routed", "sources.node/site/pin", "sinks.node/site/pin/reached"],
    "intentionally_ignored": ["edges", "sink.source", "artifact_pair_id"],
    "failures": failures,
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if failures:
    raise RuntimeError("route endpoint comparison failed:\n- " + "\n- ".join(failures))
print(
    f"endpoint gate passed for {len(reference)} routes; "
    f"raw SHA identical={result['byte_identical']}"
)
PY
}

run_checker() {
  local case_dir=$1
  local output_phys=$2
  local classpath
  classpath=$(<"$JAVA_CLASSPATH")
  local marker=$case_dir/checker.marker
  local log=$case_dir/checker.log
  if run_logged "$log" timeout --kill-after=2m \
    "${FULL_TIMEOUT_MINUTES}m" "$JAVA" -cp "$classpath" \
    "-Xms${JVM_HEAP_MIB}m" "-Xmx${JVM_HEAP_MIB}m" \
    com.xilinx.fpga24_routing_contest.CheckPhysNetlist \
    "$LOGICAL_NETLIST" "$output_phys" "$INPUT_PHYS"; then
    printf 'PASS\n' >"$marker"
  else
    printf 'FAIL\n' >"$marker"
    return 1
  fi
  [[ $(<"$marker") == PASS ]] || die "physical netlist checker did not record PASS"
}

run_router_case() {
  local case_id=$1
  local controller=$2
  local batch=$3
  local workers=$4
  local net_limit=$5
  local telemetry=$6
  local require_depth=$7
  local expected_count=$8
  local pathfinder_bin=$9
  local execution_mode=${10}
  local reference_routes=${11}
  local marker=$RUN_DIR/state/cases/$case_id.done

  if done_value "$marker"; then
    CASE_RESULT=$DONE_VALUE
    note "skip completed route case $case_id: $CASE_RESULT"
    return 0
  fi

  CURRENT_CONTEXT="route case $case_id"
  new_attempt cases "$case_id"
  local case_dir=$ATTEMPT_DIR
  local output_phys=$case_dir/routed.phys
  local work_dir=$case_dir/work
  local routes=$work_dir/routed.routes.jsonl
  local router_log=$case_dir/router.log
  local profile_raw=$case_dir/profile-raw
  local profile_prefix=
  mkdir -p -- "$work_dir"
  note "run route case $case_id: $case_dir"

  local telemetry_mode
  case "$controller" in
    host-checked) telemetry_mode=host_checked ;;
    reduced-round-trip) telemetry_mode=reduced_round_trip ;;
    *) die "unsupported controller in case $case_id: $controller" ;;
  esac

  local -a router_args=(
    "$INPUT_PHYS" "$output_phys"
    --logical-netlist "$LOGICAL_NETLIST"
    --device-graph "$DEVICE_GRAPH"
    --work-dir "$work_dir"
    --keep-work-dir
    --interchange-to-csr "$INTERCHANGE_TO_CSR"
    --pathfinder "$pathfinder_bin"
    --routes-to-phys "$ROUTES_TO_PHYS"
    --sssp-engine delta-step
    --delta 1
    --delta-force-generic
    --delta-controller "$controller"
    --parallel-net-workers "$workers"
    --strict-routing
  )
  if [[ $controller == reduced-round-trip ]]; then
    router_args+=(--delta-controller-batch-size "$batch")
  fi
  if ((net_limit > 0)); then router_args+=(--net-limit "$net_limit"); fi
  if ((telemetry == 1)); then router_args+=(--delta-telemetry); fi

  {
    printf 'case_id=%s\ncontroller=%s\nbatch=%s\nworkers=%s\n' \
      "$case_id" "$controller" "$batch" "$workers"
    printf 'net_limit=%s\ntelemetry=%s\nrequire_depth=%s\nexpected_count=%s\n' \
      "$net_limit" "$telemetry" "$require_depth" "$expected_count"
    printf 'pathfinder=%s\nexecution_mode=%s\n' "$pathfinder_bin" "$execution_mode"
  } >"$case_dir/case-config.txt"

  local router_status=0
  case "$execution_mode" in
    normal)
      run_logged "$router_log" timeout --kill-after=2m \
        "${FULL_TIMEOUT_MINUTES}m" env PATHFINDER_PROFILE_COMMAND= \
        "$CANDIDATE_WRAPPER" "${router_args[@]}" || router_status=$?
      ;;
    timed)
      require_executable /usr/bin/time "GNU time"
      run_logged "$router_log" timeout --kill-after=2m \
        "${FULL_TIMEOUT_MINUTES}m" env PATHFINDER_PROFILE_COMMAND= \
        /usr/bin/time -f $'wall_seconds=%e\npeak_rss_kib=%M' \
        -o "$case_dir/time.txt" "$CANDIDATE_WRAPPER" "${router_args[@]}" || router_status=$?
      ;;
    profile)
      mkdir -p -- "$profile_raw"
      printf -v profile_prefix '%q --runtime-trace --stats --output-format rocpd --output-directory %q --' \
        "$ROCPROFV3" "$profile_raw"
      run_logged "$router_log" timeout --kill-after=2m \
        "${PROFILE_TIMEOUT_MINUTES}m" env PATHFINDER_PROFILE_COMMAND="$profile_prefix" \
        "$CANDIDATE_WRAPPER" "${router_args[@]}" || router_status=$?
      ;;
    *) die "unsupported execution mode: $execution_mode" ;;
  esac
  scan_fatal_log "$router_log" "$case_dir/fatal-matches.txt" || router_status=1
  if [[ $execution_mode == profile ]]; then
    scan_profile_loss_log "$router_log" "$case_dir/profile-loss-matches.txt" || \
      router_status=1
  fi
  ((router_status == 0)) || return "$router_status"

  require_file "$output_phys" "routed physical netlist"
  require_file "$routes" "routes JSONL"

  # The repository's physical checker is the primary correctness gate. The
  # JSONL topology/depth validation below is supplemental and cannot replace it.
  run_checker "$case_dir" "$output_phys"

  run_logged "$case_dir/routes-summary.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/summarize_routes_jsonl.py" \
    "$routes" --require-all-routed --summary-out "$case_dir/routes-summary.json"
  assert_route_summary "$case_dir/routes-summary.json" "$expected_count" \
    >"$case_dir/routes-summary-gate.txt"

  if ((require_depth == 1)); then
    run_logged "$case_dir/depth-validation.log" "$PYTHON" \
      "$REPO_ROOT/CongestionFreeRouting/profiling/validate_route_depth.py" \
      "$routes" --expected-max-depth "$EXPECTED_DEPTH" \
      --expected-route-count "$expected_count" \
      --summary-out "$case_dir/depth-summary.json"
  fi

  if ((telemetry == 1)); then
    local -a telemetry_args=(
      "$REPO_ROOT/CongestionFreeRouting/profiling/validate_delta_telemetry.py"
      "$router_log"
      --expected-controller "$telemetry_mode"
      --expected-batch "$batch"
      --expected-workers "$workers"
      --summary-out "$case_dir/telemetry-summary.json"
    )
    if [[ $controller == reduced-round-trip ]]; then
      telemetry_args+=(--require-device-work)
    fi
    run_logged "$case_dir/telemetry-validation.log" "$PYTHON" "${telemetry_args[@]}"
    assert_telemetry_completed "$case_dir/telemetry-summary.json" \
      >"$case_dir/telemetry-query-gate.txt"
  fi

  if [[ -n $reference_routes ]]; then
    require_file "$reference_routes" "reference routes JSONL"
    compare_route_endpoints "$reference_routes" "$routes" \
      "$case_dir/endpoint-comparison.json" >"$case_dir/endpoint-comparison.log"
  fi

  sha256sum -- "$CANDIDATE_WRAPPER" "$pathfinder_bin" "$INTERCHANGE_TO_CSR" \
    "$ROUTES_TO_PHYS" "$output_phys" "$routes" >"$case_dir/artifact-sha256.txt"
  mark_done "$marker" "$case_dir"
  CASE_RESULT=$case_dir
}

case_routes() {
  printf '%s/work/routed.routes.jsonl\n' "$1"
}

assert_fewer_round_trips() {
  local host_summary=$1
  local reduced_summary=$2
  local output=$3
  "$PYTHON" - "$host_summary" "$reduced_summary" "$output" <<'PY'
import json
import pathlib
import sys

host = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
reduced = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
host_round_trips = host.get("controller_round_trips")
reduced_round_trips = reduced.get("controller_round_trips")
result = {
    "host_controller_round_trips": host_round_trips,
    "reduced_controller_round_trips": reduced_round_trips,
    "reduction_percent": (
        100.0 * (host_round_trips - reduced_round_trips) / host_round_trips
        if isinstance(host_round_trips, int) and host_round_trips > 0 else None
    ),
}
pathlib.Path(sys.argv[3]).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not isinstance(host_round_trips, int) or not isinstance(reduced_round_trips, int):
    raise RuntimeError("controller round-trip counters are missing")
if reduced_round_trips >= host_round_trips:
    raise RuntimeError(
        f"reduced controller did not lower round trips: "
        f"host={host_round_trips}, reduced={reduced_round_trips}"
    )
print(json.dumps(result, sort_keys=True))
PY
}

write_case_report() {
  local phase_name=$1
  local output=$2
  shift 2
  "$PYTHON" - "$phase_name" "$output" "$@" <<'PY'
import json
import pathlib
import sys

phase = sys.argv[1]
output = pathlib.Path(sys.argv[2])
arguments = sys.argv[3:]
if len(arguments) % 2:
    raise RuntimeError("case report arguments must be label/directory pairs")
cases = {}
for label, raw_directory in zip(arguments[0::2], arguments[1::2]):
    directory = pathlib.Path(raw_directory).resolve()
    route_summary_path = directory / "routes-summary.json"
    checker_marker = directory / "checker.marker"
    route_summary = json.loads(route_summary_path.read_text(encoding="utf-8"))
    depth_path = (
        directory / "depth-summary.json"
        if (directory / "depth-summary.json").is_file()
        else directory / "checker-depth-summary.json"
    )
    item = {
        "directory": str(directory),
        "checker": checker_marker.read_text(encoding="utf-8").strip(),
        "route_summary": route_summary,
        "route_sha256": route_summary["sha256"],
        "depth_validated": depth_path.is_file(),
        "telemetry_validated": (directory / "telemetry-summary.json").is_file(),
    }
    for name, filename in (
        ("depth", "depth-summary.json"),
        ("checker_splice_depth", "checker-depth-summary.json"),
        ("checker_splice_routes", "checker-routes-summary.json"),
        ("splice_provenance", "splice-summary.json"),
        ("telemetry", "telemetry-summary.json"),
        ("endpoint_comparison", "endpoint-comparison.json"),
    ):
        path = directory / filename
        if path.is_file():
            item[name] = json.loads(path.read_text(encoding="utf-8"))
    cases[label] = item
report = {
    "schema_version": 1,
    "phase": phase,
    "physical_checker_is_primary_gate": True,
    "raw_route_sha_is_informational": True,
    "raw_sha_may_differ_because_equal_distance_parent_ties_are_allowed": True,
    "cases": cases,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

require_completed_phase() {
  local number=$1
  local marker=$RUN_DIR/state/phase${number}.done
  done_value "$marker" || die "phase $number must complete before this phase"
}

prepare_smoke_csr() {
  local marker=$RUN_DIR/state/steps/phase1-shared-smoke-csr.done
  if done_value "$marker"; then
    SHARED_CSR_DIR=$DONE_VALUE
    note "skip completed shared smoke CSR conversion: $SHARED_CSR_DIR"
    return 0
  fi
  CURRENT_CONTEXT='phase 1 shared smoke CSR conversion'
  new_attempt attempts phase1/shared-smoke-csr
  local attempt=$ATTEMPT_DIR
  local log=$attempt/converter.log
  local csr=$attempt/logicnets.shared.csrbin
  local metadata=$attempt/logicnets.shared.csrbin.ifmeta.bin
  run_logged "$log" timeout --kill-after=2m \
    "${FULL_TIMEOUT_MINUTES}m" "$INTERCHANGE_TO_CSR" "$DEVICE_GRAPH" "$INPUT_PHYS" \
    "$LOGICAL_NETLIST" "$csr" --metadata "$metadata"
  require_file "$csr" "shared smoke CSR"
  require_file "$metadata" "shared smoke metadata"
  scan_fatal_log "$log" "$attempt/fatal-matches.txt"
  sha256sum -- "$csr" "$metadata" >"$attempt/artifact-sha256.txt"
  # All smoke PathFinder processes read these exact immutable artifacts. This
  # preserves the routes/metadata artifact_pair_id contract for later splicing.
  chmod a-w -- "$csr" "$metadata" 2>/dev/null || true
  mark_done "$marker" "$attempt"
  SHARED_CSR_DIR=$attempt
}

load_shared_csr() {
  local marker=$RUN_DIR/state/steps/phase1-shared-smoke-csr.done
  done_value "$marker" || die "shared smoke CSR conversion has not completed"
  SHARED_CSR_DIR=$DONE_VALUE
  SHARED_CSR=$SHARED_CSR_DIR/logicnets.shared.csrbin
  SHARED_METADATA=$SHARED_CSR_DIR/logicnets.shared.csrbin.ifmeta.bin
  require_file "$SHARED_CSR" "shared smoke CSR"
  require_file "$SHARED_METADATA" "shared smoke metadata"
}

build_smoke_full_reference() {
  local marker=$RUN_DIR/state/cases/phase1/smoke-full-reference/host-w4.done
  if done_value "$marker"; then
    SMOKE_FULL_REFERENCE_DIR=$DONE_VALUE
    note "skip completed smoke full reference: $SMOKE_FULL_REFERENCE_DIR"
    return 0
  fi
  CURRENT_CONTEXT='phase 1 smoke full-reference bootstrap'
  load_shared_csr
  new_attempt cases phase1/smoke-full-reference/host-w4
  local case_dir=$ATTEMPT_DIR
  local work=$case_dir/work
  local routes=$work/routed.routes.jsonl
  local output_phys=$case_dir/routed.phys
  local router_log=$case_dir/router.log
  mkdir -p -- "$work"
  note "bootstrap full host-w4 reference for smoke checker splices: $case_dir"

  local status=0
  run_logged "$router_log" timeout --kill-after=2m \
    "${FULL_TIMEOUT_MINUTES}m" "$CANDIDATE_PATHFINDER" "$SHARED_CSR" "$SHARED_METADATA" \
    --routes-out "$routes" --sssp-engine delta-step --delta 1 --delta-force-generic \
    --delta-controller host-checked --parallel-net-workers 4 --delta-telemetry || status=$?
  scan_fatal_log "$router_log" "$case_dir/fatal-matches.txt" || status=1
  ((status == 0)) || return "$status"
  require_file "$routes" "full smoke reference routes JSONL"

  run_logged "$case_dir/routes-summary.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/summarize_routes_jsonl.py" \
    "$routes" --require-all-routed --summary-out "$case_dir/routes-summary.json"
  assert_route_summary "$case_dir/routes-summary.json" "$EXPECTED_ROUTES" \
    >"$case_dir/routes-summary-gate.txt"
  run_logged "$case_dir/depth-validation.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/validate_route_depth.py" \
    "$routes" --expected-max-depth "$EXPECTED_DEPTH" \
    --expected-route-count "$EXPECTED_ROUTES" --summary-out "$case_dir/depth-summary.json"
  run_logged "$case_dir/telemetry-validation.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/validate_delta_telemetry.py" \
    "$router_log" --expected-controller host_checked --expected-batch 4 \
    --expected-workers 4 --summary-out "$case_dir/telemetry-summary.json"
  assert_telemetry_completed "$case_dir/telemetry-summary.json" \
    >"$case_dir/telemetry-query-gate.txt"
  run_logged "$case_dir/reconstruct.log" timeout --kill-after=2m \
    "${FULL_TIMEOUT_MINUTES}m" "$ROUTES_TO_PHYS" "$INPUT_PHYS" \
    "$SHARED_METADATA" "$routes" "$output_phys"
  scan_fatal_log "$case_dir/reconstruct.log" "$case_dir/reconstruct-fatal-matches.txt"
  run_checker "$case_dir" "$output_phys"
  sha256sum -- "$SHARED_CSR" "$SHARED_METADATA" "$CANDIDATE_PATHFINDER" \
    "$routes" "$output_phys" >"$case_dir/artifact-sha256.txt"
  printf '%s\n' \
    'This full host-checked artifact exists only to fill nets untouched by each 500-route smoke case; candidate records still replace by unique net name and the full splice is rechecked.' \
    >"$case_dir/reference-scope.txt"
  mark_done "$marker" "$case_dir"
  SMOKE_FULL_REFERENCE_DIR=$case_dir
}

run_smoke_case() {
  local case_id=$1
  local controller=$2
  local batch=$3
  local workers=$4
  local reference_partial=$5
  local marker=$RUN_DIR/state/cases/$case_id.done
  if done_value "$marker"; then
    CASE_RESULT=$DONE_VALUE
    note "skip completed smoke case $case_id: $CASE_RESULT"
    return 0
  fi
  CURRENT_CONTEXT="smoke case $case_id"
  load_shared_csr
  local full_reference_routes=$SMOKE_FULL_REFERENCE_DIR/work/routed.routes.jsonl
  require_file "$full_reference_routes" "full smoke reference routes"
  new_attempt cases "$case_id"
  local case_dir=$ATTEMPT_DIR
  local work=$case_dir/work
  local routes=$work/routed.routes.jsonl
  local spliced_routes=$case_dir/checker-full.routes.jsonl
  local checker_phys=$case_dir/checker-full.phys
  local router_log=$case_dir/router.log
  mkdir -p -- "$work"
  note "run smoke case $case_id against shared immutable CSR: $case_dir"

  local telemetry_mode status=0
  if [[ $controller == host-checked ]]; then
    telemetry_mode=host_checked
  else
    telemetry_mode=reduced_round_trip
  fi
  local -a smoke_args=(
    "$SHARED_CSR" "$SHARED_METADATA" --routes-out "$routes"
    --sssp-engine delta-step --delta 1 --delta-force-generic
    --delta-controller "$controller" --parallel-net-workers "$workers"
    --net-limit "$SMOKE_NET_LIMIT" --delta-telemetry
  )
  if [[ $controller == reduced-round-trip ]]; then
    smoke_args+=(--delta-controller-batch-size "$batch")
  fi
  run_logged "$router_log" timeout --kill-after=2m \
    "${SMOKE_TIMEOUT_MINUTES}m" "$CANDIDATE_PATHFINDER" "${smoke_args[@]}" || status=$?
  scan_fatal_log "$router_log" "$case_dir/fatal-matches.txt" || status=1
  ((status == 0)) || return "$status"
  require_file "$routes" "partial smoke routes JSONL"

  run_logged "$case_dir/routes-summary.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/summarize_routes_jsonl.py" \
    "$routes" --require-all-routed --summary-out "$case_dir/routes-summary.json"
  assert_route_summary "$case_dir/routes-summary.json" "$SMOKE_NET_LIMIT" \
    >"$case_dir/routes-summary-gate.txt"
  local -a telemetry_args=(
    "$REPO_ROOT/CongestionFreeRouting/profiling/validate_delta_telemetry.py"
    "$router_log" --expected-controller "$telemetry_mode" --expected-batch "$batch"
    --expected-workers "$workers" --summary-out "$case_dir/telemetry-summary.json"
  )
  if [[ $controller == reduced-round-trip ]]; then
    telemetry_args+=(--require-device-work)
  fi
  run_logged "$case_dir/telemetry-validation.log" "$PYTHON" "${telemetry_args[@]}"
  assert_telemetry_completed "$case_dir/telemetry-summary.json" \
    >"$case_dir/telemetry-query-gate.txt"
  if [[ -n $reference_partial ]]; then
    compare_route_endpoints "$reference_partial" "$routes" \
      "$case_dir/endpoint-comparison.json" >"$case_dir/endpoint-comparison.log"
  fi

  # The splice helper validates unique net names and artifact_pair_id, replaces
  # exactly 500 records by name, and preserves full-reference record order.
  run_logged "$case_dir/splice.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/splice_routes_jsonl.py" \
    "$full_reference_routes" "$routes" "$spliced_routes" \
    --expected-replacements "$SMOKE_NET_LIMIT" --summary-out "$case_dir/splice-summary.json"
  run_logged "$case_dir/checker-routes-summary.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/summarize_routes_jsonl.py" \
    "$spliced_routes" --require-all-routed \
    --summary-out "$case_dir/checker-routes-summary.json"
  assert_route_summary "$case_dir/checker-routes-summary.json" "$EXPECTED_ROUTES" \
    >"$case_dir/checker-routes-summary-gate.txt"
  run_logged "$case_dir/checker-depth-validation.log" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/validate_route_depth.py" \
    "$spliced_routes" --expected-max-depth "$EXPECTED_DEPTH" \
    --expected-route-count "$EXPECTED_ROUTES" \
    --summary-out "$case_dir/checker-depth-summary.json"
  run_logged "$case_dir/reconstruct.log" timeout --kill-after=2m \
    "${FULL_TIMEOUT_MINUTES}m" "$ROUTES_TO_PHYS" "$INPUT_PHYS" \
    "$SHARED_METADATA" "$spliced_routes" "$checker_phys"
  scan_fatal_log "$case_dir/reconstruct.log" "$case_dir/reconstruct-fatal-matches.txt"
  run_checker "$case_dir" "$checker_phys"
  sha256sum -- "$SHARED_CSR" "$SHARED_METADATA" "$full_reference_routes" \
    "$routes" "$spliced_routes" "$checker_phys" >"$case_dir/artifact-sha256.txt"
  mark_done "$marker" "$case_dir"
  CASE_RESULT=$case_dir
}

phase1() {
  local phase_marker=$RUN_DIR/state/phase1.done
  if done_value "$phase_marker"; then
    local completed_report=$DONE_VALUE
    verify_phase1_inputs
    note "phase 1 already complete: $completed_report"
    return 0
  fi
  CURRENT_CONTEXT='phase 1'
  preflight
  preserve_preexisting_binaries
  build_candidates
  load_normal_build
  run_low_level_tests
  prepare_smoke_csr
  load_shared_csr
  build_smoke_full_reference

  local host_w1 host_w4 reduced_b2w1 reduced_b4w1
  local reduced_b2w4 reduced_b4w4 reduced_b8w4 reduced_b4w8 reference

  run_smoke_case phase1/smoke/host-w1 host-checked 4 1 ''
  host_w1=$CASE_RESULT
  reference=$(case_routes "$host_w1")

  run_smoke_case phase1/smoke/host-w4 host-checked 4 4 "$reference"
  host_w4=$CASE_RESULT

  run_smoke_case phase1/smoke/reduced-b2w1 reduced-round-trip 2 1 "$reference"
  reduced_b2w1=$CASE_RESULT
  run_smoke_case phase1/smoke/reduced-b4w1 reduced-round-trip 4 1 "$reference"
  reduced_b4w1=$CASE_RESULT
  run_smoke_case phase1/smoke/reduced-b2w4 reduced-round-trip 2 4 "$reference"
  reduced_b2w4=$CASE_RESULT
  run_smoke_case phase1/smoke/reduced-b4w4 reduced-round-trip 4 4 "$reference"
  reduced_b4w4=$CASE_RESULT
  run_smoke_case phase1/smoke/reduced-b8w4 reduced-round-trip 8 4 "$reference"
  reduced_b8w4=$CASE_RESULT
  run_smoke_case phase1/smoke/reduced-b4w8 reduced-round-trip 4 8 "$reference"
  reduced_b4w8=$CASE_RESULT

  record_or_verify_source_fingerprint
  new_attempt reports phase1
  local report=$ATTEMPT_DIR
  assert_fewer_round_trips "$host_w1/telemetry-summary.json" \
    "$reduced_b2w1/telemetry-summary.json" "$report/roundtrips-host-w1-vs-reduced-b2w1.json"
  assert_fewer_round_trips "$host_w1/telemetry-summary.json" \
    "$reduced_b4w1/telemetry-summary.json" "$report/roundtrips-host-w1-vs-reduced-b4w1.json"
  assert_fewer_round_trips "$host_w4/telemetry-summary.json" \
    "$reduced_b2w4/telemetry-summary.json" "$report/roundtrips-host-w4-vs-reduced-b2w4.json"
  assert_fewer_round_trips "$host_w4/telemetry-summary.json" \
    "$reduced_b4w4/telemetry-summary.json" "$report/roundtrips-host-w4-vs-reduced-b4w4.json"
  assert_fewer_round_trips "$host_w4/telemetry-summary.json" \
    "$reduced_b8w4/telemetry-summary.json" "$report/roundtrips-host-w4-vs-reduced-b8w4.json"
  printf '%s\n' \
    'No host-w8 case exists in the required smoke matrix; reduced-b4w8 is still gated by telemetry, checker, route counts, and host-w1 endpoints.' \
    >"$report/reduced-b4w8-roundtrip-note.txt"
  printf '%s\n' \
    'Each raw smoke artifact contains 500 routes. The full-reference splice is separately required to contain all expected routes, have exact maximum depth 214, reconstruct successfully, and PASS CheckPhysNetlist.' \
    >"$report/depth-gate-note.txt"
  write_case_report phase1-smoke "$report/summary.json" \
    host-w1 "$host_w1" host-w4 "$host_w4" \
    reduced-b2w1 "$reduced_b2w1" reduced-b4w1 "$reduced_b4w1" \
    reduced-b2w4 "$reduced_b2w4" reduced-b4w4 "$reduced_b4w4" \
    reduced-b8w4 "$reduced_b8w4" reduced-b4w8 "$reduced_b4w8"
  write_case_report phase1-smoke-reference "$report/full-reference-summary.json" \
    host-w4 "$SMOKE_FULL_REFERENCE_DIR"
  write_dependency_manifest "$report/.harness-dependencies-v1" \
    "$RUN_DIR/state/steps/phase1-preflight.done" \
    "$RUN_DIR/state/steps/phase1-preexisting-binaries.done" \
    "$RUN_DIR/state/steps/phase1-build.done" \
    "$RUN_DIR/state/steps/phase1/tests/python-argument-forwarding.done" \
    "$RUN_DIR/state/steps/phase1/tests/profiling-validation.done" \
    "$RUN_DIR/state/steps/phase1/tests/delta-policy.done" \
    "$RUN_DIR/state/steps/phase1/tests/router-args.done" \
    "$RUN_DIR/state/steps/phase1/tests/pathfinder-cpu-stub.done" \
    "$RUN_DIR/state/steps/phase1/tests/unit-bfs-hip.done" \
    "$RUN_DIR/state/steps/phase1/tests/unit-bfs-reuse-stress.done" \
    "$RUN_DIR/state/steps/phase1/tests/delta-reduced-hip.done" \
    "$RUN_DIR/state/steps/phase1/tests/delta-reduced-multi-queue-stress.done" \
    "$RUN_DIR/state/steps/phase1-shared-smoke-csr.done" \
    "$RUN_DIR/state/cases/phase1/smoke-full-reference/host-w4.done" \
    "$RUN_DIR/state/cases/phase1/smoke/host-w1.done" \
    "$RUN_DIR/state/cases/phase1/smoke/host-w4.done" \
    "$RUN_DIR/state/cases/phase1/smoke/reduced-b2w1.done" \
    "$RUN_DIR/state/cases/phase1/smoke/reduced-b4w1.done" \
    "$RUN_DIR/state/cases/phase1/smoke/reduced-b2w4.done" \
    "$RUN_DIR/state/cases/phase1/smoke/reduced-b4w4.done" \
    "$RUN_DIR/state/cases/phase1/smoke/reduced-b8w4.done" \
    "$RUN_DIR/state/cases/phase1/smoke/reduced-b4w8.done"
  mark_done "$phase_marker" "$report"
  note "phase 1 complete: $report"
}

phase2() {
  local phase_marker=$RUN_DIR/state/phase2.done
  if done_value "$phase_marker"; then
    local completed_report=$DONE_VALUE
    verify_phase1_inputs
    note "phase 2 already complete: $completed_report"
    return 0
  fi
  CURRENT_CONTEXT='phase 2'
  require_completed_phase 1
  verify_phase1_inputs
  load_normal_build

  local host_full reduced_full reference
  run_router_case phase2/full-correctness/host-w4 host-checked 4 4 0 1 1 \
    "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" normal ''
  host_full=$CASE_RESULT
  reference=$(case_routes "$host_full")
  run_router_case phase2/full-correctness/reduced-b4w4 reduced-round-trip 4 4 0 1 1 \
    "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" normal "$reference"
  reduced_full=$CASE_RESULT

  record_or_verify_source_fingerprint
  new_attempt reports phase2
  local report=$ATTEMPT_DIR
  assert_fewer_round_trips "$host_full/telemetry-summary.json" \
    "$reduced_full/telemetry-summary.json" "$report/roundtrips-host-w4-vs-reduced-b4w4.json"
  write_case_report phase2-full-correctness "$report/summary.json" \
    host-w4 "$host_full" reduced-b4w4 "$reduced_full"
  write_dependency_manifest "$report/.harness-dependencies-v1" \
    "$RUN_DIR/state/phase1.done" \
    "$RUN_DIR/state/cases/phase2/full-correctness/host-w4.done" \
    "$RUN_DIR/state/cases/phase2/full-correctness/reduced-b4w4.done"
  mark_done "$phase_marker" "$report"
  note "phase 2 complete: $report"
}

phase3_preflight() {
  CURRENT_CONTEXT='phase 3 preflight'
  ((CONFIRM_GPU_IDLE == 1)) ||
    die "phase 3 requires --confirm-gpu-idle after the operator stops unrelated GPU workloads"
  require_command "$ROCPROFV3"
  require_executable /usr/bin/time "GNU time"
  require_command df

  local available_kib configured_kib required_kib phase2_case_kib
  local planned_full_cases rocpd_reserve_kib estimated_kib phase2_host_marker
  available_kib=$(df -Pk "$RUN_DIR" | awk 'NR == 2 {print $4}')
  [[ $available_kib =~ ^[0-9]+$ ]] || die "could not determine free disk space"
  configured_kib=$((MIN_FREE_GIB * 1024 * 1024))
  phase2_host_marker=$RUN_DIR/state/cases/phase2/full-correctness/host-w4.done
  done_value "$phase2_host_marker" || die "phase 2 host case is missing for disk estimation"
  phase2_case_kib=$(du -sk -- "$DONE_VALUE" | awk '{print $1}')
  [[ $phase2_case_kib =~ ^[0-9]+$ ]] || die "could not size the phase 2 host case"
  planned_full_cases=$((6 + 4 * TIMING_REPETITIONS))
  # Two runtime traces can be several GiB each. Reserve 40 GiB beyond a
  # per-case projection based on the completed phase-2 wrapper artifact.
  rocpd_reserve_kib=$((40 * 1024 * 1024))
  estimated_kib=$((phase2_case_kib * planned_full_cases + rocpd_reserve_kib))
  required_kib=$configured_kib
  ((estimated_kib <= required_kib)) || required_kib=$estimated_kib
  ((available_kib >= required_kib)) ||
    die "phase 3 requires at least $(((required_kib + 1024 * 1024 - 1) / 1024 / 1024)) GiB free (configured floor plus case-size estimate and 40 GiB RocPD reserve); only $((available_kib / 1024 / 1024)) GiB is available"

  new_attempt attempts phase3/session-preflight
  local attempt=$ATTEMPT_DIR
  local log=$attempt/preflight.log
  printf 'confirmed_by_operator=true\nconfirmed_at=%s\n' "$(timestamp_utc)" \
    >"$attempt/gpu-idle-confirmation.txt"
  {
    printf 'available_kib=%s\nconfigured_minimum_kib=%s\n' "$available_kib" "$configured_kib"
    printf 'phase2_host_case_kib=%s\nplanned_full_cases=%s\n' \
      "$phase2_case_kib" "$planned_full_cases"
    printf 'rocpd_reserve_kib=%s\nestimated_required_kib=%s\n' \
      "$rocpd_reserve_kib" "$estimated_kib"
    printf 'effective_required_kib=%s\n' "$required_kib"
  } >"$attempt/disk-estimate.txt"
  run_logged "$log" "$ROCPROFV3" --version
  run_logged "$log" df -h "$RUN_DIR"
  if command -v rocm-smi >/dev/null 2>&1; then
    run_logged "$log" rocm-smi --showproductname --showuse --showmemuse
  fi
  scan_fatal_log "$log" "$attempt/fatal-matches.txt"
}

build_roctx_candidate() {
  local marker=$RUN_DIR/state/steps/phase3-roctx-build.done
  if done_value "$marker"; then
    ROCTX_BUILD_DIR=$DONE_VALUE
    note "skip completed ROCTx candidate build: $ROCTX_BUILD_DIR"
    return 0
  fi
  CURRENT_CONTEXT='phase 3 ROCTx build'
  new_attempt attempts phase3/roctx-build
  local attempt=$ATTEMPT_DIR
  local bin=$attempt/bin
  local log=$attempt/build.log
  mkdir -p -- "$bin"
  run_logged "$log" "$HIPCC" -std=c++17 -O3 -x hip -DBF10_NO_MAIN \
    -DPATHFINDER_ENABLE_ROCTX \
    -I "$REPO_ROOT/HIP_kernel/bellman_ford/src" \
    -I "$REPO_ROOT/CongestionFreeRouting/bellman_ford" \
    -I "$REPO_ROOT/CongestionFreeRouting/delta_stepping" \
    -I "$REPO_ROOT/CongestionFreeRouting/unit_bfs" \
    "$REPO_ROOT/CongestionFreeRouting/pathfinder.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/bellman_ford/bf10.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp" \
    "$REPO_ROOT/CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp" \
    -pthread -lrocprofiler-sdk-roctx -o "$bin/pathfinder-roctx-candidate"
  sha256sum -- "$bin/pathfinder-roctx-candidate" >"$attempt/binary-sha256.txt"
  scan_fatal_log "$log" "$attempt/fatal-matches.txt"
  mark_done "$marker" "$attempt"
  ROCTX_BUILD_DIR=$attempt
}

load_roctx_build() {
  local marker=$RUN_DIR/state/steps/phase3-roctx-build.done
  done_value "$marker" || die "ROCTx candidate build has not completed"
  ROCTX_BUILD_DIR=$DONE_VALUE
  ROCTX_PATHFINDER=$ROCTX_BUILD_DIR/bin/pathfinder-roctx-candidate
  require_executable "$ROCTX_PATHFINDER" "ROCTx pathfinder candidate"
}

compute_timing_summary() {
  local label=$1
  local output=$2
  shift 2
  "$PYTHON" - "$label" "$output" "$@" <<'PY'
import json
import pathlib
import statistics
import sys

label = sys.argv[1]
output = pathlib.Path(sys.argv[2])
directories = [pathlib.Path(value).resolve() for value in sys.argv[3:]]
if len(directories) < 3:
    raise RuntimeError("at least three timing repetitions are required")
repetitions = []
for index, directory in enumerate(directories, 1):
    values = {}
    for line in (directory / "time.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    wall = float(values["wall_seconds"])
    rss = int(values["peak_rss_kib"])
    route_summary = json.loads(
        (directory / "routes-summary.json").read_text(encoding="utf-8")
    )
    repetitions.append({
        "repetition": index,
        "directory": str(directory),
        "wall_seconds": wall,
        "peak_rss_kib": rss,
        "route_sha256": route_summary["sha256"],
    })
walls = [item["wall_seconds"] for item in repetitions]
rss_values = [item["peak_rss_kib"] for item in repetitions]
summary = {
    "schema_version": 1,
    "configuration": label,
    "repetition_count": len(repetitions),
    "median_wall_seconds": statistics.median(walls),
    "minimum_wall_seconds": min(walls),
    "maximum_wall_seconds": max(walls),
    "median_peak_rss_kib": statistics.median(rss_values),
    "repetitions": repetitions,
    "profiler_enabled": False,
    "telemetry_enabled": False,
}
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

gate_timing_regression() {
  local host_summary=$1
  local reduced_summary=$2
  local output=$3
  "$PYTHON" - "$host_summary" "$reduced_summary" \
    "$MAX_TIMING_REGRESSION_PERCENT" "$output" <<'PY'
import json
import pathlib
import sys

host = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
reduced = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
limit = float(sys.argv[3])
host_median = float(host["median_wall_seconds"])
reduced_median = float(reduced["median_wall_seconds"])
regression = 100.0 * (reduced_median - host_median) / host_median
result = {
    "host_median_wall_seconds": host_median,
    "reduced_b4_median_wall_seconds": reduced_median,
    "reduced_b4_regression_percent": regression,
    "maximum_allowed_regression_percent": limit,
    "passed": regression <= limit,
}
pathlib.Path(sys.argv[4]).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if regression > limit:
    raise RuntimeError(
        f"reduced b4 median regressed {regression:.3f}% versus host; limit is {limit:.3f}%"
    )
print(json.dumps(result, sort_keys=True))
PY
}

discover_rocpd_database() {
  local raw_dir=$1
  local output=$2
  local -a candidates=()
  while IFS= read -r -d '' candidate; do
    candidates+=("$candidate")
  done < <(find "$raw_dir" -type f \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' \) -print0)
  "$PYTHON" - "$output" "${candidates[@]}" <<'PY'
import pathlib
import sqlite3
import sys

output = pathlib.Path(sys.argv[1])
required = {"processes", "regions", "kernels", "memory_copies", "region_args"}
valid = []
diagnostics = []
for raw_path in sys.argv[2:]:
    path = pathlib.Path(raw_path).resolve()
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        if required.issubset(names):
            processes = connection.execute("SELECT COUNT(*) FROM processes").fetchone()[0]
            regions = connection.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
            if processes > 0 and regions > 0:
                valid.append(path)
            else:
                diagnostics.append(f"{path}: empty processes/regions")
        else:
            diagnostics.append(f"{path}: missing {sorted(required - names)!r}")
        connection.close()
    except Exception as error:
        diagnostics.append(f"{path}: {error}")
if len(valid) != 1:
    raise RuntimeError(
        f"expected exactly one nonempty RocPD database, found {len(valid)}; "
        + "; ".join(diagnostics)
    )
output.write_text(str(valid[0]) + "\n", encoding="utf-8")
print(valid[0])
PY
}

gate_sync_summary() {
  local controller=$1
  local analysis_summary=$2
  local route_summary=$3
  local output=$4
  "$PYTHON" - "$controller" "$analysis_summary" "$route_summary" "$output" <<'PY'
import json
import math
import pathlib
import sys

controller = sys.argv[1]
analysis = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
routes = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
completed = routes.get("route_requests")
if not isinstance(completed, int) or completed <= 0:
    raise RuntimeError("strict route summary has no positive route_requests denominator")
if routes.get("routed") != completed or routes.get("unrouted") != 0:
    raise RuntimeError("strict route summary is not fully routed")
if routes.get("reached_sinks") != routes.get("sinks"):
    raise RuntimeError("strict route summary contains unreached sinks")

scopes = analysis["hip_api"]["stream_synchronize_scopes"]
counts = {name: scopes.get(name) for name in (
    "total", "generic_controller", "compact_edge_path_extraction", "other"
)}
if any(type(value) is not int or value < 0 for value in counts.values()):
    raise RuntimeError(f"invalid synchronization scope counts: {counts!r}")
api_total = analysis["hip_api"]["by_name"].get("hipStreamSynchronize", {}).get("calls")
if api_total != counts["total"]:
    raise RuntimeError(
        f"scope total {counts['total']} does not equal HIP API total {api_total}"
    )
partition = counts["generic_controller"] + counts["compact_edge_path_extraction"] + counts["other"]
if partition != counts["total"]:
    raise RuntimeError(
        f"sync scopes do not partition total: partition={partition}, total={counts['total']}"
    )
per_route = {name: value / completed for name, value in counts.items()}
reported_completed = scopes.get("completed_routes")
if reported_completed is not None and reported_completed != completed:
    raise RuntimeError(
        f"analyzer denominator mismatch: routes={completed}, analyzer={reported_completed}"
    )
reported_rates = scopes.get("per_route", {})
for name, value in per_route.items():
    if name in reported_rates and not math.isclose(
        float(reported_rates[name]), value, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"analyzer {name} rate mismatch: {reported_rates[name]!r} versus {value}"
        )

failures = []
if controller == "reduced-round-trip":
    if per_route["total"] > 15.0:
        failures.append(f"total syncs/route {per_route['total']:.6f} exceeds 15")
    if per_route["generic_controller"] > 10.0:
        failures.append(
            f"generic syncs/route {per_route['generic_controller']:.6f} exceeds 10"
        )
result = {
    "schema_version": 1,
    "controller": controller,
    "completed_routes_from_strict_summary": completed,
    "counts": counts,
    "per_route": per_route,
    "limits": {
        "total_syncs_per_route": 15.0 if controller == "reduced-round-trip" else None,
        "generic_syncs_per_route": 10.0 if controller == "reduced-round-trip" else None,
    },
    "compact_edge_path_extraction_reported_separately": True,
    "passed": not failures,
    "failures": failures,
}
pathlib.Path(sys.argv[4]).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if failures:
    raise RuntimeError("synchronization milestone failed:\n- " + "\n- ".join(failures))
print(json.dumps(result, sort_keys=True))
PY
}

analyze_profile() {
  local profile_id=$1
  local controller=$2
  local case_dir=$3
  local marker=$RUN_DIR/state/profile-analysis/$profile_id.done
  if done_value "$marker"; then
    ANALYSIS_RESULT=$DONE_VALUE
    note "skip completed profile analysis $profile_id: $ANALYSIS_RESULT"
    return 0
  fi
  CURRENT_CONTEXT="profile analysis $profile_id"
  new_attempt profile-analysis "$profile_id"
  local attempt=$ATTEMPT_DIR
  local raw_dir=$case_dir/profile-raw
  local database_file=$attempt/database-path.txt
  discover_rocpd_database "$raw_dir" "$database_file" >"$attempt/database-discovery.log"
  local database
  IFS= read -r database <"$database_file"
  mkdir -p -- "$attempt/analysis"
  # --routes-summary makes the synchronization denominator originate from the
  # already-gated full route artifact, not a caller-provided integer.
  run_logged "$attempt/analyzer.log" timeout --kill-after=2m \
    "${PROFILE_TIMEOUT_MINUTES}m" "$PYTHON" \
    "$REPO_ROOT/CongestionFreeRouting/profiling/analyze_rocpd.py" \
    "$database" --output-dir "$attempt/analysis" --no-plots \
    --routes-summary "$case_dir/routes-summary.json" --require-delta-scopes
  gate_sync_summary "$controller" "$attempt/analysis/summary.json" \
    "$case_dir/routes-summary.json" "$attempt/synchronization-gate.json" \
    >"$attempt/synchronization-gate.log"
  mark_done "$marker" "$attempt"
  ANALYSIS_RESULT=$attempt
}

write_phase3_report() {
  local output=$1
  shift
  "$PYTHON" - "$output" "$@" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
(
    host_timing,
    b2_timing,
    b4_timing,
    b8_timing,
    timing_gate,
    host_sync,
    reduced_sync,
) = [pathlib.Path(value) for value in sys.argv[2:]]
report = {
    "schema_version": 1,
    "phase": "phase3-performance-and-profile",
    "timing": {
        "host-w4": json.loads(host_timing.read_text(encoding="utf-8")),
        "reduced-b2w4": json.loads(b2_timing.read_text(encoding="utf-8")),
        "reduced-b4w4": json.loads(b4_timing.read_text(encoding="utf-8")),
        "reduced-b8w4": json.loads(b8_timing.read_text(encoding="utf-8")),
        "b4_regression_gate": json.loads(timing_gate.read_text(encoding="utf-8")),
    },
    "rocprofv3": {
        "host-w4": json.loads(host_sync.read_text(encoding="utf-8")),
        "reduced-b4w4": json.loads(reduced_sync.read_text(encoding="utf-8")),
        "profiled_timings_are_not_used_as_wall_clock_results": True,
    },
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

phase3() {
  local phase_marker=$RUN_DIR/state/phase3.done
  if done_value "$phase_marker"; then
    local completed_report=$DONE_VALUE
    verify_phase1_inputs
    note "phase 3 already complete: $completed_report"
    return 0
  fi
  CURRENT_CONTEXT='phase 3'
  require_completed_phase 2
  verify_phase1_inputs
  load_normal_build
  phase3_preflight
  build_roctx_candidate
  load_roctx_build

  local phase2_host_marker=$RUN_DIR/state/cases/phase2/full-correctness/host-w4.done
  done_value "$phase2_host_marker" || die "phase 2 host reference case is missing"
  local phase2_host=$DONE_VALUE
  local reference
  reference=$(case_routes "$phase2_host")

  local host_warmup b2_warmup b4_warmup b8_warmup
  run_router_case phase3/timing/host-w4/warmup host-checked 4 4 0 0 1 \
    "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" normal "$reference"
  host_warmup=$CASE_RESULT
  run_router_case phase3/timing/reduced-b2w4/warmup reduced-round-trip 2 4 0 0 1 \
    "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" normal "$reference"
  b2_warmup=$CASE_RESULT
  run_router_case phase3/timing/reduced-b4w4/warmup reduced-round-trip 4 4 0 0 1 \
    "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" normal "$reference"
  b4_warmup=$CASE_RESULT
  run_router_case phase3/timing/reduced-b8w4/warmup reduced-round-trip 8 4 0 0 1 \
    "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" normal "$reference"
  b8_warmup=$CASE_RESULT

  local -a host_reps=() b2_reps=() b4_reps=() b8_reps=()
  local repetition rep_label
  for ((repetition = 1; repetition <= TIMING_REPETITIONS; ++repetition)); do
    printf -v rep_label 'rep-%03d' "$repetition"
    run_router_case "phase3/timing/host-w4/$rep_label" host-checked 4 4 0 0 1 \
      "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" timed "$reference"
    host_reps+=("$CASE_RESULT")
    run_router_case "phase3/timing/reduced-b2w4/$rep_label" reduced-round-trip 2 4 0 0 1 \
      "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" timed "$reference"
    b2_reps+=("$CASE_RESULT")
    run_router_case "phase3/timing/reduced-b4w4/$rep_label" reduced-round-trip 4 4 0 0 1 \
      "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" timed "$reference"
    b4_reps+=("$CASE_RESULT")
    run_router_case "phase3/timing/reduced-b8w4/$rep_label" reduced-round-trip 8 4 0 0 1 \
      "$EXPECTED_ROUTES" "$CANDIDATE_PATHFINDER" timed "$reference"
    b8_reps+=("$CASE_RESULT")
  done

  local profile_host profile_reduced host_analysis reduced_analysis
  run_router_case phase3/profile/host-w4 host-checked 4 4 0 0 1 \
    "$EXPECTED_ROUTES" "$ROCTX_PATHFINDER" profile "$reference"
  profile_host=$CASE_RESULT
  analyze_profile host-w4 host-checked "$profile_host"
  host_analysis=$ANALYSIS_RESULT
  run_router_case phase3/profile/reduced-b4w4 reduced-round-trip 4 4 0 0 1 \
    "$EXPECTED_ROUTES" "$ROCTX_PATHFINDER" profile "$reference"
  profile_reduced=$CASE_RESULT
  analyze_profile reduced-b4w4 reduced-round-trip "$profile_reduced"
  reduced_analysis=$ANALYSIS_RESULT

  record_or_verify_source_fingerprint
  new_attempt reports phase3
  local report=$ATTEMPT_DIR
  compute_timing_summary host-w4 "$report/timing-host-w4.json" "${host_reps[@]}"
  compute_timing_summary reduced-b2w4 "$report/timing-reduced-b2w4.json" "${b2_reps[@]}"
  compute_timing_summary reduced-b4w4 "$report/timing-reduced-b4w4.json" "${b4_reps[@]}"
  compute_timing_summary reduced-b8w4 "$report/timing-reduced-b8w4.json" "${b8_reps[@]}"
  gate_timing_regression "$report/timing-host-w4.json" \
    "$report/timing-reduced-b4w4.json" "$report/timing-regression-gate.json" \
    >"$report/timing-regression-gate.log"
  write_case_report phase3-profile "$report/profile-route-summary.json" \
    host-w4 "$profile_host" reduced-b4w4 "$profile_reduced"
  write_phase3_report "$report/summary.json" \
    "$report/timing-host-w4.json" "$report/timing-reduced-b2w4.json" \
    "$report/timing-reduced-b4w4.json" "$report/timing-reduced-b8w4.json" \
    "$report/timing-regression-gate.json" \
    "$host_analysis/synchronization-gate.json" \
    "$reduced_analysis/synchronization-gate.json"
  {
    printf 'host_warmup=%s\nreduced_b2_warmup=%s\n' "$host_warmup" "$b2_warmup"
    printf 'reduced_b4_warmup=%s\nreduced_b8_warmup=%s\n' "$b4_warmup" "$b8_warmup"
    printf 'host_profile_analysis=%s\nreduced_profile_analysis=%s\n' \
      "$host_analysis" "$reduced_analysis"
  } >"$report/artifact-index.txt"
  if command -v rocm-smi >/dev/null 2>&1; then
    run_logged "$report/gpu-state-after.log" rocm-smi --showuse --showmemuse
  fi
  local -a phase3_dependencies=(
    "$RUN_DIR/state/phase2.done"
    "$RUN_DIR/state/steps/phase3-roctx-build.done"
    "$RUN_DIR/state/cases/phase3/timing/host-w4/warmup.done"
    "$RUN_DIR/state/cases/phase3/timing/reduced-b2w4/warmup.done"
    "$RUN_DIR/state/cases/phase3/timing/reduced-b4w4/warmup.done"
    "$RUN_DIR/state/cases/phase3/timing/reduced-b8w4/warmup.done"
    "$RUN_DIR/state/cases/phase3/profile/host-w4.done"
    "$RUN_DIR/state/cases/phase3/profile/reduced-b4w4.done"
    "$RUN_DIR/state/profile-analysis/host-w4.done"
    "$RUN_DIR/state/profile-analysis/reduced-b4w4.done"
  )
  for ((repetition = 1; repetition <= TIMING_REPETITIONS; ++repetition)); do
    printf -v rep_label 'rep-%03d' "$repetition"
    phase3_dependencies+=(
      "$RUN_DIR/state/cases/phase3/timing/host-w4/$rep_label.done"
      "$RUN_DIR/state/cases/phase3/timing/reduced-b2w4/$rep_label.done"
      "$RUN_DIR/state/cases/phase3/timing/reduced-b4w4/$rep_label.done"
      "$RUN_DIR/state/cases/phase3/timing/reduced-b8w4/$rep_label.done"
    )
  done
  write_dependency_manifest "$report/.harness-dependencies-v1" \
    "${phase3_dependencies[@]}"
  mark_done "$phase_marker" "$report"
  note "phase 3 complete: $report"
}

case "$PHASE" in
  1)
    phase1
    ;;
  2)
    phase2
    ;;
  3)
    phase3
    ;;
  all)
    phase1
    phase2
    phase3
    ;;
esac

CURRENT_CONTEXT=complete
note "requested phase '$PHASE' completed successfully"
note "evidence: $RUN_DIR"
