#include "../bellman_ford/bf11_graph_execution_policy.hpp"

#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace policy = bf11_graph_execution_policy;

constexpr int kInfinity = std::numeric_limits<int>::max() / 4;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void relax_one_round(std::vector<int>* distance) {
  const std::vector<int> before = *distance;
  for (std::size_t node = 0; node + 1 < before.size(); ++node) {
    if (before[node] != kInfinity) {
      (*distance)[node + 1] =
          std::min((*distance)[node + 1], before[node] + 1);
    }
  }
}

void run_segment(std::vector<int>* distance) {
  relax_one_round(distance);
  relax_one_round(distance);
}

struct Counters {
  int begin = 0;
  int enqueue = 0;
  int end = 0;
  int instantiate = 0;
  int launch = 0;
  int quiesce = 0;
  int graph_destroy = 0;
  int executable_destroy = 0;
  int fallbacks = 0;
  std::vector<std::string> cleanup_order;
};

struct ConcurrentGate {
  explicit ConcurrentGate(int participants_in)
      : participants(participants_in) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex);
    ++arrived;
    if (arrived == participants) {
      released = true;
      condition.notify_all();
    } else {
      condition.wait(lock, [&] { return released; });
    }
  }

  int participants = 0;
  int arrived = 0;
  bool released = false;
  std::mutex mutex;
  std::condition_variable condition;
};

enum class Failure {
  kNone,
  kBegin,
  kCapturedEnqueue,
  kCapturedEnqueueInvalidates,
  kEndNoGraph,
  kEndPartialGraph,
  kInstantiateNoExecutable,
  kInstantiatePartialExecutable,
  kLaunchPreSubmit,
  kLaunchAfterPartialSubmit,
  kLaunchAfterCachedReplay,
};

class FakeBackend {
 public:
  using Graph = int;
  using Executable = int;

  FakeBackend(std::vector<int>* distance,
              Counters* counters,
              Failure failure = Failure::kNone,
              ConcurrentGate* gate = nullptr)
      : distance_(distance),
        counters_(counters),
        failure_(failure),
        gate_(gate) {}

  ~FakeBackend() { invalidate_cached(); }

  bool disabled() const { return disabled_; }
  bool has_cached_executable() const { return executable_ != 0; }
  Graph null_graph() const { return 0; }
  Executable null_executable() const { return 0; }
  bool valid_graph(Graph graph) const { return graph != 0; }
  bool valid_executable(Executable executable) const {
    return executable != 0;
  }

  bool begin_capture() {
    ++counters_->begin;
    if (gate_ != nullptr) gate_->arrive_and_wait();
    if (failure_ == Failure::kBegin) return false;
    capturing_ = true;
    return true;
  }

  bool enqueue_captured_segment() {
    ++counters_->enqueue;
    return failure_ != Failure::kCapturedEnqueue &&
           failure_ != Failure::kCapturedEnqueueInvalidates;
  }

  bool end_capture(Graph* graph) {
    ++counters_->end;
    require(capturing_, "end-capture ran without a successful begin");
    capturing_ = false;
    if (failure_ == Failure::kCapturedEnqueueInvalidates ||
        failure_ == Failure::kEndNoGraph) {
      *graph = 0;
      return false;
    }
    *graph = next_handle_++;
    return failure_ != Failure::kEndPartialGraph;
  }

  bool instantiate(Executable* executable, Graph) {
    ++counters_->instantiate;
    if (failure_ == Failure::kInstantiateNoExecutable) return false;
    *executable = next_handle_++;
    return failure_ != Failure::kInstantiatePartialExecutable;
  }

  void adopt(Graph graph, Executable executable) {
    graph_ = graph;
    executable_ = executable;
  }

  bool launch_cached() {
    ++counters_->launch;
    if (failure_ == Failure::kLaunchPreSubmit) {
      real_launch_attempted_ = false;
      return false;
    }
    real_launch_attempted_ = true;
    if (failure_ == Failure::kLaunchAfterPartialSubmit ||
        (failure_ == Failure::kLaunchAfterCachedReplay &&
         counters_->launch == 2)) {
      // Model a non-atomic Graph launch that submitted one of two rounds.
      relax_one_round(distance_);
      return false;
    }
    run_segment(distance_);
    return true;
  }

  bool launch_failure_requires_restart() const {
    return real_launch_attempted_;
  }

  void prepare_launch_failure() {
    if (!real_launch_attempted_) return;
    ++counters_->quiesce;
    counters_->cleanup_order.push_back("quiesce");
  }

  void destroy_graph(Graph graph) {
    if (graph == 0) return;
    ++counters_->graph_destroy;
    counters_->cleanup_order.push_back("graph");
  }

  void destroy_executable(Executable executable) {
    if (executable == 0) return;
    ++counters_->executable_destroy;
    counters_->cleanup_order.push_back("executable");
  }

  void invalidate_cached() {
    const Executable executable = executable_;
    const Graph graph = graph_;
    executable_ = 0;
    graph_ = 0;
    destroy_executable(executable);
    destroy_graph(graph);
  }

  void disable(policy::FailureStage) {
    require(!disabled_, "sticky graph fallback was recorded twice");
    disabled_ = true;
    ++counters_->fallbacks;
  }

  void finish_failure() {
    require(!capturing_, "direct fallback left capture active");
  }

 private:
  std::vector<int>* distance_ = nullptr;
  Counters* counters_ = nullptr;
  Failure failure_ = Failure::kNone;
  ConcurrentGate* gate_ = nullptr;
  Graph graph_ = 0;
  Executable executable_ = 0;
  int next_handle_ = 1;
  bool capturing_ = false;
  bool disabled_ = false;
  bool real_launch_attempted_ = false;
};

void apply_direct_fallback(policy::AttemptResult result,
                           std::vector<int>* distance) {
  if (result == policy::AttemptResult::kDirectFallback) {
    run_segment(distance);
  } else if (result == policy::AttemptResult::kRestartQueryDirect) {
    distance->assign(distance->size(), kInfinity);
    (*distance)[0] = 0;
    run_segment(distance);
  }
}

void test_success_and_cached_replay() {
  std::vector<int> state = {0, kInfinity, kInfinity, kInfinity, kInfinity};
  Counters counters;
  {
    FakeBackend backend(&state, &counters);
    require(policy::try_launch(&backend) ==
                policy::AttemptResult::kGraphLaunched,
            "successful first capture did not launch");
    require(state == std::vector<int>({0, 1, 2, kInfinity, kInfinity}),
            "successful graph replay produced the wrong segment result");
    require(policy::try_launch(&backend) ==
                policy::AttemptResult::kGraphLaunched,
            "cached graph did not replay");
    require(state == std::vector<int>({0, 1, 2, 3, 4}),
            "cached graph replay diverged from direct rounds");
    require(counters.begin == 1 && counters.enqueue == 1 &&
                counters.end == 1 && counters.instantiate == 1 &&
                counters.launch == 2 && counters.fallbacks == 0,
            "successful graph lifecycle accounting is wrong");
  }
  require(counters.graph_destroy == 1 &&
              counters.executable_destroy == 1,
          "workspace destruction did not release a successful cache once");
}

void test_recoverable_failures_and_sticky_reuse() {
  struct Case {
    Failure failure;
    int expected_end;
    int expected_graph_destroy;
    int expected_executable_destroy;
  };
  const std::vector<Case> cases = {
      {Failure::kBegin, 0, 0, 0},
      {Failure::kCapturedEnqueue, 1, 1, 0},
      {Failure::kCapturedEnqueueInvalidates, 1, 0, 0},
      {Failure::kEndNoGraph, 1, 0, 0},
      {Failure::kEndPartialGraph, 1, 1, 0},
      {Failure::kInstantiateNoExecutable, 1, 1, 0},
      {Failure::kInstantiatePartialExecutable, 1, 1, 1},
      {Failure::kLaunchPreSubmit, 1, 1, 1},
  };

  for (const Case& test : cases) {
    std::vector<int> state = {0, kInfinity, kInfinity, kInfinity, kInfinity};
    std::vector<int> control = state;
    run_segment(&control);
    run_segment(&control);
    Counters counters;
    {
      FakeBackend backend(&state, &counters, test.failure);
      policy::AttemptResult result = policy::try_launch(&backend);
      require(result == policy::AttemptResult::kDirectFallback,
              "recoverable graph failure did not request direct fallback");
      apply_direct_fallback(result, &state);
      result = policy::try_launch(&backend);
      require(result == policy::AttemptResult::kDirectFallback,
              "failed workspace retried graph construction");
      apply_direct_fallback(result, &state);
      require(state == control,
              "graph failure plus sticky reuse diverged from direct control");
      require(counters.begin == 1 && counters.end == test.expected_end &&
                  counters.fallbacks == 1,
              "graph failure was not sticky or capture cleanup was wrong");
    }
    require(counters.graph_destroy == test.expected_graph_destroy &&
                counters.executable_destroy ==
                    test.expected_executable_destroy,
            "partial graph construction was not destroyed exactly once");
  }
}

void test_non_atomic_launch_restarts_complete_query() {
  std::vector<int> state = {0, kInfinity, kInfinity, kInfinity, kInfinity};
  std::vector<int> control = state;
  run_segment(&control);
  Counters counters;
  {
    FakeBackend backend(&state, &counters,
                        Failure::kLaunchAfterPartialSubmit);
    const policy::AttemptResult result = policy::try_launch(&backend);
    require(result == policy::AttemptResult::kRestartQueryDirect,
            "partially submitted graph launch continued in-place");
    apply_direct_fallback(result, &state);
    require(state == control,
            "whole-query direct restart diverged from clean direct control");
    require(counters.quiesce == 1 && counters.fallbacks == 1 &&
                counters.graph_destroy == 1 &&
                counters.executable_destroy == 1,
            "partial launch did not quiesce and destroy its cache once");
    require(counters.cleanup_order.size() >= 3 &&
                counters.cleanup_order[0] == "quiesce" &&
                counters.cleanup_order[1] == "executable" &&
                counters.cleanup_order[2] == "graph",
            "graph cache was destroyed before partial launch quiescence");
  }
}

void test_cached_launch_failure_restarts_and_stays_direct() {
  std::vector<int> state = {0, kInfinity, kInfinity, kInfinity, kInfinity};
  std::vector<int> control = state;
  run_segment(&control);
  run_segment(&control);
  Counters counters;
  {
    FakeBackend backend(&state, &counters,
                        Failure::kLaunchAfterCachedReplay);
    require(policy::try_launch(&backend) ==
                policy::AttemptResult::kGraphLaunched,
            "first cached graph segment did not launch");
    policy::AttemptResult result = policy::try_launch(&backend);
    require(result == policy::AttemptResult::kRestartQueryDirect,
            "cached partial launch did not request a whole-query restart");
    apply_direct_fallback(result, &state);
    result = policy::try_launch(&backend);
    require(result == policy::AttemptResult::kDirectFallback,
            "cached launch failure retried graph construction");
    apply_direct_fallback(result, &state);
    require(state == control,
            "cached launch restart plus sticky continuation diverged from "
            "clean direct execution");
    require(counters.begin == 1 && counters.instantiate == 1 &&
                counters.launch == 2 && counters.quiesce == 1 &&
                counters.fallbacks == 1 && counters.graph_destroy == 1 &&
                counters.executable_destroy == 1,
            "cached launch failure lifecycle accounting is wrong");
  }
}

void test_concurrent_first_use_is_workspace_local() {
  constexpr int kWorkers = 3;
  ConcurrentGate gate(kWorkers);
  std::vector<std::vector<int>> states(
      kWorkers, std::vector<int>({0, kInfinity, kInfinity}));
  std::vector<Counters> counters(kWorkers);
  std::vector<std::thread> workers;
  std::vector<std::string> errors(kWorkers);
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      try {
        FakeBackend backend(&states[worker], &counters[worker],
                            Failure::kNone, &gate);
        require(policy::try_launch(&backend) ==
                    policy::AttemptResult::kGraphLaunched,
                "concurrent first capture failed");
      } catch (const std::exception& error) {
        errors[worker] = error.what();
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  for (int worker = 0; worker < kWorkers; ++worker) {
    require(errors[worker].empty(),
            "concurrent workspace failed: " + errors[worker]);
    require(states[worker] == std::vector<int>({0, 1, 2}) &&
                counters[worker].begin == 1 && counters[worker].end == 1 &&
                counters[worker].fallbacks == 0 &&
                counters[worker].graph_destroy == 1 &&
                counters[worker].executable_destroy == 1,
            "concurrent workspace state or ownership crossed workers");
  }
}

}  // namespace

int main() {
  try {
    test_success_and_cached_replay();
    test_recoverable_failures_and_sticky_reuse();
    test_non_atomic_launch_restarts_complete_query();
    test_cached_launch_failure_restarts_and_stays_direct();
    test_concurrent_first_use_is_workspace_local();
    std::cout << "BF11 graph execution policy tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF11 graph execution policy test failed: " << error.what()
              << '\n';
    return 1;
  }
}
