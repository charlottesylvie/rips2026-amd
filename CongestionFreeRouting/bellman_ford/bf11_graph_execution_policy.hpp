#pragma once

// Host-testable ownership and fallback policy for one cached BF11 HIP Graph
// segment. The backend supplies the runtime operations; this policy guarantees
// that every successful begin-capture is paired with exactly one end-capture,
// partial graph objects are released before direct fallback, and a failed
// workspace becomes sticky-direct.
namespace bf11_graph_execution_policy {

enum class AttemptResult {
  kGraphLaunched,
  kDirectFallback,
  // A real runtime launch error may occur after a prefix of graph nodes was
  // submitted. The backend has quiesced that work; the caller must reset and
  // seed the complete query again before direct execution.
  kRestartQueryDirect,
};

enum class FailureStage {
  kBeginCapture,
  kCapturedEnqueue,
  kEndCapture,
  kInstantiate,
  kLaunch,
};

// Backend requirements:
//
//   using Graph; using Executable;
//   bool disabled() const;
//   bool has_cached_executable() const;
//   Graph null_graph() const; Executable null_executable() const;
//   bool valid_graph(Graph) const; bool valid_executable(Executable) const;
//   bool begin_capture();
//   bool enqueue_captured_segment();
//   bool end_capture(Graph*);
//   bool instantiate(Executable*, Graph);
//   void adopt(Graph, Executable);
//   bool launch_cached();
//   bool launch_failure_requires_restart() const;
//   void prepare_launch_failure();
//   void destroy_graph(Graph);
//   void destroy_executable(Executable);
//   void invalidate_cached();
//   void disable(FailureStage);
//   void finish_failure();
//
// Runtime-facing backends should make the bool-returning operations
// non-throwing for recoverable Graph errors. finish_failure() runs only after
// capture has ended and partial resources have been released, so it can safely
// surface an unrelated HIP error rather than masking it as a Graph fallback.
template <typename Backend>
AttemptResult try_launch(Backend* backend) {
  if (backend->disabled()) return AttemptResult::kDirectFallback;

  if (!backend->has_cached_executable()) {
    typename Backend::Graph graph = backend->null_graph();
    if (!backend->begin_capture()) {
      backend->disable(FailureStage::kBeginCapture);
      backend->finish_failure();
      return AttemptResult::kDirectFallback;
    }

    const bool enqueued = backend->enqueue_captured_segment();
    // End capture even after an invalidating enqueue. HIP specifies that this
    // is the operation that returns an invalidated capture to a non-capturing
    // stream state.
    const bool ended = backend->end_capture(&graph);
    if (!enqueued || !ended || !backend->valid_graph(graph)) {
      backend->destroy_graph(graph);
      backend->disable(!enqueued ? FailureStage::kCapturedEnqueue
                                 : FailureStage::kEndCapture);
      backend->finish_failure();
      return AttemptResult::kDirectFallback;
    }

    typename Backend::Executable executable = backend->null_executable();
    if (!backend->instantiate(&executable, graph) ||
        !backend->valid_executable(executable)) {
      backend->destroy_executable(executable);
      backend->destroy_graph(graph);
      backend->disable(FailureStage::kInstantiate);
      backend->finish_failure();
      return AttemptResult::kDirectFallback;
    }
    backend->adopt(graph, executable);
  }

  if (backend->launch_cached()) return AttemptResult::kGraphLaunched;

  const bool restart = backend->launch_failure_requires_restart();
  // A real Graph launch is not guaranteed to be submission-atomic. Quiesce
  // any prefix before destroying its executable and before the caller resets
  // query state. A pre-submit injected failure is known not to need this.
  backend->prepare_launch_failure();
  backend->invalidate_cached();
  backend->disable(FailureStage::kLaunch);
  backend->finish_failure();
  return restart ? AttemptResult::kRestartQueryDirect
                 : AttemptResult::kDirectFallback;
}

}  // namespace bf11_graph_execution_policy
