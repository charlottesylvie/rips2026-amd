#pragma once

#include <cstddef>

// Compile PathFinder with -DPATHFINDER_ENABLE_ROCTX and link against
// rocprofiler-sdk-roctx to emit these ranges. Normal builds remain completely
// independent of the profiling SDK and optimize the scopes away.
#if defined(PATHFINDER_ENABLE_ROCTX)
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace pathfinder_profile {

struct QueryIdentity {
  bool valid = false;
  std::size_t net_index = 0;
  std::size_t worker_index = 0;
};

#if defined(PATHFINDER_ENABLE_ROCTX) || \
    defined(PATHFINDER_ENABLE_BF11_DIAGNOSTICS)
inline thread_local QueryIdentity current_query_identity{};
#endif

class ScopedQueryIdentity {
 public:
  ScopedQueryIdentity(std::size_t net_index, std::size_t worker_index) noexcept {
#if defined(PATHFINDER_ENABLE_ROCTX) || \
    defined(PATHFINDER_ENABLE_BF11_DIAGNOSTICS)
    previous_ = current_query_identity;
    current_query_identity = {true, net_index, worker_index};
#else
    (void)net_index;
    (void)worker_index;
#endif
  }
  ~ScopedQueryIdentity() noexcept {
#if defined(PATHFINDER_ENABLE_ROCTX) || \
    defined(PATHFINDER_ENABLE_BF11_DIAGNOSTICS)
    current_query_identity = previous_;
#endif
  }
  ScopedQueryIdentity(const ScopedQueryIdentity&) = delete;
  ScopedQueryIdentity& operator=(const ScopedQueryIdentity&) = delete;

 private:
#if defined(PATHFINDER_ENABLE_ROCTX) || \
    defined(PATHFINDER_ENABLE_BF11_DIAGNOSTICS)
  QueryIdentity previous_{};
#endif
};

inline QueryIdentity query_identity() noexcept {
#if defined(PATHFINDER_ENABLE_ROCTX) || \
    defined(PATHFINDER_ENABLE_BF11_DIAGNOSTICS)
  return current_query_identity;
#else
  return {};
#endif
}

class ScopedRange {
 public:
  explicit ScopedRange(const char* name) noexcept {
#if defined(PATHFINDER_ENABLE_ROCTX)
    roctxRangePush(name);
#else
    (void)name;
#endif
  }

  ~ScopedRange() noexcept {
#if defined(PATHFINDER_ENABLE_ROCTX)
    roctxRangePop();
#endif
  }

  ScopedRange(const ScopedRange&) = delete;
  ScopedRange& operator=(const ScopedRange&) = delete;
};

}  // namespace pathfinder_profile

#define PATHFINDER_PROFILE_CONCAT_IMPL(left, right) left##right
#define PATHFINDER_PROFILE_CONCAT(left, right) \
  PATHFINDER_PROFILE_CONCAT_IMPL(left, right)
#define PATHFINDER_PROFILE_RANGE(name)                                  \
  ::pathfinder_profile::ScopedRange PATHFINDER_PROFILE_CONCAT(          \
      pathfinder_profile_range_, __LINE__)(name)
#define PATHFINDER_PROFILE_QUERY_IDENTITY(net_index, worker_index)       \
  ::pathfinder_profile::ScopedQueryIdentity PATHFINDER_PROFILE_CONCAT(  \
      pathfinder_profile_query_, __LINE__)(net_index, worker_index)
