#pragma once

#include <cstdint>

namespace cooperative_groups {

// Compile-only placeholder. It intentionally supplies no synchronization or
// residency semantics; those require a real HIP compiler and AMD GPU tests.
class grid_group {
 public:
  std::uint64_t thread_rank() const { return 0; }
  std::uint64_t size() const { return 1; }
  void sync() const {}
};

inline grid_group this_grid() {
  return {};
}

}  // namespace cooperative_groups
