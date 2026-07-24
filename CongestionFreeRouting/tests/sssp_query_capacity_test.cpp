#include "../sssp_query_capacity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Operation>
void require_overflow(Operation operation, const std::string& message) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, message);
}

constexpr SsspQueryCapacityHints constexpr_hints() {
  SsspQueryCapacityHints hints;
  sssp_capacity::accumulate_query_counts(hints, 2, 3);
  sssp_capacity::accumulate_query_counts(hints, 1, 5);
  return hints;
}

static_assert(sssp_capacity::checked_add(2, 3) == 5,
              "checked addition must be constexpr");
static_assert(sssp_capacity::checked_multiply(3, 4) == 12,
              "checked multiplication must be constexpr");
static_assert(sssp_capacity::checked_bytes<std::uint64_t>(2) == 16,
              "checked byte calculation must be constexpr");
static_assert(constexpr_hints().max_sources == 2 &&
                  constexpr_hints().max_targets == 5,
              "raw query accumulation must be constexpr");

}  // namespace

int main() {
  try {
    SsspQueryCapacityHints empty;
    sssp_capacity::validate_reservation(empty);
    require(empty.max_sources == 0 && empty.max_targets == 0,
            "default capacity hints are not empty");
    sssp_capacity::accumulate_query_counts(empty, 0, 0);
    require(empty.max_sources == 0 && empty.max_targets == 0,
            "an empty request changed empty hints");

    // These are raw metadata sizes. The repeated logical endpoints represented
    // by those sizes must not be removed by the capacity policy.
    SsspQueryCapacityHints duplicate_sized;
    sssp_capacity::accumulate_query_counts(duplicate_sized, 4, 6);
    require(duplicate_sized.max_sources == 4 &&
                duplicate_sized.max_targets == 6,
            "raw duplicate-sized counts were de-duplicated");

    constexpr std::array<std::array<std::size_t, 2>, 3> requests{{
        {{2, 1}},
        {{5, 3}},
        {{1, 7}},
    }};
    SsspQueryCapacityHints prefix;
    for (std::size_t index = 0; index < 2; ++index) {
      sssp_capacity::accumulate_query_counts(
          prefix, requests[index][0], requests[index][1]);
    }
    require(prefix.max_sources == 5 && prefix.max_targets == 3,
            "two-request prefix produced incorrect maxima");
    sssp_capacity::accumulate_query_counts(
        prefix, requests[2][0], requests[2][1]);
    require(prefix.max_sources == 5 && prefix.max_targets == 7,
            "extended request prefix produced incorrect maxima");

    const std::size_t int_max = static_cast<std::size_t>(
        std::numeric_limits<int>::max());
    require(sssp_capacity::checked_device_count(int_max) == int_max,
            "INT_MAX device query count was rejected");
    require(sssp_capacity::checked_target_offset_count(int_max) ==
                int_max + 1,
            "INT_MAX target sentinel count was calculated incorrectly");
    SsspQueryCapacityHints boundary;
    sssp_capacity::accumulate_query_counts(boundary, int_max, int_max);
    require(boundary.max_sources == int_max &&
                boundary.max_targets == int_max,
            "INT_MAX raw counts were not retained");

    const std::size_t beyond_int_max = int_max + 1;
    require_overflow(
        [=] { (void)sssp_capacity::checked_device_count(beyond_int_max); },
        "INT_MAX+1 device count was accepted");
    require_overflow(
        [=] {
          SsspQueryCapacityHints hints;
          sssp_capacity::accumulate_query_counts(
              hints, beyond_int_max, 0);
        },
        "INT_MAX+1 source count was accepted");
    require_overflow(
        [=] {
          SsspQueryCapacityHints hints;
          sssp_capacity::accumulate_query_counts(
              hints, 0, beyond_int_max);
        },
        "INT_MAX+1 target count was accepted");

    SsspQueryCapacityHints unchanged{3, 4};
    require_overflow(
        [&] {
          sssp_capacity::accumulate_query_counts(
              unchanged, beyond_int_max, 1);
        },
        "overflowing accumulation was accepted");
    require(unchanged.max_sources == 3 && unchanged.max_targets == 4,
            "failed accumulation changed the prior high-water hints");

    const std::size_t size_max =
        std::numeric_limits<std::size_t>::max();
    require_overflow(
        [=] { (void)sssp_capacity::checked_add(size_max, 1); },
        "SIZE_MAX addition overflow was accepted");
    require_overflow(
        [=] { (void)sssp_capacity::checked_multiply(size_max, 2); },
        "SIZE_MAX multiplication overflow was accepted");
    require_overflow(
        [=] {
          (void)sssp_capacity::checked_bytes<std::uint64_t>(size_max);
        },
        "SIZE_MAX byte overflow was accepted");

    std::size_t capacity = 0;
    for (const std::size_t required :
         std::array<std::size_t, 7>{{0, 1, 2, 3, 9, 10, 100}}) {
      const std::size_t grown =
          sssp_capacity::geometric_capacity(capacity, required);
      require(grown >= capacity,
              "geometric capacity moved below its high-water mark");
      require(grown >= required,
              "geometric capacity did not satisfy the request");
      capacity = grown;
    }
    require(sssp_capacity::geometric_capacity(capacity, 2) == capacity,
            "a smaller request caused capacity shrinkage");
    require(sssp_capacity::geometric_capacity(capacity, capacity) ==
                capacity,
            "an equal request changed retained capacity");
    require_overflow(
        [=] {
          (void)sssp_capacity::geometric_capacity(size_max - 1, size_max);
        },
        "overflowing geometric growth was accepted");

    std::cout << "SSSP query capacity policy test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SSSP query capacity policy test failed: " << error.what()
              << '\n';
    return 1;
  }
}
