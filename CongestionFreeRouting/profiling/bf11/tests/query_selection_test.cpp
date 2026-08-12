#include "../query_selection.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_rejected(Callable callable, const char* message) {
  try {
    callable();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

int main() {
  const auto parsed = bf11_profile::parse_query_indices(
      "9, 2 # stratum A\n17\n", "fixture");
  require(parsed == std::vector<std::size_t>({9, 2, 17}),
          "query ordering changed");
  bf11_profile::validate_query_indices(parsed, 18);
  require_rejected(
      [] { (void)bf11_profile::parse_query_indices("1,1", "duplicate"); },
      "duplicate query was accepted");
  require_rejected(
      [] { (void)bf11_profile::parse_query_indices("-1", "negative"); },
      "negative query was accepted");
  require_rejected(
      [&] { bf11_profile::validate_query_indices(parsed, 17); },
      "out-of-range query was accepted");
  std::cout << "BF11 query-selection test passed\n";
}
