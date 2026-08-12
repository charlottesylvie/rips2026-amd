#pragma once

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace bf11_profile {

// Parse a deterministic query manifest. Commas and ASCII whitespace are
// separators; comments start with '#' and continue to end of line. Ordering is
// retained because it is part of profiler replay provenance.
inline std::vector<std::size_t> parse_query_indices(
    const std::string& text,
    const char* source = "query selection") {
  std::vector<std::size_t> indices;
  std::unordered_set<std::size_t> seen;
  std::string token;
  bool comment = false;
  auto finish_token = [&]() {
    if (token.empty()) return;
    if (token.front() == '-') {
      throw std::invalid_argument(std::string(source) +
                                  " contains a negative index: " + token);
    }
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
      parsed = std::stoull(token, &consumed, 10);
    } catch (const std::exception&) {
      throw std::invalid_argument(std::string(source) +
                                  " contains an invalid index: " + token);
    }
    if (consumed != token.size() ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument(std::string(source) +
                                  " contains an invalid index: " + token);
    }
    const std::size_t index = static_cast<std::size_t>(parsed);
    if (!seen.insert(index).second) {
      throw std::invalid_argument(std::string(source) +
                                  " contains duplicate index " + token);
    }
    indices.push_back(index);
    token.clear();
  };

  for (const char ch : text) {
    if (comment) {
      if (ch == '\n') comment = false;
      continue;
    }
    if (ch == '#') {
      finish_token();
      comment = true;
    } else if (ch == ',' || std::isspace(static_cast<unsigned char>(ch))) {
      finish_token();
    } else {
      token.push_back(ch);
    }
  }
  finish_token();
  if (indices.empty()) {
    throw std::invalid_argument(std::string(source) + " is empty");
  }
  return indices;
}

inline std::vector<std::size_t> load_query_manifest(
    const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open query manifest: " + path);
  }
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  return parse_query_indices(text, path.c_str());
}

inline void validate_query_indices(const std::vector<std::size_t>& indices,
                                   std::size_t request_count) {
  if (indices.empty()) {
    throw std::invalid_argument("explicit query selection is empty");
  }
  std::unordered_set<std::size_t> seen;
  for (const std::size_t index : indices) {
    if (index >= request_count) {
      throw std::out_of_range("query index " + std::to_string(index) +
                              " is outside " +
                              std::to_string(request_count) +
                              " route requests");
    }
    if (!seen.insert(index).second) {
      throw std::invalid_argument("query selection contains duplicate index " +
                                  std::to_string(index));
    }
  }
}

}  // namespace bf11_profile
