#pragma once

#include <zlib.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace routing::interchange {
namespace detail {

inline std::runtime_error make_gzip_read_error(
    const std::filesystem::path& path,
    int status,
    const std::string& zlib_message,
    std::uint64_t decoded_bytes) {
  std::string detail;
  if (status == Z_BUF_ERROR) {
    // zlib deliberately reports a gzip stream that ends before its trailer or
    // final deflate block as Z_BUF_ERROR, not as a negative gzread() result.
    detail = "gzip stream ended prematurely";
  } else if (!zlib_message.empty()) {
    detail = zlib_message;
  } else if (const char* message = zError(status); message != nullptr) {
    detail = message;
  } else {
    detail = "zlib error " + std::to_string(status);
  }

  return std::runtime_error(
      "failed while reading " + path.string() + " after decoding " +
      std::to_string(decoded_bytes) + " bytes: " + detail);
}

}  // namespace detail

// Stream a gzip-compressed or plain file through consume(data, byte_count).
// gzopen() transparently handles plain input. A zero return from gzread() is
// not sufficient to prove that a gzip stream completed: truncated streams
// return all available output and defer Z_BUF_ERROR to gzerror()/gzclose().
template <typename Consume>
std::uint64_t read_gzip_or_plain_chunks(
    const std::filesystem::path& path,
    Consume&& consume) {
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    throw std::runtime_error("could not open input file: " + path.string());
  }

  constexpr unsigned int kBufferBytes = 1U << 20;
  if (gzbuffer(file, kBufferBytes) != 0) {
    (void)gzclose(file);
    throw std::runtime_error("could not allocate gzip input buffer for: " +
                             path.string());
  }

  std::array<std::uint8_t, kBufferBytes> buffer{};
  std::uint64_t decoded_bytes = 0;
  int terminal_status = Z_OK;
  std::string terminal_message;

  try {
    while (true) {
      const int read_count =
          gzread(file, buffer.data(), static_cast<unsigned int>(buffer.size()));
      if (read_count > 0) {
        const std::uint64_t chunk_size =
            static_cast<std::uint64_t>(read_count);
        if (chunk_size >
            std::numeric_limits<std::uint64_t>::max() - decoded_bytes) {
          throw std::runtime_error("decoded input is too large: " +
                                   path.string());
        }
        consume(buffer.data(), static_cast<std::size_t>(read_count));
        decoded_bytes += chunk_size;
        continue;
      }

      int zlib_status = Z_OK;
      const char* raw_message = gzerror(file, &zlib_status);
      terminal_status = zlib_status;
      if (raw_message != nullptr) {
        // gzerror() returns storage owned by gzFile, so preserve it before
        // gzclose() invalidates the pointer.
        terminal_message = raw_message;
      }
      if (read_count < 0 && terminal_status == Z_OK) {
        terminal_status = Z_ERRNO;
      }
      break;
    }

    const int close_status = gzclose(file);
    file = nullptr;
    if (terminal_status != Z_OK) {
      throw detail::make_gzip_read_error(
          path, terminal_status, terminal_message, decoded_bytes);
    }
    if (close_status != Z_OK) {
      throw detail::make_gzip_read_error(
          path, close_status, std::string(), decoded_bytes);
    }
  } catch (...) {
    if (file != nullptr) {
      (void)gzclose(file);
    }
    throw;
  }

  return decoded_bytes;
}

}  // namespace routing::interchange
