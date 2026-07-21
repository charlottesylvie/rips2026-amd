#include "../interchange/gzip_io.hpp"

#include <zlib.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::uint8_t> read_checked(
    const std::filesystem::path& path) {
  std::vector<std::uint8_t> bytes;
  routing::interchange::read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t size) {
        bytes.insert(bytes.end(), data, data + size);
      });
  return bytes;
}

void write_plain(const std::filesystem::path& path,
                 const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  out.close();
  require(static_cast<bool>(out), "failed to write plain test input");
}

void write_gzip(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
  gzFile out = gzopen(path.string().c_str(), "wb6");
  require(out != nullptr, "failed to create gzip test input");
  const int written =
      gzwrite(out, bytes.data(), static_cast<unsigned int>(bytes.size()));
  require(written == static_cast<int>(bytes.size()),
          "failed to write gzip test input");
  require(gzclose(out) == Z_OK, "failed to close gzip test input");
}

std::vector<std::uint8_t> read_raw(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>());
}

void expect_truncated(const std::filesystem::path& path) {
  try {
    (void)read_checked(path);
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    require(message.find(path.string()) != std::string::npos,
            "truncated-input error omitted its path");
    require(message.find("after decoding") != std::string::npos,
            "truncated-input error omitted its decoded byte count");
    require(message.find("ended prematurely") != std::string::npos,
            "truncated-input error omitted the gzip cause");
    return;
  }
  throw std::runtime_error("truncated gzip input was accepted");
}

}  // namespace

int main() {
  std::vector<std::filesystem::path> cleanup;
  try {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() /
        ("rips-gzip-io-test-" + std::to_string(nonce));
    const std::filesystem::path plain_path = base.string() + ".plain";
    const std::filesystem::path gzip_path = base.string() + ".gz";
    const std::filesystem::path no_trailer_path =
        base.string() + ".no-trailer.gz";
    const std::filesystem::path cut_body_path =
        base.string() + ".cut-body.gz";
    cleanup = {plain_path, gzip_path, no_trailer_path, cut_body_path};

    std::vector<std::uint8_t> expected(3U << 20);
    for (std::size_t index = 0; index < expected.size(); ++index) {
      expected[index] = static_cast<std::uint8_t>((index * 37U) & 0xffU);
    }

    write_plain(plain_path, expected);
    write_gzip(gzip_path, expected);
    require(read_checked(plain_path) == expected,
            "plain input changed while reading");
    require(read_checked(gzip_path) == expected,
            "gzip input changed while reading");

    const std::vector<std::uint8_t> encoded = read_raw(gzip_path);
    require(encoded.size() > 32, "gzip fixture is unexpectedly small");
    write_plain(no_trailer_path,
                std::vector<std::uint8_t>(encoded.begin(), encoded.end() - 8));
    write_plain(cut_body_path,
                std::vector<std::uint8_t>(encoded.begin(),
                                          encoded.begin() + encoded.size() / 2));
    expect_truncated(no_trailer_path);
    expect_truncated(cut_body_path);

    for (const std::filesystem::path& path : cleanup) {
      std::filesystem::remove(path);
    }
    std::cout << "gzip/plain integrity test passed\n";
    return 0;
  } catch (const std::exception& error) {
    for (const std::filesystem::path& path : cleanup) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
    std::cerr << "gzip/plain integrity test failed: " << error.what() << '\n';
    return 1;
  }
}
