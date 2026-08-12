// Same-device streaming-memory reference for BF11 counter normalization.
// The reference reports unique application payload; profiler counters report
// request/cache/fabric traffic and must establish their own units separately.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
  }
}

__global__ void streaming_copy(const float* __restrict__ input,
                               float* __restrict__ output,
                               std::size_t count,
                               int repetitions) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
      output[index] = input[index] + static_cast<float>(repetition & 1);
    }
  }
}

__global__ void streaming_warmup(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 std::size_t count) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    output[index] = input[index];
  }
}

std::size_t parse_size(const char* text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t bytes = 512ULL * 1024ULL * 1024ULL;
    int repetitions = 32;
    for (int arg = 1; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (arg + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value");
      }
      if (option == "--bytes") {
        bytes = parse_size(argv[++arg], "bytes");
      } else if (option == "--repetitions") {
        const std::size_t parsed = parse_size(argv[++arg], "repetitions");
        if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          throw std::invalid_argument("repetitions exceeds int");
        }
        repetitions = static_cast<int>(parsed);
      } else {
        throw std::invalid_argument("unknown option: " + option);
      }
    }
    bytes -= bytes % sizeof(float);
    const std::size_t count = bytes / sizeof(float);
    if (count == 0) throw std::invalid_argument("byte count is too small");

    std::vector<float> host(count, 1.0f);
    float* input = nullptr;
    float* output = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&input), bytes),
          "allocate streaming input");
    try {
      check(hipMalloc(reinterpret_cast<void**>(&output), bytes),
            "allocate streaming output");
      check(hipMemcpy(input, host.data(), bytes, hipMemcpyHostToDevice),
            "initialize streaming input");
      check(hipMemset(output, 0, bytes), "initialize streaming output");
      int device = 0;
      check(hipGetDevice(&device), "get streaming device");
      hipDeviceProp_t properties{};
      check(hipGetDeviceProperties(&properties, device),
            "get streaming device properties");
      const unsigned blocks = static_cast<unsigned>(
          std::max(1, properties.multiProcessorCount * 8));
      constexpr unsigned threads = 256;

      hipLaunchKernelGGL(streaming_warmup, dim3(blocks), dim3(threads), 0, 0,
                         input, output, count);
      check(hipGetLastError(), "launch streaming warmup");
      check(hipDeviceSynchronize(), "synchronize streaming warmup");

      hipEvent_t begin = nullptr;
      hipEvent_t end = nullptr;
      check(hipEventCreate(&begin), "create streaming begin event");
      check(hipEventCreate(&end), "create streaming end event");
      check(hipEventRecord(begin), "record streaming begin event");
      hipLaunchKernelGGL(streaming_copy, dim3(blocks), dim3(threads), 0, 0,
                         input, output, count, repetitions);
      check(hipGetLastError(), "launch streaming reference");
      check(hipEventRecord(end), "record streaming end event");
      check(hipEventSynchronize(end), "synchronize streaming end event");
      float milliseconds = 0.0f;
      check(hipEventElapsedTime(&milliseconds, begin, end),
            "measure streaming reference");
      (void)hipEventDestroy(begin);
      (void)hipEventDestroy(end);

      float sample = 0.0f;
      check(hipMemcpy(&sample, output, sizeof(sample), hipMemcpyDeviceToHost),
            "read streaming checksum");
      const std::uint64_t unique_read_bytes =
          static_cast<std::uint64_t>(bytes) * repetitions;
      const std::uint64_t unique_write_bytes = unique_read_bytes;
      const double seconds = static_cast<double>(milliseconds) * 1.0e-3;
      const double payload_gib_per_second =
          seconds > 0.0
              ? static_cast<double>(unique_read_bytes + unique_write_bytes) /
                    seconds / static_cast<double>(1ULL << 30)
              : 0.0;
      std::cout << "{\"type\":\"bf11_streaming_reference\""
                << ",\"schema_version\":1"
                << ",\"device\":" << device
                << ",\"architecture\":\"" << properties.gcnArchName << "\""
                << ",\"array_bytes\":" << bytes
                << ",\"repetitions\":" << repetitions
                << ",\"unique_read_bytes\":" << unique_read_bytes
                << ",\"unique_write_bytes\":" << unique_write_bytes
                << ",\"seconds\":" << seconds
                << ",\"unique_payload_gib_per_second\":"
                << payload_gib_per_second
                << ",\"checksum\":" << sample << "}\n";
      (void)hipFree(output);
      (void)hipFree(input);
      return 0;
    } catch (...) {
      if (output) (void)hipFree(output);
      if (input) (void)hipFree(input);
      throw;
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
