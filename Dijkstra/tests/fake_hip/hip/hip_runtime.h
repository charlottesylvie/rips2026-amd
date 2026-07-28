#pragma once

// Host-only HIP surface used by preds_GPU_standalone_host_test.sh.
//
// This is deliberately a compile/link shim, not a GPU emulator.  The
// regression only executes CLI paths that return before device discovery.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define __device__
#define __global__
#define __host__
#define __shared__
#define __forceinline__ inline

struct dim3 {
  constexpr dim3(unsigned int x_value = 1,
                 unsigned int y_value = 1,
                 unsigned int z_value = 1)
      : x(x_value), y(y_value), z(z_value) {}

  unsigned int x;
  unsigned int y;
  unsigned int z;
};

inline constexpr dim3 threadIdx{};
inline constexpr dim3 blockIdx{};
inline constexpr dim3 blockDim{};
inline constexpr dim3 gridDim{};

// HIP's `extern __shared__` declarations normally resolve to launch-provided
// storage.  These one-element placeholders are never executed, but let a
// regular host linker resolve kernel specializations used in occupancy calls.
namespace rips_predicates_gpu {
namespace {
inline unsigned int shared_values[1] = {};
inline unsigned char shared_raw[1] = {};
inline unsigned long long success_counts[1] = {};
}  // namespace
}  // namespace rips_predicates_gpu

inline void __syncthreads() {}
inline void __threadfence() {}

inline unsigned int __float_as_uint(float value) {
  unsigned int bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float __uint_as_float(unsigned int bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T, typename Compare, typename Value>
inline T atomicCAS(T* address, Compare compare, Value value) {
  const T prior = *address;
  if (prior == static_cast<T>(compare)) {
    *address = static_cast<T>(value);
  }
  return prior;
}

template <typename T>
inline T atomicAdd(T* address, T value) {
  const T prior = *address;
  *address += value;
  return prior;
}

template <typename T>
inline T atomicMin(T* address, T value) {
  const T prior = *address;
  if (value < prior) {
    *address = value;
  }
  return prior;
}

template <typename T>
inline T atomicExch(T* address, T value) {
  const T prior = *address;
  *address = value;
  return prior;
}

using hipError_t = int;
using hipStream_t = void*;
using hipEvent_t = void*;

inline constexpr hipError_t hipSuccess = 0;
inline constexpr unsigned int hipEventDefault = 0;
inline constexpr unsigned int hipEventDisableTiming = 1;
inline constexpr unsigned int hipStreamNonBlocking = 1;
inline constexpr unsigned int hipHostMallocDefault = 0;
inline constexpr int hipMemcpyHostToDevice = 0;
inline constexpr int hipMemcpyDeviceToHost = 1;
inline constexpr int hipDeviceAttributeIntegrated = 0;

struct hipDeviceProp_t {
  char name[256] = "host-only HIP test shim";
  char gcnArchName[256] = "host-only";
  int warpSize = 32;
  int multiProcessorCount = 1;
  int maxThreadsPerBlock = 256;
};

inline const char* hipGetErrorString(hipError_t) {
  return "host-only HIP test shim";
}

inline hipError_t hipGetDevice(int* device) {
  *device = 0;
  return hipSuccess;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties, int) {
  *properties = hipDeviceProp_t{};
  return hipSuccess;
}

inline hipError_t hipDeviceGetAttribute(int* value, int, int) {
  *value = 0;
  return hipSuccess;
}

inline hipError_t hipMemGetInfo(std::size_t* free_bytes,
                                std::size_t* total_bytes) {
  *free_bytes = static_cast<std::size_t>(1) << 30;
  *total_bytes = static_cast<std::size_t>(1) << 30;
  return hipSuccess;
}

inline hipError_t hipMalloc(void** pointer, std::size_t bytes) {
  *pointer = std::malloc(bytes == 0 ? 1 : bytes);
  return *pointer == nullptr ? 1 : hipSuccess;
}

inline hipError_t hipFree(void* pointer) {
  std::free(pointer);
  return hipSuccess;
}

inline hipError_t hipHostMalloc(void** pointer,
                                std::size_t bytes,
                                unsigned int) {
  return hipMalloc(pointer, bytes);
}

inline hipError_t hipHostFree(void* pointer) {
  return hipFree(pointer);
}

inline hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int) {
  *event = reinterpret_cast<void*>(1);
  return hipSuccess;
}

inline hipError_t hipEventDestroy(hipEvent_t) {
  return hipSuccess;
}

inline hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
  return hipSuccess;
}

inline hipError_t hipEventSynchronize(hipEvent_t) {
  return hipSuccess;
}

inline hipError_t hipEventElapsedTime(float* elapsed_ms,
                                      hipEvent_t,
                                      hipEvent_t) {
  *elapsed_ms = 0.0f;
  return hipSuccess;
}

inline hipError_t hipStreamCreateWithFlags(hipStream_t* stream,
                                           unsigned int) {
  *stream = reinterpret_cast<void*>(1);
  return hipSuccess;
}

inline hipError_t hipStreamSynchronize(hipStream_t) {
  return hipSuccess;
}

inline hipError_t hipStreamDestroy(hipStream_t) {
  return hipSuccess;
}

inline hipError_t hipMemcpyAsync(void* destination,
                                 const void* source,
                                 std::size_t bytes,
                                 int,
                                 hipStream_t) {
  std::memcpy(destination, source, bytes);
  return hipSuccess;
}

inline hipError_t hipGetLastError() {
  return hipSuccess;
}

// Do not instantiate or execute device kernels in a host-only CLI test.
#define hipLaunchKernelGGL(...) ((void)0)

// Similarly, avoid ODR-using kernel specializations solely for occupancy
// queries.  The guarded-link smoke test exercises the public host API only.
#define hipOccupancyMaxActiveBlocksPerMultiprocessor( \
    active_blocks, kernel, block_size, shared_bytes)  \
  (*(active_blocks) = 1, hipSuccess)
