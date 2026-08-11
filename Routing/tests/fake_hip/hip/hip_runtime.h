#pragma once

// A compile-only subset of HIP used by the host-side regression tests. This
// header deliberately does not emulate kernel scheduling, memory visibility,
// atomics, streams, or graphs; it only lets a normal C++ compiler type-check
// HIP translation units on machines without ROCm.

#include <cstddef>
#include <cstdint>
#include <cstring>

using hipStream_t = void*;
using hipEvent_t = void*;
using hipError_t = int;

struct dim3 {
  constexpr dim3(unsigned int x_value = 1,
                 unsigned int y_value = 1,
                 unsigned int z_value = 1)
      : x(x_value), y(y_value), z(z_value) {}

  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct hipDeviceProp_t {
  int warpSize = 64;
  int multiProcessorCount = 1;
};

constexpr hipError_t hipSuccess = 0;
constexpr hipError_t hipErrorStreamCaptureUnsupported = 900;
constexpr hipError_t hipErrorStreamCaptureInvalidated = 901;
constexpr hipError_t hipErrorStreamCaptureWrongThread = 908;
constexpr unsigned int hipStreamNonBlocking = 1;
constexpr unsigned int hipHostMallocDefault = 0;

enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
};

enum hipDeviceAttribute_t {
  hipDeviceAttributeCooperativeLaunch = 0,
  hipDeviceAttributeWallClockRate = 1,
};

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
// Static storage best approximates zero-initialized block-shared declarations
// for host-only diagnostics. This remains compile-only and models no barrier.
#define __shared__ static
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

// Device built-ins are only placeholders for syntax checking.
inline dim3 threadIdx{};
inline dim3 blockIdx{};
inline dim3 blockDim{};
inline dim3 gridDim{};

inline void __syncthreads() {}

template <typename T>
inline T min(T left, T right) {
  return right < left ? right : left;
}

inline unsigned int __float_as_uint(float value) {
  unsigned int bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "fake HIP float size mismatch");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float __uint_as_float(unsigned int bits) {
  float value = 0.0f;
  static_assert(sizeof(bits) == sizeof(value), "fake HIP float size mismatch");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline unsigned long long wall_clock64() {
  return 0;
}

#ifndef __HIP_MEMORY_SCOPE_AGENT
#define __HIP_MEMORY_SCOPE_AGENT 4
#endif

template <typename T>
inline T atomicCAS(T* address, T compare, T value) {
  const T observed = *address;
  if (observed == compare) *address = value;
  return observed;
}

template <typename T, typename U>
inline T atomicAdd(T* address, U value) {
  const T observed = *address;
  *address = static_cast<T>(observed + static_cast<T>(value));
  return observed;
}

template <typename T, typename U>
inline T atomicExch(T* address, U value) {
  const T observed = *address;
  *address = static_cast<T>(value);
  return observed;
}

template <typename T>
inline T atomicMin(T* address, T value) {
  const T observed = *address;
  if (value < observed) *address = value;
  return observed;
}

template <typename T>
inline T atomicMax(T* address, T value) {
  const T observed = *address;
  if (value > observed) *address = value;
  return observed;
}

#define hipLaunchKernelGGL(kernel, grid, block, shared, stream, ...) \
  ((void)(grid), (void)(block), (void)(shared), (void)(stream),       \
   kernel(__VA_ARGS__))

inline const char* hipGetErrorString(hipError_t) {
  return "fake HIP error";
}

inline hipError_t hipGetLastError() {
  return hipSuccess;
}

inline hipError_t hipPeekAtLastError() {
  return hipSuccess;
}

inline hipError_t hipGetDevice(int* device) {
  *device = 0;
  return hipSuccess;
}

inline hipError_t hipSetDevice(int) {
  return hipSuccess;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties, int) {
  properties->warpSize = 64;
  properties->multiProcessorCount = 1;
  return hipSuccess;
}

inline hipError_t hipDeviceGetAttribute(int* value,
                                        hipDeviceAttribute_t attribute,
                                        int) {
  *value = attribute == hipDeviceAttributeCooperativeLaunch ? 1 : 1000000;
  return hipSuccess;
}

inline hipError_t hipMemGetInfo(std::size_t* free_bytes,
                                std::size_t* total_bytes) {
  *free_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  *total_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
  return hipSuccess;
}

inline hipError_t hipMalloc(void**, std::size_t) {
  return hipSuccess;
}

inline hipError_t hipFree(void*) {
  return hipSuccess;
}

inline hipError_t hipHostMalloc(void**, std::size_t, unsigned int) {
  return hipSuccess;
}

inline hipError_t hipHostFree(void*) {
  return hipSuccess;
}

inline hipError_t hipMemcpyAsync(void*, const void*, std::size_t,
                                 hipMemcpyKind, hipStream_t = nullptr) {
  return hipSuccess;
}

inline hipError_t hipMemsetAsync(void*, int, std::size_t,
                                 hipStream_t = nullptr) {
  return hipSuccess;
}

inline hipError_t hipStreamCreateWithFlags(hipStream_t* stream,
                                           unsigned int) {
  static int stream_token = 0;
  *stream = &stream_token;
  return hipSuccess;
}

inline hipError_t hipStreamDestroy(hipStream_t) {
  return hipSuccess;
}

inline hipError_t hipStreamSynchronize(hipStream_t) {
  return hipSuccess;
}

inline hipError_t hipEventCreate(hipEvent_t* event) {
  static int event_token = 0;
  *event = &event_token;
  return hipSuccess;
}

inline hipError_t hipEventDestroy(hipEvent_t) {
  return hipSuccess;
}

inline hipError_t hipEventRecord(hipEvent_t, hipStream_t = nullptr) {
  return hipSuccess;
}

inline hipError_t hipEventElapsedTime(float* milliseconds, hipEvent_t,
                                      hipEvent_t) {
  *milliseconds = 0.0f;
  return hipSuccess;
}

template <typename Kernel>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* blocks, Kernel, int, std::size_t) {
  *blocks = 1;
  return hipSuccess;
}

template <typename Kernel>
inline hipError_t hipLaunchCooperativeKernel(Kernel, dim3, dim3, void**,
                                             std::size_t,
                                             hipStream_t = nullptr) {
  return hipSuccess;
}

// Graph declarations are opt-in so a graph-disabled syntax build fails if the
// implementation leaks an unguarded graph symbol into that configuration.
#if defined(BF11_FAKE_HIP_ENABLE_GRAPHS)
using hipGraph_t = void*;
using hipGraphExec_t = void*;

enum hipStreamCaptureMode {
  hipStreamCaptureModeGlobal = 0,
  hipStreamCaptureModeThreadLocal = 1,
};

enum hipStreamCaptureStatus {
  hipStreamCaptureStatusNone = 0,
  hipStreamCaptureStatusActive = 1,
  hipStreamCaptureStatusInvalidated = 2,
};

inline hipError_t hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode) {
  return hipSuccess;
}

inline hipError_t hipStreamEndCapture(hipStream_t, hipGraph_t* graph) {
  static int graph_token = 0;
  *graph = &graph_token;
  return hipSuccess;
}

inline hipError_t hipStreamIsCapturing(hipStream_t,
                                       hipStreamCaptureStatus* status) {
  *status = hipStreamCaptureStatusNone;
  return hipSuccess;
}

inline hipError_t hipGraphInstantiate(hipGraphExec_t* executable,
                                      hipGraph_t, void*, void*,
                                      unsigned long long) {
  static int executable_token = 0;
  *executable = &executable_token;
  return hipSuccess;
}

inline hipError_t hipGraphLaunch(hipGraphExec_t, hipStream_t) {
  return hipSuccess;
}

inline hipError_t hipGraphExecDestroy(hipGraphExec_t) {
  return hipSuccess;
}

inline hipError_t hipGraphDestroy(hipGraph_t) {
  return hipSuccess;
}
#endif
