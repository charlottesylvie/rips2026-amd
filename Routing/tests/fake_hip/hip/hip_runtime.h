#pragma once

#include <cstddef>

using hipStream_t = void*;
using hipError_t = int;

struct hipDeviceProp_t {
  int warpSize = 64;
};

constexpr hipError_t hipSuccess = 0;
constexpr unsigned int hipStreamNonBlocking = 1;

inline const char* hipGetErrorString(hipError_t) {
  return "fake HIP error";
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
  return hipSuccess;
}

inline hipError_t hipMemGetInfo(std::size_t* free_bytes,
                                std::size_t* total_bytes) {
  *free_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  *total_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
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
