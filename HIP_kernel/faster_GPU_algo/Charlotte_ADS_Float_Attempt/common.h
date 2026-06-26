#ifndef ADS_FLOAT_HIP_COMMON_H_
#define ADS_FLOAT_HIP_COMMON_H_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using uint = unsigned int;

constexpr unsigned NOT_FOUND = 0xffffffffu;
constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr unsigned ADS_WARP_SIZE = 32u;

template <typename A, typename B>
__host__ __device__ __forceinline__ constexpr std::common_type_t<A, B>
hd_min(A a, B b) {
    using R = std::common_type_t<A, B>;
    const R left = static_cast<R>(a);
    const R right = static_cast<R>(b);
    return left < right ? left : right;
}

template <typename A, typename B>
__host__ __device__ __forceinline__ constexpr std::common_type_t<A, B>
hd_max(A a, B b) {
    using R = std::common_type_t<A, B>;
    const R left = static_cast<R>(a);
    const R right = static_cast<R>(b);
    return left > right ? left : right;
}

inline void hip_check_impl(hipError_t status, const char* expression,
                           const char* file, int line) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s:%d: HIP call %s failed: %s (%d)\n", file,
                     line, expression, hipGetErrorString(status),
                     static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define HIP_CHECK(expr) hip_check_impl((expr), #expr, __FILE__, __LINE__)

__device__ __forceinline__ void break_pt() { __builtin_trap(); }

// HIP does not expose the PTX cache modifiers used by CUB ThreadLoad/ThreadStore.
// The original algorithm already places explicit fences around its
// producer/consumer hand-offs, so volatile loads/stores preserve the polling
// behavior without relying on NVIDIA-specific cache operators.
template <typename T>
__device__ __forceinline__ T global_load(const T* address) {
    return *reinterpret_cast<volatile const T*>(address);
}

template <typename T>
__device__ __forceinline__ void global_store(T* address, T value) {
    *reinterpret_cast<volatile T*>(address) = value;
}

__host__ __device__ __forceinline__ unsigned low_mask32(unsigned length) {
    return length >= 32u
               ? 0xffffffffu
               : (length == 0u ? 0u : ((1u << length) - 1u));
}

__host__ __device__ __forceinline__ unsigned long long low_mask64(
    unsigned length) {
    return length >= 64u
               ? 0xffffffffffffffffull
               : (length == 0u ? 0ull : ((1ull << length) - 1ull));
}

__device__ __forceinline__ unsigned find_ms_bit(unsigned bit_mask) {
    return bit_mask == 0u
               ? NOT_FOUND
               : static_cast<unsigned>(31 - __clz(bit_mask));
}

__device__ __forceinline__ unsigned count_bit(unsigned bit_mask) {
    return static_cast<unsigned>(__popc(bit_mask));
}

__host__ __device__ __forceinline__ unsigned extract_bits(
    unsigned bit_mask, unsigned start_pos, unsigned length) {
    if (start_pos >= 32u || length == 0u) {
        return 0u;
    }
    const unsigned width =
        length > (32u - start_pos) ? (32u - start_pos) : length;
    return (bit_mask >> start_pos) & low_mask32(width);
}

__host__ __device__ __forceinline__ unsigned set_bits(
    unsigned bit_mask, unsigned value, unsigned start_pos, unsigned length) {
    if (start_pos >= 32u || length == 0u) {
        return bit_mask;
    }
    const unsigned width =
        length > (32u - start_pos) ? (32u - start_pos) : length;
    const unsigned field_mask = low_mask32(width) << start_pos;
    return (bit_mask & ~field_mask) | ((value << start_pos) & field_mask);
}

__host__ __device__ __forceinline__ unsigned long long extract_bits_64(
    unsigned long long bit_mask, unsigned start_pos, unsigned length) {
    if (start_pos >= 64u || length == 0u) {
        return 0ull;
    }
    const unsigned width =
        length > (64u - start_pos) ? (64u - start_pos) : length;
    return (bit_mask >> start_pos) & low_mask64(width);
}

__host__ __device__ __forceinline__ unsigned long long set_bits_64(
    unsigned long long bit_mask, unsigned long long value, unsigned start_pos,
    unsigned length) {
    if (start_pos >= 64u || length == 0u) {
        return bit_mask;
    }
    const unsigned width =
        length > (64u - start_pos) ? (64u - start_pos) : length;
    const unsigned long long field_mask = low_mask64(width) << start_pos;
    return (bit_mask & ~field_mask) | ((value << start_pos) & field_mask);
}

// Emulates PTX fns.b32 for the positive, one-based offsets used by ADDS.
__device__ __forceinline__ unsigned find_nth_bit(unsigned bit_mask,
                                                  unsigned base,
                                                  unsigned offset) {
    if (offset == 0u || base >= 32u) {
        return NOT_FOUND;
    }
    unsigned remaining = bit_mask & (0xffffffffu << base);
    for (unsigned n = 1u; n < offset; ++n) {
        if (remaining == 0u) {
            return NOT_FOUND;
        }
        remaining &= remaining - 1u;
    }
    return remaining == 0u
               ? NOT_FOUND
               : static_cast<unsigned>(__ffs(static_cast<int>(remaining)) - 1);
}

__device__ __forceinline__ unsigned get_lane_id() {
    // The build forces wave32 and the host verifies warpSize == 32.
    return static_cast<unsigned>(threadIdx.x) & (ADS_WARP_SIZE - 1u);
}

__device__ __forceinline__ unsigned get_clock32() {
    return static_cast<unsigned>(wall_clock64());
}

__device__ __forceinline__ unsigned thread_id_x() {
    return static_cast<unsigned>(threadIdx.x);
}
__device__ __forceinline__ unsigned block_id_x() {
    return static_cast<unsigned>(blockIdx.x);
}
__device__ __forceinline__ unsigned block_dim_x() {
    return static_cast<unsigned>(blockDim.x);
}
__device__ __forceinline__ unsigned grid_dim_x() {
    return static_cast<unsigned>(gridDim.x);
}

__host__ __device__ __forceinline__ int round_up(int a, int r) {
    return ((a + r - 1) / r) * r;
}

// HIP warp masks are 64-bit. ADDS has exactly 32 logical lanes, so the public
// compatibility wrappers retain 32-bit masks and widen only at the intrinsic.
__device__ __forceinline__ unsigned warp_active_mask() {
    return static_cast<unsigned>(__activemask()) & FULL_MASK;
}

__device__ __forceinline__ unsigned warp_ballot_sync(unsigned mask,
                                                      bool predicate) {
    return static_cast<unsigned>(
               __ballot_sync(static_cast<unsigned long long>(mask), predicate)) &
           FULL_MASK;
}

template <typename T>
__device__ __forceinline__ unsigned warp_match_any(unsigned active_mask,
                                                   T value) {
    return static_cast<unsigned>(__match_any_sync(
               static_cast<unsigned long long>(active_mask), value)) &
           FULL_MASK;
}

template <typename T>
__device__ __forceinline__ T warp_shfl(unsigned mask, T value,
                                       unsigned src_lane) {
    return __shfl_sync(static_cast<unsigned long long>(mask), value,
                       static_cast<int>(src_lane), ADS_WARP_SIZE);
}

template <typename T>
__device__ __forceinline__ T warp_shfl_up(unsigned mask, T value,
                                          unsigned delta) {
    return __shfl_up_sync(static_cast<unsigned long long>(mask), value, delta,
                          ADS_WARP_SIZE);
}

template <typename T>
__device__ __forceinline__ T warp_shfl_down(unsigned mask, T value,
                                            unsigned delta) {
    return __shfl_down_sync(static_cast<unsigned long long>(mask), value, delta,
                            ADS_WARP_SIZE);
}

__device__ __forceinline__ void warp_sync(unsigned mask = FULL_MASK) {
    (void)mask;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    // AMD waves execute in lockstep.  The wave barrier is the convergence
    // primitive corresponding to CUDA's __syncwarp; the workgroup-scoped
    // fence makes preceding LDS/shared-memory writes visible to peer lanes.
    __builtin_amdgcn_wave_barrier();
#endif
    __threadfence_block();
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    static_assert(std::is_arithmetic<T>::value,
                  "warp_reduce_sum requires an arithmetic type");
    for (unsigned offset = ADS_WARP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        value += warp_shfl_down(FULL_MASK, value, offset);
    }
    return warp_shfl(FULL_MASK, value, 0u);
}

__device__ __forceinline__ void warp_exclusive_sum(int value, int& exclusive,
                                                    int& total) {
    int inclusive = value;
    const unsigned lane = get_lane_id();
    for (unsigned offset = 1u; offset < ADS_WARP_SIZE; offset <<= 1u) {
        const int other = warp_shfl_up(FULL_MASK, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    exclusive = inclusive - value;
    total = warp_shfl(FULL_MASK, inclusive, ADS_WARP_SIZE - 1u);
}

#endif  // ADS_FLOAT_HIP_COMMON_H_
