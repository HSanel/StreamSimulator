#pragma push_macro("CPUGPU")
#ifdef CPUGPU
#undef CPUGPU
#endif

#ifdef __CUDACC__
#define CPUGPU __device__ __host__
#else
#define CPUGPU
#endif

#pragma push_macro("FORCEINLINE")
#ifdef FORCEINLINE
#undef FORCEINLINE
#endif

#if defined(__GNUC__) || defined(__CUDA_LIBDEVICE__) || defined(__CUDACC_RTC__)
#define FORCEINLINE __inline__ __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORCEINLINE __forceinline
#endif

#pragma push_macro("ALIGN")
#ifdef ALIGN
#undef ALIGN
#endif

#ifdef __CUDACC__
#define ALIGN(x) __align__(x)
#elif defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

#pragma push_macro("RESTRICT")
#ifdef RESTRICT
#undef RESTRICT
#endif

#if defined(__CUDACC__) || !defined(_MSC_VER)
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

#ifndef CPUGPU_MACROS_PUSH_INCLUDE_GUARD
#define CPUGPU_MACROS_PUSH_INCLUDE_GUARD
static FORCEINLINE int divRnd(int n, int d) { return n / d + (n % d != 0); }
#endif
