#ifndef ATOMIC_ADD
#define ATOMIC_ADD
#include "LBM_Types.h"

template <typename T>
struct CustomAtomicAdd
{
	__device__ T AtomicAdd(T* ref, T value)
	{
		extern __device__ void error(void);
		error(); // Ensure that we won't compile any un-specialized types
		return NULL;
	}
};

template <>
struct CustomAtomicAdd <float>
{
	__device__ float AtomicAdd(float* ref, float value)
	{
		return atomicAdd(ref, value);
	}
};

template <>
struct CustomAtomicAdd <double>
{
	__device__ double AtomicAdd(double* ref, double value)
	{
		// double is different becase it is only supported in later architectures
#if __CUDA_ARCH__ < 600
		unsigned long long int* address_as_ull = (unsigned long long int*)ref;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull,
				assumed,
				__double_as_longlong(
					value + __longlong_as_double(assumed)
				)
			);
		} while (assumed != old);
		return __longlong_as_double(old);
#else
		return atomicAdd(ref, value);
#endif
	}
};

template <typename T>
struct CustomAtomicMax
{
	__device__ T AtomicMax(T* ref, T value)
	{
		extern __device__ void error(void);
		error(); // Ensure that we won't compile any un-specialized types
		return NULL;
	}
};

template <>
struct CustomAtomicMax <float>
{
	__device__ static float atomicMax(float* address, float val)
	{
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_i, assumed,
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
		} while (assumed != old);
		return __int_as_float(old);
	}
};

template <>
struct CustomAtomicMax <double>
{
	__device__ static double atomicMax(double* address, double val)
	{
		unsigned long long int* address_as_i = (unsigned long long int*)address;
		unsigned long long int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_i, assumed,
				__double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
		} while (assumed != old);
		return __longlong_as_double(old);
	}
};

template <typename T>
struct SharedMemory
{
	//! @brief Return a pointer to the runtime-sized shared memory array.
	//! @returns Pointer to runtime-sized shared memory array
	__device__ T* getPointer()
	{
		extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
		Error_UnsupportedType();
		return (T*)0;
	}
	// TODO: Use operator overloading to make this class look like a regular array
};

template <>
struct SharedMemory <float>
{
	__device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }
};

template <>
struct SharedMemory <double>
{
	__device__ double* getPointer() { extern __shared__ double s_double[]; return s_double; }
};
#endif