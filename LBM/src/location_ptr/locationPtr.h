#pragma once
#include "location.h"

#ifdef WITH_CUDA
#include "cuda_ptr.h"
#endif



#include <memory>

#include <cstring>

namespace detail
{
	template<typename Location, typename T>
	struct location_ptr_impl;

	template<typename T>
	struct location_ptr_impl<location_cpu, T>
	{
		using type = std::unique_ptr<T[]>;
	};

#ifdef WITH_CUDA
	template<typename T>
	struct location_ptr_impl<location_gpu, T>
	{
		using type = cuda_ptr<T[]>;
	};
#endif
}

template<typename Location, typename T>
using location_ptr = typename detail::location_ptr_impl<Location, T>::type;

namespace detail
{
	template<typename Location, typename T>
	struct make_location_array_helper;

	template<typename T>
	struct make_location_array_helper<location_cpu, T>
	{
		static std::unique_ptr<T[]> apply(std::size_t count)
		{
			return std::unique_ptr<T[]>(new T[count]);
		}
	};

#ifdef WITH_CUDA
	template<typename T>
	struct make_location_array_helper<location_gpu, T>
	{
		static cuda_ptr<T[]> apply(std::size_t count)
		{
			return make_cuda_array<T>(count);
		}
	};
#endif
}

template<typename Location, typename T>
location_ptr<Location, T> make_location_array(std::size_t count)
{
	if(!count)
		return nullptr;
	return detail::make_location_array_helper<Location, T>::apply(count);
}

namespace detail
{
	template<typename LocIn, typename LocOut> struct copy_helper;

	template<>
	struct copy_helper<location_cpu, location_cpu>
	{
		template<typename T>
		static void apply(T * out, T const * in, std::size_t count)
		{
			//std::memcpy(out, in, sizeof(T) * count);
			std::copy(in, in+count, out);
		}
	};

#ifdef WITH_CUDA
	template<>
	struct copy_helper<location_cpu, location_gpu>
	{
		template<typename T>
		static void apply(T * out, T const * in, std::size_t count)
		{
			cudaMemcpy(out, in, sizeof(T) * count, cudaMemcpyHostToDevice);
		}
	};

	template<>
	struct copy_helper<location_gpu, location_gpu>
	{
		template<typename T>
		static void apply(T * out, T const * in, std::size_t count)
		{
			cudaMemcpy(out, in, sizeof(T) * count, cudaMemcpyDeviceToDevice);
		}
	};

	template<>
	struct copy_helper<location_gpu, location_cpu>
	{
		template<typename T>
		static void apply(T * out, T const * in, std::size_t count)
		{
			cudaMemcpy(out, in, sizeof(T) * count, cudaMemcpyDeviceToHost);
		}
	};
#endif
}

template<typename LocOut, typename LocIn, typename T>
void location_copy(T * out, T const * in, std::size_t count)
{
	detail::copy_helper<LocIn, LocOut>::apply(out, in, count);
}

template<typename OtherLocation>
void location_cast(void *) = delete;
