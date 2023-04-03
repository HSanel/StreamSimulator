#pragma once

#include <cstddef>
#include "CudaMemoryInfo.h"

#include <memory>
#include <new>

#include <cuda_runtime_api.h>

struct CudaDeleter
{
	void operator() (void * p) const { cudaFree(p); }
};

template<typename T>
using cuda_ptr = std::unique_ptr<T, CudaDeleter>;

template<typename T>
cuda_ptr<T[]> make_cuda_array(std::size_t n)
{
	T * r = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&r), sizeof(T) * n);
	postCudaMalloc_Info(r != nullptr, sizeof(T) * n);
	if(!r)
		throw std::bad_alloc();
	return cuda_ptr<T[]>{r};
}

struct PinnedDeleter
{
	void operator() (void * p) const { cudaFreeHost(p); }
};

template<typename T>
using pinned_host_ptr = std::unique_ptr<T, PinnedDeleter>;

template<typename T>
pinned_host_ptr<T[]> make_pinned_array(std::size_t n)
{
	T * r = nullptr;
	cudaMallocHost(reinterpret_cast<void **>(&r), sizeof(T) * n);
	if(!r)
		throw std::bad_alloc();
	return pinned_host_ptr<T[]>{r};
}
