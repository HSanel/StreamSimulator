#ifndef LBM_KERNEL
#define LBM_KERNEL

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cudaErrorHandle.h"


__global__ void test_kernel(float* rho, float* u)
{
	//auto idx = blockDim.x * blockIdx.x + threadIdx.x;

	//rho[idx] = 5.f;

	//u[idx * 2 + 0] = 6.0;
	//u[idx * 2 + 1] = 7.0;
};

void test_kernel_call(float* rho, float* u)
{
	test_kernel<<<2,8>>>(rho, u);
}
#endif