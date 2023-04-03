// ========================================================================= //
//                                                                           //
// Filename: locationArrayMath.cu
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2020 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 07.2020 Andreas Giebel
//                                                                           //
// ========================================================================= //
#include "locationArrayMath.h"

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime.h>

#define DIVRND(a,b) ((a+b-1) / b)
#define BlockSize 1024

template<typename Real, int E>
__global__
void abs_kernel(array<Real, E> * result, array<Real, E> const * vec, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = abs(vec[i][e]);
	}
}

template<typename Real, int E>
__global__
void multArrays_kernel(array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = vec1[i][e] * vec2[i][e];
	}
}

template<typename Real, int E>
__global__
void subVector_kernel(array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = vec1[i][e] - vec2[i][e];
	}
}

template<typename Real, int E>
__global__
void addVector_kernel(array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = vec1[i][e] + vec2[i][e];
	}
}

template<typename Real, int E>
__global__
void multScalarDev_kernel(array<Real, E> * result, array<Real, E> const * vec1, Real * scalar, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
		Real alpha = *scalar;
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = alpha * vec1[i][e];
	}
}

template<typename Real, int E>
__global__
void multScalar_kernel(array<Real, E> * result, array<Real, E> const * vec1, Real scalar, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = scalar * vec1[i][e];
	}
}

template<typename Real, int E>
__global__
void divScalarDevice_kernel(array<Real, E> * result, array<Real, E> const * vec1, Real * scalar, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
		Real d = Real(1) / *scalar;
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = d * vec1[i][e];
	}
}

template<typename Real, int E>
__global__
void multScalarSingleArgument_kernel(array<Real, E> * result, Real scalar, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i < numNodes) {
#pragma unroll
		for(int e = 0; e < E; e++)
			result[i][e] *= scalar;
	}
}

template<typename Real, int E>
__global__
void axpyDev_kernel(array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
		Real a = *alpha;
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = a * vecX[i][e] + vecY[i][e];
	}
}

template<typename Real, int E>
__global__
void axpy_kernel(array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real alpha, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {

#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = alpha * vecX[i][e] + vecY[i][e];
	}
}


template<typename Real, int E>
__global__
void maxpy_kernel(array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
		Real a = -*alpha;
#pragma unroll
		for (int e = 0; e < E; e++)
			result[i][e] = a * vecX[i][e] + vecY[i][e];
	}
}

template<typename Real, int E>
__global__
void fill_kernel(array<Real, E> * vec, Real const value, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numNodes) {
#pragma unroll
		for (int e = 0; e < E; e++)
			vec[i][e] = value;
	}
}

template<typename Real, int E>
Real dot(location_gpu, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp, int numNodes)
{
	if (numNodes <= 0)
		return Real(0.);

	int GridSize = DIVRND(numNodes, BlockSize);
	multArrays_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(temp, vec1, vec2, numNodes);

	return sum<Real, E>(location_gpu(), temp, numNodes);
}

template<typename Real, int E>
Real sum(location_gpu, array<Real, E> const * vec, int numNodes)
{
	if (numNodes <= 0)
		return Real(0.);
	static cub::CachingDeviceAllocator tempAllocator;

	int sizei = numNodes * E;

	Real * resd = nullptr;
	const Real * vec_ptr = reinterpret_cast<const Real *>(vec);
	void * d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	tempAllocator.DeviceAllocate(reinterpret_cast<void **>(&resd), sizeof(Real));
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vec_ptr, resd, sizei);

	tempAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vec_ptr, resd, sizei);

	Real resh;
	cudaMemcpy(&resh, resd, sizeof(Real), cudaMemcpyDeviceToHost);

	tempAllocator.DeviceFree(d_temp_storage);
	tempAllocator.DeviceFree(resd);

	return resh;
}

template<typename Real, int E>
Real max(location_gpu, array<Real, E> const * vec, int numNodes)
{
	if (numNodes <= 0)
		return Real(0.);
	static cub::CachingDeviceAllocator tempAllocator;

	int sizei = numNodes * E;

	Real * resd = nullptr;
	const Real * vec_ptr = reinterpret_cast<const Real *>(vec);
	void * d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	tempAllocator.DeviceAllocate(reinterpret_cast<void **>(&resd), sizeof(Real));
	cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, vec_ptr, resd, sizei);

	tempAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, vec_ptr, resd, sizei);

	Real resh;
	cudaMemcpy(&resh, resd, sizeof(Real), cudaMemcpyDeviceToHost);

	tempAllocator.DeviceFree(d_temp_storage);
	tempAllocator.DeviceFree(resd);

	return resh;
}

template<typename Real, int E>
void dotDevice(location_gpu, Real * result, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp, array<Real, E> * temp2, int numNodes)
{
	if (numNodes <= 0)
		return;
	int GridSize = DIVRND(numNodes, BlockSize);
	multArrays_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(temp, vec1, vec2, numNodes);

	sumDevice<Real, E>(location_gpu(), result, temp, temp2, numNodes);
}

template<typename Real, int E>
void sumDevice(location_gpu, Real * result, array<Real, E> const * vec, array<Real, E> * temp, int numNodes)
{
	if (numNodes <= 0)
		return;
	int sizei = numNodes * E;

	const Real * vec_ptr = reinterpret_cast<const Real *>(vec);
	void * d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	size_t available_temp_storage_bytes = (numNodes - 1) * sizeof(Real) * E;

	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vec_ptr, result, sizei);

	if (available_temp_storage_bytes >= temp_storage_bytes) {
		d_temp_storage = reinterpret_cast<void *>(temp);
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vec_ptr, result, sizei);
	}
	else {
		static cub::CachingDeviceAllocator tempAllocator;
		tempAllocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vec_ptr, result, sizei);
		tempAllocator.DeviceFree(d_temp_storage);
	}
}

#define INSTANTIATE_DOT_DEVICE(Real, E) template void dotDevice<Real, E>(location_gpu, Real *, array<Real, E> const *, array<Real, E> const *, array<Real, E> *, array<Real, E> *, int);
#define INSTANTIATE_DOT(Real, E) template Real dot<Real, E>(location_gpu, array<Real, E> const *, array<Real, E> const *, array<Real, E> *, int);
#define INSTANTIATE_SUM_DEVICE(Real, E) template void sumDevice<Real, E>(location_gpu, Real *, array<Real, E> const *, array<Real, E> *, int);
#define INSTANTIATE_SUM(Real, E) template Real sum<Real, E>(location_gpu, array<Real, E> const *, int);
#define INSTANTIATE_MAX(Real, E) template Real max<Real, E>(location_gpu, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_DOT_DEVICE);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_DOT);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SUM_DEVICE);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SUM);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MAX);

template<typename Real, int E>
void multScalar(location_gpu, array<Real, E> * result, array<Real, E> const * vec, Real scalar, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	multScalar_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vec, scalar, numNodes);
}

template<typename Real, int E>
void multScalarDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	multScalarDev_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vec, scalar, numNodes);
}

template<typename Real, int E>
void multScalar(location_gpu, array<Real, E> * result, Real scalar, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	multScalarSingleArgument_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, scalar, numNodes);
}

#define INSTANTIATE_MULT_SCALAR_DEV(Real, E) template void multScalarDevice<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, Real *, int);
#define INSTANTIATE_MULT_SCALAR(Real, E) template void multScalar<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, Real, int);
#define INSTANTIATE_MULT_SCALAR2(Real, E) template void multScalar<Real, E>(location_gpu, array<Real, E> *, Real, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MULT_SCALAR);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MULT_SCALAR2);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MULT_SCALAR_DEV);

template<typename Real, int E>
void abs(location_gpu, array<Real, E> * result, array<Real, E> const * vec, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	abs_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vec, numNodes);
}


#define INSTANTIATE_ABS(Real, E) template void abs<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_ABS);

template<typename Real, int E>
void addVector(location_gpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	addVector_kernel<<<(unsigned int)GridSize, BlockSize>>>(result, vec1, vec2, numNodes);

}

#define INSTANTIATE_ADDVECTOR(Real, E) template void addVector<Real, E>(location_gpu, array<Real, E>*, array<Real, E> const *, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_ADDVECTOR);

template<typename Real, int E>
void divScalarDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	divScalarDevice_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vec, scalar, numNodes);
}

#define INSTANTIATE_DIV_SCALAR_DEV(Real, E) template void divScalarDevice<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_DIV_SCALAR_DEV);

template<typename Real, int E>
void subVector(location_gpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	subVector_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vec1, vec2, numNodes);
}

#define INSTANTIATE_SUB_VECTOR(Real, E) template void subVector<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SUB_VECTOR);

template<typename Real, int E>
void alphaXplusYDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	axpyDev_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vecX, vecY, alpha, numNodes);
}

template<typename Real, int E>
void alphaXplusY(location_gpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real alpha, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	axpy_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vecX, vecY, alpha, numNodes);
}

#define INSTANTIATE_AXPY(Real, E) template void alphaXplusY<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, Real, int);
#define INSTANTIATE_AXPY_DEV(Real, E) template void alphaXplusYDevice<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_AXPY_DEV);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_AXPY);

template<typename Real, int E>
void minusAlphaXplusYDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	maxpy_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(result, vecX, vecY, alpha, numNodes);
}

#define INSTANTIATE_MAXPY_DEV(Real, E) template void minusAlphaXplusYDevice<Real, E>(location_gpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MAXPY_DEV);

template<typename Real, int E>
void setZero(location_gpu, array<Real, E> * vec, int numNodes)
{
	cudaMemset(vec, 0, numNodes * E * sizeof(Real));
}

#define INSTANTIATE_SETZERO(Real, E) template void setZero<Real, E>(location_gpu, array<Real, E> *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SETZERO);

template<typename Real, int E>
void fill(location_gpu, array<Real, E> * vec, Real value, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	fill_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(vec, value, numNodes);
}

#define INSTANTIATE_FILL(Real, E) template void fill<Real, E>(location_gpu, array<Real, E> *, Real, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_FILL);


template <typename Real>
struct randomNumberGenerator
{
	__host__ __device__
		Real operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<Real> dist(-1.0f, 1.0f);
		rng.discard(n);

		return dist(rng);
	}
};

template<typename Real, int E>
void randomize(location_gpu, array<Real, E> * vec, int numNodes)
{
	thrust::counting_iterator<unsigned int> sequenceBegin(0);
	thrust::transform(thrust::cuda::par, sequenceBegin, sequenceBegin + (numNodes * E), reinterpret_cast<Real*>(vec), randomNumberGenerator<Real>());
}

#define INSTANTIATE_RANDOM(Real, E) template void randomize<Real, E>(location_gpu, array<Real, E> *, int);
EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_RANDOM);

#ifdef WITH_DOUBLE_PRECISION // TODO DW DOUBLE: Strange thrust compiler error with GCC: Solve it later
template<int E>
void convertDoubleToFloatArray(location_gpu, array<float, E> * vecOut, array<double, E> const * vecIn, int numNodes)
{
	auto convertDoubleToFloat = [=]  __device__(double in) { return static_cast<float>(in); };
	thrust::device_ptr<const double> thrustVecIn(reinterpret_cast<double const*>(vecIn));
	thrust::device_ptr<float> thrustVecOut(reinterpret_cast<float *>(vecOut));
		
	thrust::transform(thrustVecIn, thrustVecIn + E * numNodes, thrustVecOut, convertDoubleToFloat);
}
template void convertDoubleToFloatArray<3>(location_gpu, array<float, 3> *, array<double, 3> const *, int);
#endif


template<typename Real, int E>
__global__ void applyPermutation_kernel(array<Real, E> * __restrict__ vec, array<Real, E> const * vecIn, int const * permutation, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < numNodes)
	{
		auto destination = permutation[i];
		vec[destination] = vecIn[i];
	}
}

template<typename Real, int E>
void applyPermutation(location_gpu, array<Real, E> * vec, array<Real, E> const * vecIn, int const * permutation, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	applyPermutation_kernel<Real, E><<<(unsigned int)GridSize, BlockSize>>>(vec, vecIn, permutation, numNodes);
}

template void applyPermutation<float, 3>(location_gpu, array<float, 3> *, array<float, 3> const *, int const *, int);
template void applyPermutation<double, 3>(location_gpu, array<double, 3> *, array<double, 3> const *, int const *, int);
template void applyPermutation<float, 6>(location_gpu, array<float, 6> *, array<float, 6> const *, int const *, int);
template void applyPermutation<double, 6>(location_gpu, array<double, 6> *, array<double, 6> const *, int const *, int);

__global__ void computeInversePermutation_kernel(int * __restrict__ inversePermutation, int const * permutation, int numNodes)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < numNodes)
	{
		auto dest = permutation[i];
		inversePermutation[dest] = i;
	}
}

void computeInversePermutation(location_gpu, int * inversePermutation, int const * permutation, int numNodes)
{
	int GridSize = DIVRND(numNodes, BlockSize);
	computeInversePermutation_kernel<<<(unsigned int)GridSize, BlockSize>>>(inversePermutation, permutation, numNodes);
}
