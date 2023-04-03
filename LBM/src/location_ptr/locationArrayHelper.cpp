#include "locationArrayHelper.h"
#include "locationPtr.h"

#ifdef WITH_CPPAD
#include <cppad/cppad.hpp>
#include <Real.h>
#endif

template<>
void copyDependentOnTypeCPUtoCPU<float>(array<float, 3> * cpuOut, array<float, 3> const * cpuIn, int numNodes)
{
	location_copy<location_cpu, location_cpu>(cpuOut, cpuIn, numNodes);
}

template<>
void copyDependentOnTypeCPUtoCPU<double>(array<double, 3> * cpuOut, array<float, 3> const * cpuIn, int numNodes)
{
	for(int i = 0; i < numNodes; ++i)
	{
		for(int j = 0; j < 3; ++j)
			cpuOut[i][j] = cpuIn[i][j];
	}
}

#ifdef WITH_CPPAD
template<>
void copyDependentOnTypeCPUtoCPU<CppAD::AD<double>>(array<CppAD::AD<double>, 3> * cpuOut, array<float, 3> const * cpuIn, int numNodes)
{
	for(int i = 0; i < numNodes; ++i)
	{
		for(int j = 0; j < 3; ++j)
			cpuOut[i][j] = cpuIn[i][j];
	}
}
#endif

#ifdef WITH_CUDA
template<>
void copyDependentOnTypeCPUtoGPU<float>(array<float, 3> * gpuOut, array<float, 3> const * cpuIn, int numNodes)
{
	location_copy<location_gpu, location_cpu>(gpuOut, cpuIn, numNodes);
}

template<>
void copyDependentOnTypeCPUtoGPU<double>(array<double, 3> * gpuOut, array<float, 3> const * cpuIn, int numNodes)
{
	auto doubleCPUVector = make_location_array<location_cpu, array<double, 3>>(numNodes);
	for(int i = 0; i < numNodes; ++i)
	{
		for(int j = 0; j < 3; ++j)
			doubleCPUVector[i][j] = cpuIn[i][j];
	}
	location_copy<location_gpu, location_cpu>(gpuOut, doubleCPUVector.get(), numNodes);
}

#endif