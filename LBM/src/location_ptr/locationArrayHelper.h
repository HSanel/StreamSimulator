#pragma once

#include "array.h"

template<typename Real>
void copyDependentOnTypeCPUtoCPU(array<Real, 3> * cpuOut, array<float, 3> const * cpuIn, int numNodes);

#ifdef WITH_CUDA
template<typename Real>
void copyDependentOnTypeCPUtoGPU(array<Real, 3> * gpuOut, array<float, 3> const * cpuIn, int numNodes);
#endif
