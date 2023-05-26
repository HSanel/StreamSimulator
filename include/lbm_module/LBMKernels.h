#ifndef LBM_KERNELS
#define LBM_KERNELS
#include <cuda_runtime.h>
#include <array>
#include "SimDomain.h"
#include "SimState.h"
#include <DataStructureAlg.h>
#include "CudaErrorHandle.h"

static cudaDeviceProp prop;
 __constant__ SimDomain_dev sd_dev;



constexpr unsigned int warpCount = 8;

void initializeField(const SimDomain* sd, const SimStateDev* st_dev);
#endif