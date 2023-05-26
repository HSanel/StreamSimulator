#include <iostream>
#include "LBMKernels.h"


__device__ __host__ float dot(const float* a, const float* b, const unsigned int a_pos, const unsigned int b_pos)
{
	return a[arrayLayout(a_pos, 0)] * b[arrayLayout(b_pos, 0)] + a[arrayLayout(a_pos, 1)] * b[arrayLayout(b_pos, 1)] + a[arrayLayout(a_pos, 2)] * b[arrayLayout(b_pos, 2)];
}

__global__ void initializeField_kernel(float* rho_L, float* u_L, float* F_ext_L, float* f_star)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	//if (pos < sd_dev.maxNodeCount)
	//{

	//	rho_L[pos] = 1.f;
	//	u_L[arrayLayout(pos, 0)] = 0.f;
	//	u_L[arrayLayout(pos, 1)] = 0.f;
	//	u_L[arrayLayout(pos, 2)] = 0.f;

	//	F_ext_L[arrayLayout(pos, 0)] = 0.f;
	//	F_ext_L[arrayLayout(pos, 1)] = 0.f;
	//	F_ext_L[arrayLayout(pos, 2)] = 0.f;

	//	for (int i = 0; i < LBMDimensions::Q; ++i)
	//	{
	//		f_star[dataLayout(LBMDimensions::Q, sd_dev.alpha, pos, i)] = sd_dev.w[i] * rho_L[pos] * (1.f + dot(sd_dev.c, u_L, i, pos) / LBMDimensions::cs_sq
	//			+ dot(sd_dev.c, u_L, i, pos) * dot(sd_dev.c, u_L, i, pos) / (2.f * LBMDimensions::cs_sq * LBMDimensions::cs_sq) - dot(sd_dev.c, u_L, i, pos) / (2.f * LBMDimensions::cs_sq));
	//	}
	//}

}


void initializeField(const SimDomain* sd, const SimStateDev* st_dev)
{
	initializeField_kernel <<<(sd->getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >>>
		(st_dev->rho_L, st_dev->u_L, st_dev->F_ext_L, st_dev->f_star);
}
