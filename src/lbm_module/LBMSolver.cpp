#include "LBMSolver.h"
#include "lbm_kernels.cu"
#include <cuda_runtime.h>

LBMSolver::LBMSolver(SimDomain simDom):sd(simDom){}

void LBMSolver::streaming(const float& C_u, const float& C_p)
{

}

void LBMSolver::collision_BGK(float r_vis)
{

}

void LBMSolver::collision_CM(float r_vis)
{

}

void LBMSolver::updateMoments()
{
	SimDomain sd{ {2,2,2},1,1, 1 };
	sd.calcConvFactors(2);

	SimStateHost st_host{ sd };
	SimStateDev st_dev{ sd };

	for (int i = 0; i < st_host.getNodeCount(); ++i)
	{
		st_host.rho_L[i] = i;
		st_host.u_L[i] = std::array<float, 3>{(float)i, (float)i, (float)i};
	}

	st_dev.memCpyMomentsFrom(st_host);
	
	//test_kernel<<<2,8>>>(st_host.rho, st_host.u);

	st_host.memCpyMomentsFrom(st_dev);
}

void LBMSolver::rescaleMoments_CM(const float& scale_u, const float& scale_F)
{

}

void LBMSolver::rescaleMoments_BGK(const float& scale_u, const float& scale_F)
{

}

void LBMSolver::enableBGK(bool BGK_on) { BGKused = BGK_on; }

void LBMSolver::solve()
{
	float uMax_mag = 0, scale_u = 1, scale_F = 1;

	streaming(sd.getC_u(), sd.getC_p());

	if (BGKused)
	{
		collision_BGK(sd.getRelaxationConstant());
	}
	else
	{
		collision_CM(sd.getRelaxationConstant());
	}

	updateMoments();

	++simulationStep;

	//if (adaptiveTimeStep)
	//{
	//	//Problem mit überskalierten Zeitschritt
	//	uMax_mag = 0;
	//	HANDLE_ERROR(cudaMemcpy(st_P.uMag.get(), &uMax_mag, sizeof(float), cudaMemcpyHostToDevice));
	//	HANDLE_ERROR(cudaMemcpy(&uMax_mag, st_P.uMag.get(), sizeof(float), cudaMemcpyDeviceToHost));

	//	if (uMax_mag > 0 && (uMax_mag < sd.get_uRef_L() - 0.02 || uMax_mag > sd.get_uRef_L() + 0.02))
	//	{
	//		sd.rescaleConFactors(uMax_mag, scale_u, scale_F);

	//		if (BGKused)
	//			rescaleMoments_BGK(scale_u, scale_F);
	//		else
	//			rescaleMoments_CM(scale_u, scale_F);
	//	}

	//}
	cudaDeviceSynchronize();
}