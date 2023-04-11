#include <LBMSolver.h>

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

	++simulationStep;

	if (adaptiveTimeStep)
	{
		//Problem mit überskalierten Zeitschritt
		uMax_mag = 0;
		HANDLE_ERROR(cudaMemcpy(st_P.uMag.get(), &uMax_mag, sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(&uMax_mag, st_P.uMag.get(), sizeof(float), cudaMemcpyDeviceToHost));

		if (uMax_mag > 0 && (uMax_mag < sd.get_uRef_L() - 0.02 || uMax_mag > sd.get_uRef_L() + 0.02))
		{
			sd.rescaleConFactors(uMax_mag, scale_u, scale_F);

			if (BGKused)
				rescaleMoments_BGK(scale_u, scale_F);
			else
				rescaleMoments_CM(scale_u, scale_F);
		}

	}
	cudaDeviceSynchronize();
}