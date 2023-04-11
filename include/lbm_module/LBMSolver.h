#ifndef SOLVER
#define SOLVER
#include "SimDomain.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cudaErrorHandle.h"

class LBMSolver
{
private:
	float Cd, Cl, A;
	int simulationStep = 0;
	bool BGKused = false;
	bool adaptiveTimeStep = true;
	float maxSimulationTime;
	SimDomain sd;

	void collision_CM(float r_vis);
	void collision_BGK(float r_vis);
	void streaming(const float& C_u, const float& C_p);
	void updateMoments();
	void rescaleMoments_BGK(const float& scale_u, const  float& scale_F);
	void rescaleMoments_CM(const float& scale_u, const float& scale_F);

public:
	LBMSolver(SimDomain simDom);

	void enableBGK(bool BGK_on = true);
	void solve();
};
#endif // !SOLVER
