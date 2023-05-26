#ifndef SOLVER
#define SOLVER
#include "SimDomain.h"
#include "SimState.h"

#include <cuda_runtime_api.h>
#include "CudaErrorHandle.h"

class LBMSolver
{
private:
	float _Cd, _Cl, _A;
	int _simulationStep = 0;
	bool _BGKused = false;
	bool _adaptiveTimeStep = true;
	float _maxSimulationTime;
	SimDomain _sd;
	SimStateDev _st_dev;
	SimStateHost _st_host;

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

	const SimStateHost* getSimState();
};
#endif // !SOLVER
