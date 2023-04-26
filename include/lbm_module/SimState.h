#ifndef STATE_LBM
#define STATE_LBM

#include "SimDomain.h"

#include <cuda_runtime_api.h>
#include "CudaErrorHandle.h"
#include <array>
#include <vector>

constexpr unsigned int D = 3;

struct SimStateDev;

struct SimStateHost
{
private:
	unsigned int nodeCount;

public:
	std::vector<float> rho_L;
	std::vector<std::array<float, 3>> u_L;
	SimStateHost(const SimDomain& sd);
	void memCpyMomentsFrom(SimStateDev st);
	unsigned int getNodeCount();
};

struct SimStateDev
{
private:
	unsigned int nodeCount;

public:
	float* rho_L = nullptr;
	float* u_L = nullptr;

	SimStateDev(const SimDomain& sd);
	void memCpyMomentsFrom(SimStateHost st);
	unsigned int getNodeCount();

	~SimStateDev();
};


#endif