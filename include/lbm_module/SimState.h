#ifndef STATE_LBM
#define STATE_LBM

#include "SimDomain.h"
#include <DataStructureAlg.h>

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
	std::vector<float> u_L;
	std::vector<float> F_ext_L;

	SimStateHost(const SimDomain& sd);
	void memCpyMomentsFrom(const SimStateDev *st_dev);
	unsigned int getNodeCount();
};

struct SimStateDev
{
private:
	unsigned int nodeCount;

public:
	float* rho_L = nullptr;
	float* u_L = nullptr;
	float* F_ext_L = nullptr;

	float* f = nullptr;
	float* f_star = nullptr;

	SimStateDev(const SimDomain& sd);
	void memCpyMomentsFrom(const SimStateHost *st_host);
	unsigned int getNodeCount();

	//rule of five
	~SimStateDev();
	SimStateDev(SimStateDev&&);
	SimStateDev& operator=(SimStateDev&&);

	//copy deleted because the data are on the GPU
	SimStateDev(const SimStateDev& orig)= delete;
	SimStateDev& operator=(const SimStateDev& orig) = delete; 	
};


#endif