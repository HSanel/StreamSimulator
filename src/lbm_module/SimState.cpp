#include "SimState.h"

SimStateHost::SimStateHost(const SimDomain& sd): 
	nodeCount(sd.getMaxNodeCount()), rho_L(std::vector<float>(sd.getMaxNodeCount())), u_L(std::vector<float>(packCount(sd.getMaxNodeCount()) * LBMDimensions::D)), F_ext_L(std::vector<float>(packCount(sd.getMaxNodeCount()) * LBMDimensions::D)){}

void SimStateHost::memCpyMomentsFrom(const SimStateDev *st_dev)
{
	HANDLE_ERROR(cudaMemcpy(rho_L.data(), st_dev->rho_L, nodeCount * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(u_L.data(), st_dev->u_L, packCount(nodeCount) * sizeof(float) * LBMDimensions::D, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(F_ext_L.data(), st_dev->F_ext_L, packCount(nodeCount) * sizeof(float) * LBMDimensions::D, cudaMemcpyDeviceToHost));
}

unsigned int SimStateHost::getNodeCount()
{
	return nodeCount;
}

//######################################

SimStateDev::SimStateDev(const SimDomain& sd) : nodeCount(sd.getMaxNodeCount())
{
	HANDLE_ERROR(cudaMalloc((void**)&rho_L, nodeCount * sizeof(float) * LBMDimensions::D));
	HANDLE_ERROR(cudaMalloc((void**)&u_L, packCount(nodeCount) * sizeof(float) * LBMDimensions::D));
	HANDLE_ERROR(cudaMalloc((void**)&F_ext_L, packCount(nodeCount) * sizeof(float) * LBMDimensions::D));

	HANDLE_ERROR(cudaMalloc((void**)&f, nodeCount * sizeof(float) * LBMDimensions::Q));
	HANDLE_ERROR(cudaMalloc((void**)&f_star, nodeCount * sizeof(float) * LBMDimensions::Q));
}

void SimStateDev::memCpyMomentsFrom(const SimStateHost *st_host)
{
	HANDLE_ERROR(cudaMemcpy(rho_L, st_host->rho_L.data(), nodeCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(u_L, st_host->u_L.data(), nodeCount * sizeof(std::array<float, LBMDimensions::D>), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(F_ext_L, st_host->F_ext_L.data(), nodeCount * sizeof(std::array<float, LBMDimensions::D>), cudaMemcpyHostToDevice));
}

unsigned int SimStateDev::getNodeCount()
{
	return nodeCount;
}

SimStateDev::~SimStateDev()
{
	cudaFree(rho_L);
	cudaFree(u_L);
	cudaFree(F_ext_L);
	cudaFree(f);
	cudaFree(f_star);
}

SimStateDev::SimStateDev(SimStateDev&& other): rho_L(other.rho_L), u_L(other.u_L), F_ext_L(other.F_ext_L), f(other.f), f_star(other.f_star){}

SimStateDev& SimStateDev::operator=(SimStateDev&& other)
{
	if (this != &other)
	{
		cudaFree(rho_L);
		cudaFree(u_L);
		cudaFree(F_ext_L);
		cudaFree(f);
		cudaFree(f_star);
	}

	rho_L = other.rho_L;
	u_L = other.u_L;
	F_ext_L = other.F_ext_L;
	f = other.f;
	f_star = other.f_star;

	return *this;
}

