#include "SimState.h"

SimStateHost::SimStateHost(const SimDomain& sd): nodeCount(sd.getMaxNodeCount()), rho_L(std::vector<float>(sd.getMaxNodeCount())), u_L(std::vector<std::array<float,3>>(sd.getMaxNodeCount())){}

void SimStateHost::memCpyMomentsFrom(SimStateDev st)
{
	HANDLE_ERROR(cudaMemcpy(rho_L.data(), st.rho_L, nodeCount * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(u_L.data(), st.u_L, nodeCount * sizeof(float) * D, cudaMemcpyDeviceToHost));
}

unsigned int SimStateHost::getNodeCount()
{
	return nodeCount;
}

SimStateDev::SimStateDev(const SimDomain& sd) : nodeCount(sd.getMaxNodeCount())
{
	HANDLE_ERROR(cudaMalloc((void**)&rho_L, nodeCount * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&u_L, nodeCount * sizeof(float[3])));
}

void SimStateDev::memCpyMomentsFrom(SimStateHost st)
{
	HANDLE_ERROR(cudaMemcpy(rho_L, st.rho_L.data(), nodeCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(u_L, st.u_L.data(), nodeCount * sizeof(float) * D, cudaMemcpyHostToDevice));
}

unsigned int SimStateDev::getNodeCount()
{
	return nodeCount;
}

SimStateDev::~SimStateDev()
{
	cudaFree(rho_L);
	cudaFree(u_L);
}