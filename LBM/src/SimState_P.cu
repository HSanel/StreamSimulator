#include "SimState_P.h"
template struct SimState_P<float, 2>;
template struct SimState_P<float, 3>;
template struct SimState_P<double, 2>;
template struct SimState_P<double, 3>;

template<typename T, int D>
SimState_P<T, D>::SimState_P(const SimDomain<T, D>& sd, bool im_used) :im_used(im_used), particle_used(false)
{
	maxNodeCount = sd.getMaxNodeCount();

	rho_L = make_location_array<location_gpu, T>(sd.getMaxNodeCount());
	uMag = make_location_array<location_gpu, T>(1);
	zerothMomSum = make_location_array<location_gpu, T>(1);
	firstMomSum = make_location_array<location_gpu, T>(1);
	secondMomSum = make_location_array<location_gpu, T>(1);
	Fsg = make_location_array<location_gpu, vec<T, D>>(1);
	u_L = make_location_array<location_gpu, vec<T, D>>(sd.getMaxNodeCount());
	F_ext_L = make_location_array<location_gpu, vec<T, D>>(sd.getMaxNodeCount());
	f = make_location_array<location_gpu, T>(sd.getMaxNodeCount()*Q);
	f_star = make_location_array<location_gpu, T>(sd.getMaxNodeCount()*Q);
	f_t = make_location_array<location_gpu, T>(sd.getMaxNodeCount()*Q);

	if (im_used)
		u_unc_L = make_location_array<location_gpu, vec<T, D>>(sd.getMaxNodeCount());
}

template<typename T, int D>
void SimState_P<T, D>::allocateSecondMom()
{
	secondMom_L = make_location_array<location_gpu, T>(maxNodeCount);
}

template<typename T, int D>
void SimState_P<T, D>::allocateParticleDensity()
{
	particleDensity = make_location_array<location_gpu, T>(maxNodeCount);
	particle_used = true;
}

template<typename T, int D>
void SimState_P<T, D>::memCpyMoments(SimState<T, D>& st, const unsigned int direction)
{
	if (direction == HostToDev)
	{
		HANDLE_ERROR(cudaMemcpy(rho_L.get(), st.rho_L.get(), sizeof(T) * maxNodeCount, cudaMemcpyHostToDevice));
		if(particle_used)
			HANDLE_ERROR(cudaMemcpy(particleDensity.get(), st.particleDensity.get(), sizeof(T) * maxNodeCount, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(u_L.get(), st.u_L.get(), sizeof(vec<T, D>) * maxNodeCount, cudaMemcpyHostToDevice));
	}
	else if (direction == DevToHost)
	{
		HANDLE_ERROR(cudaMemcpy(st.rho_L.get(), rho_L.get(), sizeof(T) * maxNodeCount, cudaMemcpyDeviceToHost));
		if(particle_used)
			HANDLE_ERROR(cudaMemcpy(st.particleDensity.get(), particleDensity.get(), sizeof(T) * maxNodeCount, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(st.u_L.get(), u_L.get(), sizeof(vec<T, D>) * maxNodeCount, cudaMemcpyDeviceToHost));
	}
}