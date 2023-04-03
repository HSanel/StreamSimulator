#include "SimState.h"
template struct SimState<float, 2>;
template struct SimState<float, 3>;
template struct SimState<double, 2>;
template struct SimState<double, 3>;

template<typename T, size_t D>
SimState<T, D>::SimState(const SimDomain<T, D> &sd, bool im_used):im_used(im_used), particle_used(false)
{
	maxNodeCount = sd.getMaxNodeCount();

	rho_L = make_location_array<location_cpu, T>(sd.getMaxNodeCount());
	uMag = make_location_array<location_cpu, T>(sd.getMaxNodeCount());
	inletGhosCells = make_location_array<location_cpu, vec<InletGhostCells<T, D>, Q>>(sd.getMaxNodeCount());
	u_L = make_location_array<location_cpu, vec<T, D>>(sd.getMaxNodeCount());
	F_ext_L = make_location_array<location_cpu, vec<T, D>>(sd.getMaxNodeCount());
	f = make_location_array<location_cpu, vec<T, Q>>(sd.getMaxNodeCount());
	f_star = make_location_array<location_cpu, vec<T, Q>>(sd.getMaxNodeCount());

	if (im_used)
	{
		u_unc_L = make_location_array<location_cpu, vec<T, D>>(sd.getMaxNodeCount());
	}
}

template<typename T, size_t D>
void SimState<T, D>::allocateParticleDensity()
{
	particleDensity = make_location_array<location_cpu, T>(maxNodeCount);
	for (int i = 0; i < maxNodeCount; ++i)
		particleDensity[i] = 0;
	particle_used = true;
}