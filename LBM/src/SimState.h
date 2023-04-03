 #ifndef STATE_LBM
#define STATE_LBM
#include <memory>
#include "LBM_Types.h"
#include "SimDomain.h"
#include "ImmersedBody.h"

template<typename T, size_t D>
struct SimInitialiser
{
	T rho;
	vec<T, D> u;
	vec<T, D> F_ext;

	SimInitialiser(T rho, vec<T, D> u, vec<T, D> F_ext) :rho(rho), u(u), F_ext(F_ext) {}
	SimInitialiser() : SimInitialiser((T)1.0, vec<T, D>{}, vec<T, D>{}){}
};


template<typename T,size_t D>
struct SimState
{
	static constexpr size_t Q = SimDom_Specialisation<D>::Q;
	bool im_used, particle_used;
	scal_field<location_cpu, T> rho_L = nullptr;
	scal_field<location_cpu, T> uMag = nullptr;
	scal_field<location_cpu, T> particleDensity = nullptr;
	vec_field<location_cpu, InletGhostCells<T, D>, Q> inletGhosCells = nullptr;
	vec_field<location_cpu,T, D> u_L = nullptr;
	vec_field<location_cpu,T, D> u_unc_L = nullptr;
	vec_field<location_cpu,T, D> F_ext_L = nullptr;
	vec_field<location_cpu,T, Q> f = nullptr;
	vec_field<location_cpu,T, Q> f_star = nullptr;
	SimState(const SimDomain<T, D> &sd, bool im_used);
	void allocateParticleDensity();
private:
	int maxNodeCount;
};

//typedefs
template<typename T>
using SimInitialiser2D = SimInitialiser<T, 2>;

template<typename T>
using SimInitialiser3D = SimInitialiser<T, 3>;

template<typename T>
using SimState2D = SimState<T, 2>;

template<typename T>
using SimState3D = SimState<T, 3>;
#endif
