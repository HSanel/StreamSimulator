#ifndef STATE_LBM_P
#define STATE_LBM_P
#include "SimState.h"
#include "cudaErrorHandle.h"

#define HostToDev 1
#define DevToHost 2

template<typename T, size_t D>
struct SimInitialiser_P
{
	T rho;
	vec<T, D> u;
	vec<T, D> F_ext;
	SimInitialiser_P(T rho, vec<T, D> u, vec<T, D> F_ext) :rho(rho), u(u), F_ext(F_ext) {}
};

template<typename T, int D>
struct SimState_P
{
	static constexpr int Q = SimDom_Specialisation<D>::Q;
	bool im_used, particle_used;
	scal_field<location_gpu,T> rho_L = nullptr;
	scal_field<location_gpu, T> secondMom_L = nullptr;
	scal_field<location_gpu, T> particleDensity = nullptr;
	scal_field<location_gpu, T> uMag = nullptr;
	scal_field<location_gpu, T> zerothMomSum = nullptr;
	scal_field<location_gpu, T> firstMomSum = nullptr;
	scal_field<location_gpu, T> secondMomSum = nullptr;
	vec_field<location_gpu, T,D> Fsg = nullptr;
	vec_field<location_gpu,T, D> u_L = nullptr;
	vec_field<location_gpu,T, D> u_unc_L = nullptr;
	vec_field<location_gpu,T, D> F_ext_L = nullptr;
	scal_field<location_gpu,T> f = nullptr;
	scal_field<location_gpu,T> f_star = nullptr;
	scal_field<location_gpu, T> f_t = nullptr;
	SimState_P(const SimDomain<T, D>& sd, bool im_used);
	void memCpyMoments(SimState<T, D>& st, const unsigned int direction);
	void allocateParticleDensity();
	void allocateSecondMom();
private:
	int maxNodeCount;
};


#endif