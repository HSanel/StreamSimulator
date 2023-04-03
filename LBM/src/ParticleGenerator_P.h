#ifndef PARTICLE_GENERATOR_P
#define PARTICLE_GENERATOR_P
#include "ParticleSource.h"
#include "SimDomain.h"
#include "cudaErrorHandle.h"
#include "LBM_Types.h"
#include <vector>

class ParticleGenerator_Base_P
{
protected:
	cudaDeviceProp prop;
	int warpCount;
	unsigned char sharpness;
	int activeParticleCount = 0;
	scal_field<location_gpu, int> activeParticleCount_dev;
	ParticleGenerator_Base_P(unsigned char sharpness);
};

template<typename T, size_t D> class ParticleGenerator_Specialisation_P;

#pragma region Part_Spec_2D
template<typename T>
class ParticleGenerator_Specialisation_P<T, 2> : public ParticleGenerator_Base_P
{
protected:
	std::vector<ParticleSource<T, 2>> sources;
	std::vector<ParticleData<T, 2>> particlesTemp;
	scal_field<location_gpu, ParticleData<T, 2>> particles = nullptr;
	T gravity = 0.0;
	ParticleGenerator_Specialisation_P(unsigned char sharpness);
public:
	void updateParticles(const T& time, vec<T, 2>* u_L, T* rho_L, const SimDomain<T, 2>& sd);
	void calcParticleDensity(const T& time, T* particleDensity, const SimDomain<T, 2>& sd);
	void activateParticlesSimultaneously(const SimDomain<T, 2>& sd, vec<T, 2>* u_L, const T& currentTime);
};
#pragma endregion

template<typename T>
class ParticleGenerator_Specialisation_P<T, 3> : public ParticleGenerator_Base_P
{
protected:
	std::vector<ParticleSource<T, 3>> sources;
	std::vector<ParticleData<T, 3>> particlesTemp;
	scal_field<location_gpu, ParticleData<T, 3>> particles = nullptr;
	T gravity = 0.0;
	ParticleGenerator_Specialisation_P(unsigned char sharpness);
public:
	void updateParticles(const T& time, vec<T, 3>* u_L, T* rho_L, const SimDomain<T,3> &sd);
	void calcParticleDensity(const T& time, T* particleDensity, const SimDomain<T, 3>& sd);
	void activateParticlesSimultaneously(const SimDomain<T, 3>& sd, vec<T, 3>* u_L, const T& currentTime);
};

template<typename T, size_t D>
class ParticleGenerator_P : public ParticleGenerator_Specialisation_P<T, D>
{
	bool simultan = false;
public:
	ParticleGenerator_P(unsigned char sharpness = MEDIUM_SMOOTH);
	void createParticles(const SimDomain<T, D>& sd, bool simultan = false);
	void updateSourcePosition(const vec<T, D> &position,const vec<T, D* D> &R, const T& currentTime, int const& sourceID);
	void appendSource(ParticleSource<T, D> source);
	const std::vector<ParticleData<T, D>>& getParticles() const;
	int getActiveParticleCount() const;
	void enableGravity();
	void setCudaProp(cudaDeviceProp prop);
	void setWarpCount(unsigned int warpCount);
	void rescaleParticleData(const T& scale_u);
	bool isSimultan() const;
};
#endif