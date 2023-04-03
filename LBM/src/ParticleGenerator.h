#ifndef PARTICLE_GENERATOR
#define PARTICLE_GENERATOR
#include "ParticleSource.h"
#include "SimDomain.h"
#include <vector>

class ParticleGenerator_Base
{
protected:
	unsigned char sharpness = MEDIUM_SMOOTH;
	int activeParticleCount = 0;
	ParticleGenerator_Base(unsigned char sharpness);
};

template<typename T, size_t D> class ParticleGenerator_Specialisation;

template<typename T>
class ParticleGenerator_Specialisation<T, 2> :public ParticleGenerator_Base
{
protected:

	std::vector<ParticleSource<T, 2>> sources;
	std::vector<ParticleData<T, 2>> particles;
	T gravity = 0.0;
	ParticleGenerator_Specialisation(unsigned char sharpness);
public:
	void updateParticles(const int& time, vec<T, 2>* u_L, T* rho_L, T viscosity_L, const T& dh, const T& dt, vec<grid_size_t, 3> &gridDim_L);
	void calcParticleDensity(const int& time, T* particleDensity, int maxNodeCount, vec<grid_size_t, 3>& gridDim_L);
};

template<typename T>
class ParticleGenerator_Specialisation<T, 3> :public ParticleGenerator_Base
{
protected:
	std::vector<ParticleSource<T, 3>> sources;
	std::vector<ParticleData<T, 3>> particles;
	T gravity = 0.0;
	ParticleGenerator_Specialisation(unsigned char sharpness);
public:
	void updateParticles(const int& time, vec<T, 3>* u_L, T* rho_L, T viscosity_L, const T& dh, const T& dt, vec<grid_size_t, 3> &gridDim_L);
	void calcParticleDensity(const int& time, T* particleDensity, int maxNodeCount, vec<grid_size_t, 3>& gridDim_L);
};

template<typename T, size_t D>
class ParticleGenerator :public ParticleGenerator_Specialisation<T, D>
{
public:
	ParticleGenerator(unsigned char sharpness = MEDIUM_SMOOTH);
	void createParticles(const SimDomain<T, D>& sd);
	void appendSource(ParticleSource<T, D> source);
	const std::vector<ParticleData<T, D>>& getParticles() const;
	int getActiveParticleCount() const;
	void enableGravity();
	void rescaleParticleData(const T& scale_u);
};

#endif // !PARTICLE_GENERATOR
