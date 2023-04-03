#include "ParticleGenerator_P.h"
#include "ParticleGenerator_Kernels.h"

template class ParticleGenerator_Specialisation_P<float, 2>;
template class ParticleGenerator_Specialisation_P<double, 2>;
template class ParticleGenerator_Specialisation_P<float, 3>;
template class ParticleGenerator_Specialisation_P<double, 3>;

template class ParticleGenerator_P<float, 2>;
template class ParticleGenerator_P<double, 2>;
template class ParticleGenerator_P<float, 3>;
template class ParticleGenerator_P<double, 3>;

ParticleGenerator_Base_P::ParticleGenerator_Base_P(unsigned char sharpness):sharpness(sharpness){}

#pragma region part_spec_2D
template<typename T> 
ParticleGenerator_Specialisation_P<T,2>::ParticleGenerator_Specialisation_P(unsigned char sharpness): ParticleGenerator_Base_P(sharpness){}

template<typename T>
void ParticleGenerator_Specialisation_P<T, 2>::updateParticles(const T& time, vec<T, 2>* u_L, T* rho_L, const SimDomain<T,2> &sd)
{
	T g_L = gravity / (sd.getGridSize() / (sd.getTimeStep() * sd.getTimeStep()));
	updateParticles2D_kernel<T> << < (this->particlesTemp.size() + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize* this->warpCount >> >
		(time, this->particles.get(), u_L, rho_L, sd.getViscosity_L(), g_L, sd.getGridDim_L(0), sd.getGridDim_L(1), activeParticleCount_dev.get(), this->particlesTemp.size());
}

template<typename T>
void ParticleGenerator_Specialisation_P<T, 2>::calcParticleDensity(const T& time, T* particleDensity,const SimDomain<T,2> &sd)
{
	int maxNodeCount = sd.getMaxNodeCount();
	resetParticleDensity_kernel << <(maxNodeCount + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize* this->warpCount >> > (particleDensity, maxNodeCount);
	calcParticleDensity2D_kernel<T> << < (this->particlesTemp.size() + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize* this->warpCount >> >
		(time, this->particles.get(), particleDensity, sd.getGridDim_L(0), this->sharpness, this->particlesTemp.size());
}

template<typename T>
void ParticleGenerator_Specialisation_P<T, 2>::activateParticlesSimultaneously(const SimDomain<T, 2>& sd, vec<T, 2>* u_L, const T& currentTime)
{
	//DUMMY
}
#pragma endregion
//3D:
//----------------
template<typename T>
ParticleGenerator_Specialisation_P<T, 3>::ParticleGenerator_Specialisation_P(unsigned char sharpness) : ParticleGenerator_Base_P(sharpness) {}

template<typename T>
void ParticleGenerator_Specialisation_P<T, 3>::activateParticlesSimultaneously(const SimDomain<T, 3>& sd, vec<T, 3>* u_L, const T& currentTime)
{
	for (auto& source : sources)
	{
		if (!source.rate)
			throw std::runtime_error("ERROR::Particle Generator:: simultan-variant on but a rate-function is not defined");


		if (source.currentTempTimeStep < source.getMaxTimeStepPerEvent())
		{
			source.integratedParticleCount += source.rate(currentTime, sd.getTimeStep());
			source.currentTempTimeStep++;
		}
		else
		{
			source.currentTempTimeStep = 0;
			int count = std::round(source.integratedParticleCount);
			source.integratedParticleCount = 0;

			int loc = source.memLocation + source.notActiveMemLocation;
			source.notActiveMemLocation += count;
			if ((source.notActiveMemLocation - source.memLocation) <= source.getCount() && count > 0)
				activateParticles3D_kernel<T> << < (count + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize* this->warpCount >> >
				(loc, count, this->particles.get(), u_L, sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2), activeParticleCount_dev.get());
		}
	}
}


template<typename T, size_t D>
ParticleGenerator_P<T, D>::ParticleGenerator_P(unsigned char sharpness) : ParticleGenerator_Specialisation_P<T,D>(sharpness) {}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::createParticles(const SimDomain<T, D>& sd, bool simultan)
{
	this->simultan = simultan;
	vec<grid_size_t, 3> gridDim_L{ sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2) };
	for (auto& source : sources)
	{
		source.memLocation = this->particlesTemp.size();
		source.create(this->particlesTemp, sd.getGridSize(), sd.getTimeStep(), sd.getRho(), gridDim_L, this->activeParticleCount, simultan);
	}

	this->particles = make_location_array<location_gpu, ParticleData<T, D>>(this->particlesTemp.size());
	this->activeParticleCount_dev = make_location_array<location_gpu, int>(1);

	HANDLE_ERROR(cudaMemcpy(this->particles.get(), this->particlesTemp.data(), sizeof(ParticleData<T, D>) * this->particlesTemp.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(this->activeParticleCount_dev.get(), &(this->activeParticleCount), sizeof(int), cudaMemcpyHostToDevice));
}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::updateSourcePosition(const vec<T, D>& position, const vec<T, D* D>& R, const T& currentTime, int const &sourceID)
{
	for (auto& source : sources)
	{
		if (source.getID() == sourceID)
		{
			vec<T, D> sourceOldPos = source.getPosition();
			source.setPosition(position);
			source.setRotation(R);

			updateNonActiveParticles_kernel<T, D> << < (source.getCount() + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize* this->warpCount >> >
				(source.memLocation, source.getCount(), this->particles.get(), sourceOldPos, source.getPosition(), source.getRotMat());
		}

	}
}

template<typename T, size_t D>
bool ParticleGenerator_P<T, D>::isSimultan() const
{
	return simultan;
}

template<typename T>
void ParticleGenerator_Specialisation_P<T, 3>::updateParticles(const T& time, vec<T, 3>* u_L, T* rho_L,const SimDomain<T,3> &sd)
{
	T g_L = gravity / (sd.getGridSize() / (sd.getTimeStep() * sd.getTimeStep()));
	updateParticles3D_kernel<T> <<< (this->particlesTemp.size() + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize * this->warpCount >>>
		(time, this->particles.get(), u_L, rho_L, sd.getViscosity_L(), g_L, sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2), activeParticleCount_dev.get(), this->particlesTemp.size());
}



template<typename T>
void ParticleGenerator_Specialisation_P<T, 3>::calcParticleDensity(const T& time, T* particleDensity, const SimDomain<T, 3>& sd)
{
	int maxNodeCount = sd.getMaxNodeCount();
	resetParticleDensity_kernel <<<(maxNodeCount + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize * this->warpCount >> > (particleDensity, maxNodeCount);
	calcParticleDensity3D_kernel<T> <<< (this->particlesTemp.size() + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize * this->warpCount >>>
		(time, this->particles.get(), particleDensity, sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2), this->sharpness, this->particlesTemp.size());
}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::setCudaProp(cudaDeviceProp prop)
{
	this->prop = prop;
}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::setWarpCount(unsigned int warpCount)
{
	this->warpCount = warpCount;
}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::enableGravity()
{
	this->gravity = 9.81;
}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::appendSource(ParticleSource<T, D> source)
{
	sources.push_back(source);
}

template<typename T, size_t D>
const std::vector<ParticleData<T, D>>& ParticleGenerator_P<T, D>::getParticles() const
{
	HANDLE_ERROR(cudaMemcpy((void*)this->particlesTemp.data(), this->particles.get(), sizeof(ParticleData<T, D>) * this->particlesTemp.size(), cudaMemcpyDeviceToHost));
	return particlesTemp;
}

template<typename T, size_t D>
int ParticleGenerator_P<T, D>::getActiveParticleCount() const
{
	HANDLE_ERROR(cudaMemcpy((void*) &this->activeParticleCount, this->activeParticleCount_dev.get(), sizeof(int), cudaMemcpyDeviceToHost));
	return this->activeParticleCount;
}

template<typename T, size_t D>
void ParticleGenerator_P<T, D>::rescaleParticleData(const T& scale_u)
{
	rescaleParticleData_kernel<T,D> <<< (this->particlesTemp.size() + this->prop.warpSize * this->warpCount - 1) / (this->prop.warpSize * this->warpCount), this->prop.warpSize* this->warpCount >>>
		(this->particles.get(), scale_u, this->particlesTemp.size());
}
