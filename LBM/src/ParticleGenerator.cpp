#include "ParticleGenerator.h"

template class ParticleGenerator_Specialisation<float,2>;
template class ParticleGenerator_Specialisation<double, 2>;
template class ParticleGenerator_Specialisation<float, 3>;
template class ParticleGenerator_Specialisation<double, 3>;

template class ParticleGenerator<float, 2>;
template class ParticleGenerator<double, 2>;
template class ParticleGenerator<float, 3>;
template class ParticleGenerator<double, 3>;

ParticleGenerator_Base::ParticleGenerator_Base(unsigned char sharpness):sharpness(sharpness) {}

template<typename T>
ParticleGenerator_Specialisation<T, 2>::ParticleGenerator_Specialisation(unsigned char sharpness) : ParticleGenerator_Base(sharpness) {}

template<typename T>
ParticleGenerator_Specialisation<T, 3>::ParticleGenerator_Specialisation(unsigned char sharpness) : ParticleGenerator_Base(sharpness) {}

template<typename T, size_t D>
ParticleGenerator<T, D>::ParticleGenerator(unsigned char sharpness) : ParticleGenerator_Specialisation<T, D>(sharpness) {}


template<typename T, size_t D>
void ParticleGenerator<T, D>::createParticles(const SimDomain<T,D> &sd)
{
	vec<grid_size_t, 3> gridDim_L{ sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2) };
	for (auto & source : sources)
	{
		source.create(particles, sd.getGridSize(), sd.getTimeStep(), sd.getRho(), gridDim_L, this->activeParticleCount, false);
	}
}

template<typename T>
void ParticleGenerator_Specialisation<T, 2>::updateParticles(const int& time, vec<T,2> *u_L, T* rho_L, T viscosity_L, const T& dh, const T& dt, vec<grid_size_t, 3> &gridDim_L)
{
	T Re, Cd, tau_p, mu;
	T g_L = gravity / (dh / (dt * dt));
	CustomVecLength<T, 2> vecLength;
	
	for (auto& particle : particles)
	{
		if (particle.is_active && time >= particle.time)
		{
			int indX = round(particle.position[0]);
			int indY = round(particle.position[1]);
			int pos = indY * gridDim_L[0] + indX;

			particle.position += particle.velocity;

			if (particle.position[0] >= 0 && particle.position[0] <= gridDim_L[0]-1
				&& particle.position[1] >= 0 && particle.position[1] <= gridDim_L[1]-1)
			{
				Re = (vecLength.length(u_L[pos] - particle.velocity) * particle.diameter) / viscosity_L;

				if (Re > 0.0)
				{
					if (Re <= 0.1)
						Cd = (T)24.0 / Re;
					else if (Re < 1000)
						Cd = ((T)24.0 / Re) * (1.0 + 0.15 * pow(Re, 0.687));
					else
						Cd = (T)0.44;

					mu = rho_L[pos] * viscosity_L;
					tau_p = particle.rho_p * particle.diameter * particle.diameter / ((T)18.0 * mu) * 24.0 / (Re * Cd);

					particle.velocity += (u_L[pos] - particle.velocity) * tau_p;
				}

				particle.velocity += (1.0 - rho_L[pos] / particle.rho_p) * vec<T, 2>{0.0, -g_L};
			}
			else
			{
				particle.is_active = false;
				this->activeParticleCount--;
			}		
		}
	}

}

template<typename T>
void ParticleGenerator_Specialisation<T, 3>::updateParticles(const int& time, vec<T, 3>* u_L, T* rho_L, T viscosity_L, const T& dh, const T& dt, vec<grid_size_t, 3> &gridDim_L)
{
	T Re, Cd, tau_p, mu;
	T g_L = gravity / (dh / (dt * dt));
	CustomVecLength<T, 3> vecLength;

	for (auto& particle : particles)
	{
		if (particle.is_active && time >= particle.time)
		{
			int indX = round(particle.position[0]);
			int indY = round(particle.position[1]);
			int indZ = round(particle.position[2]);
			int pos = (indZ * gridDim_L[1] + indY) * gridDim_L[0] + indX;

			particle.position += particle.velocity;

			if (particle.position[0] >= 0 && particle.position[0] <= gridDim_L[0]-1
				&& particle.position[1] >= 0 && particle.position[1] <= gridDim_L[1]-1
				&& particle.position[2] >= 0 && particle.position[2] <= gridDim_L[2]-1)
			{
				Re = (vecLength.length(u_L[pos] - particle.velocity) * particle.diameter) / viscosity_L;

				if (Re > 0.0)
				{
					if (Re <= 0.1)
						Cd = (T)24.0 / Re;
					else if (Re < 1000)
						Cd = ((T)24.0 / Re) * (1.0 + 0.15 * pow(Re, 0.687));
					else
						Cd = (T)0.44;

					mu = rho_L[pos] * viscosity_L;
					tau_p = particle.rho_p * particle.diameter * particle.diameter / ((T)18.0 * mu) * 24.0 / (Re * Cd);

					particle.velocity += (u_L[pos] - particle.velocity) * tau_p;
				}

				particle.velocity += (1.0 - rho_L[pos] / particle.rho_p) * vec<T, 3>{0.0, -g_L, 0.0};
			}
			else
			{
				particle.is_active = false;
				this->activeParticleCount--;
			}
		}
	}
}

template<typename T>
void ParticleGenerator_Specialisation<T, 2>::calcParticleDensity(const int& time, T* particleDensity, int maxNodeCount, vec<grid_size_t, 3> &gridDim_L)
{
	for (int i = 0; i < maxNodeCount ; ++i)
	{
		particleDensity[i] = 0;
	}

	for (auto& particle : particles)
	{
		if (particle.is_active && time >= particle.time)
		{
			int indX = round(particle.position[0]);
			int indY = round(particle.position[1]);

			int r, x_loc, y_loc;
			T delta;

			T volume = (T)(4.0 / 24.0) * PI<T> * particle.diameter * particle.diameter * particle.diameter;

			int kernelSize;
			if (sharpness == SHARP)
				kernelSize = 3;
			else
				kernelSize = 5;

			for (int yk = 0; yk < kernelSize; ++yk)
				for (int xk = 0; xk < kernelSize; ++xk)
				{
					r = kernelSize / 2;
					x_loc = indX - r + xk;
					y_loc = indY - r + yk;

					delta = deltaFunc(particle.position, { (T)x_loc, (T)y_loc }, sharpness);
					particleDensity[y_loc * gridDim_L[0] + x_loc] += volume * delta;
				}
		}
	}
}

template<typename T>
void ParticleGenerator_Specialisation<T, 3>::calcParticleDensity(const int& time, T* particleDensity, int maxNodeCount, vec<grid_size_t, 3>& gridDim_L)
{
	for (int i = 0; i < maxNodeCount; ++i)
	{
		particleDensity[i] = 0;
	}

	for (auto& particle : particles)
	{
		if (particle.is_active && time >= particle.time)
		{
			int indX = round(particle.position[0]);
			int indY = round(particle.position[1]);
			int indZ = round(particle.position[2]);

			int r, x_loc, y_loc, z_loc;
			T delta;

			T volume = (T)(4.0 / 24.0) * PI<T> * particle.diameter * particle.diameter * particle.diameter;

			int kernelSize;
			if (sharpness == SHARP)
				kernelSize = 3;
			else
				kernelSize = 5;

			for (int zk = 0; zk < kernelSize; ++zk)
				for (int yk = 0; yk < kernelSize; ++yk)
					for (int xk = 0; xk < kernelSize; ++xk)
					{
						r = kernelSize / 2;
						x_loc = indX - r + xk;
						y_loc = indY - r + yk;
						z_loc = indZ - r + zk;

						delta = deltaFunc(particle.position, { (T)x_loc, (T)y_loc, (T)z_loc }, sharpness);
						particleDensity[(z_loc * gridDim_L[1] + y_loc) * gridDim_L[0] + x_loc] += volume * delta;
					}
		}
	}
}

template<typename T, size_t D>
void ParticleGenerator<T, D>::enableGravity()
{
	this->gravity = 9.81;
}

template<typename T, size_t D>
void ParticleGenerator<T, D>::appendSource(ParticleSource<T, D> source)
{
	sources.push_back(source);
}

template<typename T, size_t D>
const std::vector<ParticleData<T, D>> &ParticleGenerator<T, D>::getParticles() const
{
	return particles;
}

template<typename T, size_t D>
int ParticleGenerator<T, D>::getActiveParticleCount() const
{
	return activeParticleCount;
}

template<typename T, size_t D>
void ParticleGenerator<T, D> ::rescaleParticleData(const T& scale_u)
{
	for (int i = 0; i < this->particles.size(); ++i)
		particles[i].velocity *= scale_u;
}