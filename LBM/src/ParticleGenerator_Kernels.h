#ifndef PARTICLE_KERNELS
#define PARTICLE_KERNELS
#include "ParticleSource.h"
#include <cuda_runtime_api.h>
#include "LBM_Types.h"
#include "CustomCudaExtensions.h"

#pragma region part_generator_kernels_2D
template<typename T>
__global__ void updateParticles2D_kernel(T time, ParticleData<T,2> *particles, vec<T, 2>* u_L, T* rho_L, T viscosity_L, T gravity, int gridDimX_L, int gridDimY_L,int *activeParticleCount, int particleCount)
{
	T Re, Cd, tau_p, mu;
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomVecLength<T,2> vecLength;
	
	if (ID < particleCount && particles[ID].is_active && time >= particles[ID].time)
	{
		int indX = roundf(particles[ID].position[0]);
		int indY = roundf(particles[ID].position[1]);
		int pos = indY * gridDimX_L + indX;

		particles[ID].position += particles[ID].velocity;

		if (particles[ID].position[0] >= 0 && particles[ID].position[0] <= gridDimX_L - 1
			&& particles[ID].position[1] >= 0 && particles[ID].position[1] <= gridDimY_L - 1)
		{
			Re = (vecLength.length(u_L[pos] - particles[ID].velocity) * particles[ID].diameter) / viscosity_L;

			if (Re > 0.0)
			{
				if (Re <= 0.1)
					Cd = (T)24.0 / Re;
				else if (Re < 1000)
					Cd = ((T)24.0 / Re) * (1.0 + 0.15 * pow(Re, 0.687));
				else
					Cd = (T)0.44;

				T rho_init =  bilinearDensInterpolation(particles[ID].position, rho_L, gridDimX_L);
				mu = rho_init * viscosity_L;
				tau_p = particles[ID].rho_p * particles[ID].diameter * particles[ID].diameter / ((T)18.0 * mu) * 24.0 / (Re * Cd);
				
				vec<T, 2> u_init = bilinearVelInterpolation(particles[ID].position, u_L, gridDimX_L);
				particles[ID].velocity += (u_init - particles[ID].velocity) * tau_p;
			}

			particles[ID].velocity += (1.0 - rho_L[pos] / particles[ID].rho_p) * vec<T, 2>{0.0, -gravity};
		}
		else
		{
			particles[ID].is_active = false;
			atomicAdd(activeParticleCount,-1);
		}
	}
	
}

template<typename T>
__global__ void calcParticleDensity2D_kernel(T time, ParticleData<T, 2>* particles, T* particleDensity, int gridDimX_L, unsigned char sharpness, int particleCount)
{
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomAtomicAdd<T> atomAdd;

	if (ID < particleCount && particles[ID].is_active && time >= particles[ID].time)
	{
		int indX = roundf(particles[ID].position[0]);
		int indY = roundf(particles[ID].position[1]);
		int r, x_loc, y_loc;
		T delta;

		T volume = (T)(4.0 / 24.0) * PI<T> *particles[ID].diameter * particles[ID].diameter * particles[ID].diameter;

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

				delta = deltaFunc<T, 2>(particles[ID].position, { (T)x_loc, (T)y_loc }, sharpness);
				atomAdd.AtomicAdd(&(particleDensity[y_loc * gridDimX_L + x_loc]), volume * delta);
			}
	}

}
#pragma endregion

//3D
//--------------------
template<typename T>
__global__ void updateParticles3D_kernel(T time, ParticleData<T, 3>* particles, vec<T, 3>* u_L, T* rho_L, T viscosity_L, T gravity, int gridDimX_L, int gridDimY_L, int gridDimZ_L, int* activeParticleCount, int particleCount)
{
	T Re, Cd, tau_p, mu;
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomVecLength<T, 3> vecLength;

	if (ID < particleCount && particles[ID].is_active && time >= particles[ID].time)
	{
		particles[ID].position += particles[ID].velocity;

		if (particles[ID].position[0] >= 0 && particles[ID].position[0] <= gridDimX_L
			&& particles[ID].position[1] >= 0 && particles[ID].position[1] <= gridDimY_L
			&& particles[ID].position[2] >= 0 && particles[ID].position[2] <= gridDimZ_L)
		{
			vec<T, 3> u_init = trilinearVelInterpolation(particles[ID].position, u_L, gridDimX_L, gridDimY_L, gridDimZ_L);
			T rho_init = trilinearDensInterpolation(particles[ID].position, rho_L, gridDimX_L, gridDimY_L, gridDimZ_L);

			Re = (vecLength.length(u_init - particles[ID].velocity) * particles[ID].diameter) / viscosity_L;

			if (Re > 0.0)
			{
				if (Re <= 0.1)
					Cd = (T)24.0 / Re;
				else if (Re < 1000)
					Cd = ((T)24.0 / Re) * (1.0 + 0.15 * pow(Re, 0.687));
				else
					Cd = (T)0.44;

				
				mu = rho_init * viscosity_L;
				tau_p = particles[ID].rho_p * particles[ID].diameter * particles[ID].diameter / ((T)18.0 * mu) * 24.0 / (Re * Cd);
				
				
				particles[ID].velocity += (u_init - particles[ID].velocity) * tau_p;
			}

			particles[ID].velocity += (1.0 - rho_init / particles[ID].rho_p) * vec<T, 3>{0.0, -gravity};
		}
		else
		{
			particles[ID].is_active = false;
			atomicAdd(activeParticleCount, -1);
		}
	}
}

template <typename T, int D>
__global__ void updateNonActiveParticles_kernel(int location, int count, ParticleData<T, D>* particles, vec<T, D> sourceOldPos, vec<T, D> sourcePos, vec<T, D*D> sourceRot)
{
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomVecLength<T, 3> vecLength;

	if (ID < count)
	{
		if (!particles[ID + location].is_active)
		{
			particles[ID + location].position = sourcePos + sourceRot *(particles[ID + location].position - sourceOldPos);
		}
	}
}

template<typename T>
__global__ void activateParticles3D_kernel(int location, int count, ParticleData<T, 3>* particles, vec<T, 3>* u_L, int gridDimX_L, int gridDimY_L, int gridDimZ_L, int* activeParticleCount)
{
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ID < count)
	{
		vec<T, 3> sampPos = particles[ID + location].position;
		int x = (int)roundf(sampPos[0]);
		int y = (int)roundf(sampPos[1]);
		int z = (int)roundf(sampPos[2]);

		
		if (x >= 0 && x < gridDimX_L
			&& y >= 0 && y < gridDimY_L
			&& z >= 0 && z < gridDimZ_L)
		{
			particles[ID + location].is_active = true;
			int pos = (z * gridDimY_L + y) * gridDimX_L + x;
			particles[ID + location].velocity = u_L[pos];
			atomicAdd(activeParticleCount,1);
		}
	}
}

template<typename T, int D>
__global__ void rescaleParticleData_kernel(ParticleData<T, D>* particles, T scale_u, int particleCount)
{
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if (ID < particleCount)
	{
		particles[ID].velocity *= scale_u;
	}
}

template<typename T>
__global__ void resetParticleDensity_kernel(T* particleDensity, int maxNodeCount)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos < maxNodeCount)
		particleDensity[pos] = 0;
}


template<typename T>
__global__ void calcParticleDensity3D_kernel(T time, ParticleData<T, 3>* particles, T* particleDensity, int gridDimX_L, int gridDimY_L, int gridDimZ_L, unsigned char sharpness, int particleCount)
{
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomAtomicAdd<T> atomAdd;
	if (ID < particleCount && particles[ID].is_active && time >= particles[ID].time)
	{
		int indX = roundf(particles[ID].position[0]);
		int indY = roundf(particles[ID].position[1]);
		int indZ = roundf(particles[ID].position[2]);
		int r, x_loc, y_loc, z_loc;
		T delta;

		T volume = (T)(4.0 / 24.0) * PI<T> *particles[ID].diameter * particles[ID].diameter * particles[ID].diameter;

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
					z_loc = indY - r + zk;

					if (x_loc >= 0 && x_loc < gridDimX_L
						&& y_loc >= 0 && y_loc < gridDimY_L
						&& z_loc >= 0 && z_loc < gridDimZ_L)
					{
						delta = deltaFunc<T, 3>(particles[ID].position, { (T)x_loc, (T)y_loc, (T)z_loc }, sharpness);
						atomAdd.AtomicAdd(&(particleDensity[(z_loc * gridDimY_L + y_loc) * gridDimX_L + x_loc]), volume * delta);
					}
				}
	}
}

#endif