#ifndef SOLVER_KERNELS
#define SOLVER_KERNELS

#include "SimState_P.h"
#include "ImmersedBody.h"
#include "LBM_Types.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "CustomCudaExtensions.h"

template<typename T, int D, int Q>
__constant__ SimDomain_dev<T, D, Q> sd_dev;

cudaDeviceProp prop;
constexpr unsigned int warpCount = 8;

template<typename T, int D, int Q>
__global__ void initializeField_kernel(T* rho_L, vec<T, D>* u_L, vec<T, D>* F_ext_L, T* f_star, SimInitialiser_P<T, D> simInit)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int alpha = sd_dev<T, D, Q>.alpha;

	if (pos < sd_dev<T, D, Q>.maxNodeCount)
	{

		rho_L[pos] = simInit.rho;
		u_L[pos] = simInit.u;
		F_ext_L[pos] = simInit.F_ext;

		for (int i = 0; i < Q; ++i)
		{
			f_star[dataLayout(Q, alpha, pos, i)] = sd_dev<T, D, Q>.w[i] * rho_L[pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L[pos]) / cs_sq<T>
				+dot(sd_dev<T, D, Q>.c[i], u_L[pos]) * dot(sd_dev<T, D, Q>.c[i], u_L[pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L[pos], u_L[pos]) / ((T)2.0 * cs_sq<T>));
		}
	}

}

//3D
//--------------------
template<typename T>
__device__ int calcPos3D(int idx, int x, int y, int z)
{
	return ((z + static_cast<int>(sd_dev<T, 3, 27>.c[idx][2])) * sd_dev<T, 3, 27>.gridDim_L[1] + y + static_cast<int>(sd_dev<T, 3, 27>.c[idx][1])) * sd_dev<T, 3, 27>.gridDim_L[0] + x + static_cast<int>(sd_dev<T, 3, 27>.c[idx][0]);
}

template<typename T, int D, int Q>
__device__ T calc_f_with_velBound(int idx, T f_star, T rhoVB_border, vec<T, D> uwVB)
{
	return f_star - 2.0 * sd_dev<T, D, Q>.w[idx] * (rhoVB_border / cs_sq<T>) * dot(sd_dev<T, D, Q>.c[idx], uwVB);
}

template<typename T, int D, int Q>
__device__ T calc_f_with_presBound(int idx, T f_star, T rhoPB_border, vec<T, D> uwPB)
{
	T scalarProdSquared = dot(sd_dev<T, D, Q>.c[idx], uwPB) * dot(sd_dev<T, D, Q>.c[idx], uwPB);
	return -f_star + 2.0 * sd_dev<T, D, Q>.w[idx] * rhoPB_border * (1.0 + scalarProdSquared / (2.0 * cs_sq<T> *cs_sq<T>) - dot(uwPB, uwPB) * dot(uwPB, uwPB) / (2.0 * cs_sq<T>));
}

//template<typename T>
//__device__ void wetNodeBoundTest_celBound(T* f, T* f_star, vec<T, 3> u_w, int pos)
//{
//	T rho=((T)1.0-u_w[0])*(f[dataLayout(27, alpha, pos, 3)] + f[dataLayout(27, alpha, pos, 4)] + f[dataLayout(27, alpha, pos, 5)] )
//}

template<typename T>
__global__ void streaming3D_kernel(T* f, T* f_star, vec<T, 3>* u_L, T* rho_L, T C_u, T C_p)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;


	if (x < sd_dev<T, 3, 27>.gridDim_L[0] && y < sd_dev<T, 3, 27>.gridDim_L[1] && z < sd_dev<T, 3, 27>.gridDim_L[2])
	{

		int pos = (z * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 3, 27>.alpha;

		T rhoVB_border = rho_L[pos];
		T rhoPB_border = 1.0;
		vec<T, 3> uwVB{ 0.0, 0.0, 0.0 }, uwPB{ 0.0, 0.0, 0.0 };
		bool velBound_defined = false;
		bool pressBound_defined = false;

		for (int vbIdx = 0; vbIdx < sd_dev<T, 3, 27>.velBound_Count; ++vbIdx)
		{
			auto& velBound = sd_dev<T, 3, 27>.velBounds[vbIdx];

			if (x == 0 && velBound.side == left)	//left
			{
				if (y < sd_dev<T, 3, 27>.gridDim_L[1] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					velBound_defined = true;
					uwVB[0] = velBound.u_w/C_u;
					uwVB[1] = 0.0;
					uwVB[2] = 0.0;
					break;
				}
			}
			else if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 && velBound.side == right)	//right
			{
				if (y < sd_dev<T, 3, 27>.gridDim_L[1] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					velBound_defined = true;
					uwVB[0] = -velBound.u_w/C_u;
					uwVB[1] = 0.0;
					uwVB[2] = 0.0;
					break;
				}
			}
			else if (y == 0 && velBound.side == bottom)	//bottom
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = velBound.u_w/C_u;
					uwVB[2] = 0.0;
					break;
				}
			}
			else if (y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 && velBound.side == top)	//top
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = -velBound.u_w/C_u;
					uwVB[2] = 0.0;
					break;
				}
			}
			else if (z == 0 && velBound.side == back)	//back
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && y < sd_dev<T, 3, 27>.gridDim_L[1])
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = 0.0;
					uwVB[2] = velBound.u_w/C_u;
					break;
				}
			}
			else if (z == sd_dev<T, 3, 27>.gridDim_L[2] - 1 && velBound.side == front)	//front
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && y < sd_dev<T, 3, 27>.gridDim_L[1])
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = 0.0;
					uwVB[2] = -velBound.u_w/C_u;
					break;
				}
			}
		}

		for (int pbIdx = 0; pbIdx < sd_dev<T, 3, 27>.pressBound_Count; ++pbIdx)
		{
			auto& pressBound = sd_dev<T, 3, 27>.pressBounds[pbIdx];

			if (x == 0 && pressBound.side == left)	//left
			{
				if (y < sd_dev<T, 3, 27>.gridDim_L[1] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					pressBound_defined = true;
					int pos_next = (z * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x + 1;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w/C_p) / cs_sq<T> + (T)1.0;
					break;
				}
			}
			else if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 && pressBound.side == right)	//right
			{
				if (y < sd_dev<T, 3, 27>.gridDim_L[1] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					pressBound_defined = true;
					int pos_next = (z * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x - 1;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w/C_p) / cs_sq<T> +(T)1.0;
					break;
				}
			}
			else if (y == 0 && pressBound.side == bottom)	//bottom
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					pressBound_defined = true;
					int pos_next = (z * sd_dev<T, 3, 27>.gridDim_L[1] + y + 1) * sd_dev<T, 3, 27>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w/C_p) / cs_sq<T> +(T)1.0;
					break;
				}
			}
			else if (y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 && pressBound.side == top)	//top
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					pressBound_defined = true;
					int pos_next = (z * sd_dev<T, 3, 27>.gridDim_L[1] + y - 1) * sd_dev<T, 3, 27>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w/C_p) / cs_sq<T> +(T)1.0;
					break;
				}
			}
			else if (z == 0 && pressBound.side == back)	//back
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && y < sd_dev<T, 3, 27>.gridDim_L[1])
				{
					pressBound_defined = true;
					int pos_next = ((z + 1) * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w / C_p) / cs_sq<T> +(T)1.0;
					break;
				}
			}
			else if (z == sd_dev<T, 3, 27>.gridDim_L[2] - 1 && pressBound.side == front)	//front
			{
				if (x < sd_dev<T, 3, 27>.gridDim_L[0] && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					pressBound_defined = true;
					int pos_next = ((z - 1) * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w/C_p) / cs_sq<T> +(T)1.0;
					break;
				}
			}
		}

		int i = 0;
		f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 1;		//2
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 2)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 2)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 2)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 2;
		if (x == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 1)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 1)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 1)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 3;
		if (y == sd_dev<T, 3, 27>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 4)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 4)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 4)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 4;
		if (y == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 3)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 3)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 3)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 5;
		if (z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 6)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 6)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 6)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 6;
		if (z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 5)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 5)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 5)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 7;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || y == sd_dev<T, 3, 27>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 8)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 8)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 8)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 8;
		if (x == 0 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 7)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 7)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 7)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 9;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 10)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 10)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 10)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 10;
		if (x == 0 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 9)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 9)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 9)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 11;
		if (y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 12)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 12)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 12)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 12;
		if (y == 0 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 11)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 11)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 11)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 13;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 14)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 14)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 14)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 14;
		if (x == 0 || y == sd_dev<T, 3, 27>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 13)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 13)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 13)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 15;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 16)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 16)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 16)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 16;
		if (x == 0 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 15)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 15)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 15)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 17;
		if (y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 18)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 18)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 18)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 18;
		if (y == 0 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 17)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 17)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 17)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 19;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 20)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 20)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 20)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 20;
		if (x == 0 || y == 0 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 19)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 19)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 19)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 21;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 22)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 22)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 22)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 22;
		if (x == 0 || y == 0 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 21)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 21)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 21)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 23;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || y == 0 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 24)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 24)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 24)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 24;
		if (x == 0 || y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 23)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 23)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 23)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 25;
		if (x == 0 || y == sd_dev<T, 3, 27>.gridDim_L[1] - 1 || z == sd_dev<T, 3, 27>.gridDim_L[2] - 1)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 26)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 26)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 26)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];

		i = 26;
		if (x == sd_dev<T, 3, 27>.gridDim_L[0] - 1 || y == 0 || z == 0)
		{
			if (velBound_defined)
				f[dataLayout(27, alpha, pos, 25)] = calc_f_with_velBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(27, alpha, pos, 25)] = calc_f_with_presBound<T, 3, 27>(i, f_star[dataLayout(27, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(27, alpha, pos, 25)] = f_star[dataLayout(27, alpha, pos, i)];
		}
		else
			f[dataLayout(27, alpha, calcPos3D<T>(i, x, y, z), i)] = f_star[dataLayout(27, alpha, pos, i)];
	}
}


template<typename T, int D, int Q>
__global__ void collision_kernel_BGK(T* rho_L, vec<T, D>* u_L, vec<T, D>* F_ext_L, T* f, T* f_star, T r_vis)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int alpha = sd_dev<T, D, Q>.alpha;

	if (pos < sd_dev<T, D, Q>.maxNodeCount)
	{
		for (int i = 0; i < Q; ++i)
		{
			T F_i = ((T)1.0 - r_vis / (T)2.0) * sd_dev<T, D, Q>.w[i] *
				dot((sd_dev<T, D, Q>.c[i] - u_L[pos]) * ((T)1.0 / cs_sq<T>) + dot(sd_dev<T, D, Q>.c[i], u_L[pos])
					/ (cs_sq<T> *cs_sq<T>) * sd_dev<T, D, Q>.c[i], F_ext_L[pos]);

			T fi_eq = sd_dev<T, D, Q>.w[i] * rho_L[pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L[pos]) / cs_sq<T> +dot(sd_dev<T, D, Q>.c[i], u_L[pos]) *
				dot(sd_dev<T, D, Q>.c[i], u_L[pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L[pos], u_L[pos]) / ((T)2.0 * cs_sq<T>));

			f_star[dataLayout(Q, alpha, pos, i)] = f[dataLayout(Q, alpha, pos, i)] - r_vis * (f[dataLayout(Q, alpha, pos, i)] - fi_eq) + F_i;
		}
	}
}

template<typename T>
__global__ void collision3D_kernel_CM(T* rho_L, vec<T, 3>* u_L, T* secondMom_L, vec<T, 3>* F_ext_L, T* f, T* f_star, T r_vis, T zerothMomentMean = 0, T firstMomentMean = 0, T seconMomentMean = 0)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	CustomSqrt<T> customSqrt;

	if (x < sd_dev<T, 3, 27>.gridDim_L[0] && y < sd_dev<T, 3, 27>.gridDim_L[1] && z < sd_dev<T, 3, 27>.gridDim_L[2])
	{
		int pos = (z * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 3, 27>.alpha;


		T f0 = f[dataLayout(27, alpha, pos, 0)],
			f1 = f[dataLayout(27, alpha, pos, 1)],
			f2 = f[dataLayout(27, alpha, pos, 2)],
			f3 = f[dataLayout(27, alpha, pos, 3)],
			f4 = f[dataLayout(27, alpha, pos, 4)],
			f5 = f[dataLayout(27, alpha, pos, 5)],
			f6 = f[dataLayout(27, alpha, pos, 6)],
			f7 = f[dataLayout(27, alpha, pos, 7)],
			f8 = f[dataLayout(27, alpha, pos, 8)],
			f9 = f[dataLayout(27, alpha, pos, 9)],
			f10 = f[dataLayout(27, alpha, pos, 10)],
			f11 = f[dataLayout(27, alpha, pos, 11)],
			f12 = f[dataLayout(27, alpha, pos, 12)],
			f13 = f[dataLayout(27, alpha, pos, 13)],
			f14 = f[dataLayout(27, alpha, pos, 14)],
			f15 = f[dataLayout(27, alpha, pos, 15)],
			f16 = f[dataLayout(27, alpha, pos, 16)],
			f17 = f[dataLayout(27, alpha, pos, 17)],
			f18 = f[dataLayout(27, alpha, pos, 18)],
			f19 = f[dataLayout(27, alpha, pos, 19)],
			f20 = f[dataLayout(27, alpha, pos, 20)],
			f21 = f[dataLayout(27, alpha, pos, 21)],
			f22 = f[dataLayout(27, alpha, pos, 22)],
			f23 = f[dataLayout(27, alpha, pos, 23)],
			f24 = f[dataLayout(27, alpha, pos, 24)],
			f25 = f[dataLayout(27, alpha, pos, 25)],
			f26 = f[dataLayout(27, alpha, pos, 26)];

		T rho = rho_L[pos];
		T ux = u_L[pos][0];
		T uy = u_L[pos][1];
		T uz = u_L[pos][2];
		T Fx = F_ext_L[pos][0];
		T Fy = F_ext_L[pos][1];
		T Fz = F_ext_L[pos][2];

		T r0 = sd_dev<T, 3, 27>.zerothRelaxationTime,
			r1 = sd_dev<T, 3, 27>.lowRelaxationTimes,
			r2 = r_vis,
			r3 = r_vis,
			r4, r5, r6, r7;

		if (sd_dev<T, 3, 27>.localRelaxation)
		{
			T tau_local = sd_dev<T, 3, 27>.param_0 * std::abs(rho) / zerothMomentMean
				+ sd_dev<T, 3, 27>.param_1 * std::abs(rho) * customSqrt.sqrt(ux * ux + uy * uy + uz * uz) / firstMomentMean
				+ sd_dev<T, 3, 27>.param_2 * std::abs(secondMom_L[pos]) / seconMomentMean
				+ sd_dev<T, 3, 27>.param_3;
			//r3 = r;
			r4 = (T)1.0/(tau_local+0.5);
			r5 = (T)1.0/(tau_local+0.5);
			r6 = (T)1.0/(tau_local+0.5);
			r7 = (T)1.0/(tau_local+0.5);
		}
		else
		{
			r3 = sd_dev<T, 3, 27>.r_3;
			r4 = sd_dev<T, 3, 27>.r_3;
			r5 = sd_dev<T, 3, 27>.r_4;
			r6 = sd_dev<T, 3, 27>.r_5;
			r7 = sd_dev<T, 3, 27>.r_6;
		}

		T f_t0 = r0 * (f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19 + f20 + f21 + f22 + f23 + f24 + f25 + f26 - rho);
		T f_t1 = -r1 * (f0 * ux + f3 * ux + f4 * ux + f5 * ux + f6 * ux + f11 * ux + f12 * ux + f17 * ux + f18 * ux + f1 * (ux - 1.0) + f2 * (ux + 1.0) + f7 * (ux - 1.0) + f8 * (ux + 1.0) + f9 * (ux - 1.0) + f10 * (ux + 1.0) + f13 * (ux - 1.0) + f14 * (ux + 1.0) + f15 * (ux - 1.0) + f16 * (ux + 1.0) + f19 * (ux - 1.0) + f20 * (ux + 1.0) + f21 * (ux - 1.0) + f22 * (ux + 1.0) + f23 * (ux - 1.0) + f24 * (ux + 1.0) + f25 * (ux + 1.0) + f26 * (ux - 1.0));
		T f_t2 = -r1 * (f0 * uy + f1 * uy + f2 * uy + f5 * uy + f6 * uy + f9 * uy + f10 * uy + f15 * uy + f16 * uy + f3 * (uy - 1.0) + f4 * (uy + 1.0) + f7 * (uy - 1.0) + f8 * (uy + 1.0) + f11 * (uy - 1.0) + f12 * (uy + 1.0) + f13 * (uy + 1.0) + f14 * (uy - 1.0) + f17 * (uy - 1.0) + f18 * (uy + 1.0) + f19 * (uy - 1.0) + f20 * (uy + 1.0) + f21 * (uy - 1.0) + f22 * (uy + 1.0) + f23 * (uy + 1.0) + f24 * (uy - 1.0) + f25 * (uy - 1.0) + f26 * (uy + 1.0));
		T f_t3 = -r1 * (f0 * uz + f1 * uz + f2 * uz + f3 * uz + f4 * uz + f7 * uz + f8 * uz + f13 * uz + f14 * uz + f5 * (uz - 1.0) + f6 * (uz + 1.0) + f9 * (uz - 1.0) + f10 * (uz + 1.0) + f11 * (uz - 1.0) + f12 * (uz + 1.0) + f15 * (uz + 1.0) + f16 * (uz - 1.0) + f17 * (uz + 1.0) + f18 * (uz - 1.0) + f19 * (uz - 1.0) + f20 * (uz + 1.0) + f21 * (uz + 1.0) + f22 * (uz - 1.0) + f23 * (uz - 1.0) + f24 * (uz + 1.0) + f25 * (uz - 1.0) + f26 * (uz + 1.0));
		T f_t4 = r2 * (f1 * uy * (ux - 1.0) + f2 * uy * (ux + 1.0) + f3 * ux * (uy - 1.0) + f4 * ux * (uy + 1.0) + f9 * uy * (ux - 1.0) + f10 * uy * (ux + 1.0) + f11 * ux * (uy - 1.0) + f12 * ux * (uy + 1.0) + f15 * uy * (ux - 1.0) + f16 * uy * (ux + 1.0) + f17 * ux * (uy - 1.0) + f18 * ux * (uy + 1.0) + f7 * (ux - 1.0) * (uy - 1.0) + f8 * (ux + 1.0) * (uy + 1.0) + f13 * (ux - 1.0) * (uy + 1.0) + f14 * (ux + 1.0) * (uy - 1.0) + f19 * (ux - 1.0) * (uy - 1.0) + f20 * (ux + 1.0) * (uy + 1.0) + f21 * (ux - 1.0) * (uy - 1.0) + f22 * (ux + 1.0) * (uy + 1.0) + f23 * (ux - 1.0) * (uy + 1.0) + f24 * (ux + 1.0) * (uy - 1.0) + f25 * (ux + 1.0) * (uy - 1.0) + f26 * (ux - 1.0) * (uy + 1.0) + f0 * ux * uy + f5 * ux * uy + f6 * ux * uy);
		T f_t5 = r2 * (f1 * uz * (ux - 1.0) + f2 * uz * (ux + 1.0) + f5 * ux * (uz - 1.0) + f6 * ux * (uz + 1.0) + f7 * uz * (ux - 1.0) + f8 * uz * (ux + 1.0) + f11 * ux * (uz - 1.0) + f12 * ux * (uz + 1.0) + f13 * uz * (ux - 1.0) + f14 * uz * (ux + 1.0) + f17 * ux * (uz + 1.0) + f18 * ux * (uz - 1.0) + f9 * (ux - 1.0) * (uz - 1.0) + f10 * (ux + 1.0) * (uz + 1.0) + f15 * (ux - 1.0) * (uz + 1.0) + f16 * (ux + 1.0) * (uz - 1.0) + f19 * (ux - 1.0) * (uz - 1.0) + f20 * (ux + 1.0) * (uz + 1.0) + f21 * (ux - 1.0) * (uz + 1.0) + f22 * (ux + 1.0) * (uz - 1.0) + f23 * (ux - 1.0) * (uz - 1.0) + f24 * (ux + 1.0) * (uz + 1.0) + f25 * (ux + 1.0) * (uz - 1.0) + f26 * (ux - 1.0) * (uz + 1.0) + f0 * ux * uz + f3 * ux * uz + f4 * ux * uz);
		T f_t6 = r2 * (f3 * uz * (uy - 1.0) + f4 * uz * (uy + 1.0) + f5 * uy * (uz - 1.0) + f6 * uy * (uz + 1.0) + f7 * uz * (uy - 1.0) + f8 * uz * (uy + 1.0) + f9 * uy * (uz - 1.0) + f10 * uy * (uz + 1.0) + f13 * uz * (uy + 1.0) + f14 * uz * (uy - 1.0) + f15 * uy * (uz + 1.0) + f16 * uy * (uz - 1.0) + f11 * (uy - 1.0) * (uz - 1.0) + f12 * (uy + 1.0) * (uz + 1.0) + f17 * (uy - 1.0) * (uz + 1.0) + f18 * (uy + 1.0) * (uz - 1.0) + f19 * (uy - 1.0) * (uz - 1.0) + f20 * (uy + 1.0) * (uz + 1.0) + f21 * (uy - 1.0) * (uz + 1.0) + f22 * (uy + 1.0) * (uz - 1.0) + f23 * (uy + 1.0) * (uz - 1.0) + f24 * (uy - 1.0) * (uz + 1.0) + f25 * (uy - 1.0) * (uz - 1.0) + f26 * (uy + 1.0) * (uz + 1.0) + f0 * uy * uz + f1 * uy * uz + f2 * uy * uz);
		T f_t7 = r2 * (f0 * (ux * ux - uy * uy) + f5 * (ux * ux - uy * uy) + f6 * (ux * ux - uy * uy) + f1 * (square(ux - 1.0) - uy * uy) + f2 * (square(ux + 1.0) - uy * uy) - f3 * (square(uy - 1.0) - ux * ux) - f4 * (square(uy + 1.0) - ux * ux) + f9 * (square(ux - 1.0) - uy * uy) + f10 * (square(ux + 1.0) - uy * uy) - f11 * (square(uy - 1.0) - ux * ux) - f12 * (square(uy + 1.0) - ux * ux) + f15 * (square(ux - 1.0) - uy * uy) + f16 * (square(ux + 1.0) - uy * uy) - f17 * (square(uy - 1.0) - ux * ux) - f18 * (square(uy + 1.0) - ux * ux) + f7 * (square(ux - 1.0) - square(uy - 1.0)) + f8 * (square(ux + 1.0) - square(uy + 1.0)) + f13 * (square(ux - 1.0) - square(uy + 1.0)) + f14 * (square(ux + 1.0) - square(uy - 1.0)) + f19 * (square(ux - 1.0) - square(uy - 1.0)) + f20 * (square(ux + 1.0) - square(uy + 1.0)) + f21 * (square(ux - 1.0) - square(uy - 1.0)) + f22 * (square(ux + 1.0) - square(uy + 1.0)) + f23 * (square(ux - 1.0) - square(uy + 1.0)) + f24 * (square(ux + 1.0) - square(uy - 1.0)) + f25 * (square(ux + 1.0) - square(uy - 1.0)) + f26 * (square(ux - 1.0) - square(uy + 1.0)));
		T f_t8 = r2 * (f0 * (ux * ux - uz * uz) + f3 * (ux * ux - uz * uz) + f4 * (ux * ux - uz * uz) + f1 * (square(ux - 1.0) - uz * uz) + f2 * (square(ux + 1.0) - uz * uz) - f5 * (square(uz - 1.0) - ux * ux) - f6 * (square(uz + 1.0) - ux * ux) + f7 * (square(ux - 1.0) - uz * uz) + f8 * (square(ux + 1.0) - uz * uz) - f11 * (square(uz - 1.0) - ux * ux) - f12 * (square(uz + 1.0) - ux * ux) + f13 * (square(ux - 1.0) - uz * uz) + f14 * (square(ux + 1.0) - uz * uz) - f17 * (square(uz + 1.0) - ux * ux) - f18 * (square(uz - 1.0) - ux * ux) + f9 * (square(ux - 1.0) - square(uz - 1.0)) + f10 * (square(ux + 1.0) - square(uz + 1.0)) + f15 * (square(ux - 1.0) - square(uz + 1.0)) + f16 * (square(ux + 1.0) - square(uz - 1.0)) + f19 * (square(ux - 1.0) - square(uz - 1.0)) + f20 * (square(ux + 1.0) - square(uz + 1.0)) + f21 * (square(ux - 1.0) - square(uz + 1.0)) + f22 * (square(ux + 1.0) - square(uz - 1.0)) + f23 * (square(ux - 1.0) - square(uz - 1.0)) + f24 * (square(ux + 1.0) - square(uz + 1.0)) + f25 * (square(ux + 1.0) - square(uz - 1.0)) + f26 * (square(ux - 1.0) - square(uz + 1.0)));
		T f_t9 = r3 * (-rho + f1 * (square(ux - 1.0) + uy * uy + uz * uz) + f2 * (square(ux + 1.0) + uy * uy + uz * uz) + f3 * (square(uy - 1.0) + ux * ux + uz * uz) + f4 * (square(uy + 1.0) + ux * ux + uz * uz) + f5 * (square(uz - 1.0) + ux * ux + uy * uy) + f6 * (square(uz + 1.0) + ux * ux + uy * uy) + f7 * (square(ux - 1.0) + square(uy - 1.0) + uz * uz) + f8 * (square(ux + 1.0) + square(uy + 1.0) + uz * uz) + f9 * (square(ux - 1.0) + square(uz - 1.0) + uy * uy) + f10 * (square(ux + 1.0) + square(uz + 1.0) + uy * uy) + f11 * (square(uy - 1.0) + square(uz - 1.0) + ux * ux) + f12 * (square(uy + 1.0) + square(uz + 1.0) + ux * ux) + f13 * (square(ux - 1.0) + square(uy + 1.0) + uz * uz) + f14 * (square(ux + 1.0) + square(uy - 1.0) + uz * uz) + f15 * (square(ux - 1.0) + square(uz + 1.0) + uy * uy) + f16 * (square(ux + 1.0) + square(uz - 1.0) + uy * uy) + f17 * (square(uy - 1.0) + square(uz + 1.0) + ux * ux) + f18 * (square(uy + 1.0) + square(uz - 1.0) + ux * ux) + f0 * (ux * ux + uy * uy + uz * uz) + f19 * (square(ux - 1.0) + square(uy - 1.0) + square(uz - 1.0)) + f20 * (square(ux + 1.0) + square(uy + 1.0) + square(uz + 1.0)) + f21 * (square(ux - 1.0) + square(uy - 1.0) + square(uz + 1.0)) + f22 * (square(ux + 1.0) + square(uy + 1.0) + square(uz - 1.0)) + f23 * (square(ux - 1.0) + square(uy + 1.0) + square(uz - 1.0)) + f24 * (square(ux + 1.0) + square(uy - 1.0) + square(uz + 1.0)) + f25 * (square(ux + 1.0) + square(uy - 1.0) + square(uz - 1.0)) + f26 * (square(ux - 1.0) + square(uy + 1.0) + square(uz + 1.0)));
		T f_t10 = -r4 * (f1 * ((uy * uy) * (ux - 1.0) + (uz * uz) * (ux - 1.0)) + f2 * ((uy * uy) * (ux + 1.0) + (uz * uz) * (ux + 1.0)) + f11 * (ux * square(uy - 1.0) + ux * square(uz - 1.0)) + f12 * (ux * square(uy + 1.0) + ux * square(uz + 1.0)) + f17 * (ux * square(uy - 1.0) + ux * square(uz + 1.0)) + f18 * (ux * square(uy + 1.0) + ux * square(uz - 1.0)) + f0 * (ux * (uy * uy) + ux * (uz * uz)) + f7 * ((uz * uz) * (ux - 1.0) + (ux - 1.0) * square(uy - 1.0)) + f8 * ((uz * uz) * (ux + 1.0) + (ux + 1.0) * square(uy + 1.0)) + f9 * ((uy * uy) * (ux - 1.0) + (ux - 1.0) * square(uz - 1.0)) + f10 * ((uy * uy) * (ux + 1.0) + (ux + 1.0) * square(uz + 1.0)) + f13 * ((uz * uz) * (ux - 1.0) + (ux - 1.0) * square(uy + 1.0)) + f14 * ((uz * uz) * (ux + 1.0) + (ux + 1.0) * square(uy - 1.0)) + f15 * ((uy * uy) * (ux - 1.0) + (ux - 1.0) * square(uz + 1.0)) + f16 * ((uy * uy) * (ux + 1.0) + (ux + 1.0) * square(uz - 1.0)) + f3 * (ux * square(uy - 1.0) + ux * (uz * uz)) + f4 * (ux * square(uy + 1.0) + ux * (uz * uz)) + f5 * (ux * square(uz - 1.0) + ux * (uy * uy)) + f6 * (ux * square(uz + 1.0) + ux * (uy * uy)) + f19 * ((ux - 1.0) * square(uy - 1.0) + (ux - 1.0) * square(uz - 1.0)) + f20 * ((ux + 1.0) * square(uy + 1.0) + (ux + 1.0) * square(uz + 1.0)) + f21 * ((ux - 1.0) * square(uy - 1.0) + (ux - 1.0) * square(uz + 1.0)) + f22 * ((ux + 1.0) * square(uy + 1.0) + (ux + 1.0) * square(uz - 1.0)) + f23 * ((ux - 1.0) * square(uy + 1.0) + (ux - 1.0) * square(uz - 1.0)) + f24 * ((ux + 1.0) * square(uy - 1.0) + (ux + 1.0) * square(uz + 1.0)) + f25 * ((ux + 1.0) * square(uy - 1.0) + (ux + 1.0) * square(uz - 1.0)) + f26 * ((ux - 1.0) * square(uy + 1.0) + (ux - 1.0) * square(uz + 1.0)));
		T f_t11 = -r4 * (f3 * ((ux * ux) * (uy - 1.0) + (uz * uz) * (uy - 1.0)) + f4 * ((ux * ux) * (uy + 1.0) + (uz * uz) * (uy + 1.0)) + f9 * (uy * square(ux - 1.0) + uy * square(uz - 1.0)) + f10 * (uy * square(ux + 1.0) + uy * square(uz + 1.0)) + f15 * (uy * square(ux - 1.0) + uy * square(uz + 1.0)) + f16 * (uy * square(ux + 1.0) + uy * square(uz - 1.0)) + f0 * ((ux * ux) * uy + uy * (uz * uz)) + f7 * ((uz * uz) * (uy - 1.0) + square(ux - 1.0) * (uy - 1.0)) + f8 * ((uz * uz) * (uy + 1.0) + square(ux + 1.0) * (uy + 1.0)) + f11 * ((ux * ux) * (uy - 1.0) + (uy - 1.0) * square(uz - 1.0)) + f12 * ((ux * ux) * (uy + 1.0) + (uy + 1.0) * square(uz + 1.0)) + f13 * ((uz * uz) * (uy + 1.0) + square(ux - 1.0) * (uy + 1.0)) + f14 * ((uz * uz) * (uy - 1.0) + square(ux + 1.0) * (uy - 1.0)) + f17 * ((ux * ux) * (uy - 1.0) + (uy - 1.0) * square(uz + 1.0)) + f18 * ((ux * ux) * (uy + 1.0) + (uy + 1.0) * square(uz - 1.0)) + f1 * (uy * square(ux - 1.0) + uy * (uz * uz)) + f2 * (uy * square(ux + 1.0) + uy * (uz * uz)) + f5 * (uy * square(uz - 1.0) + (ux * ux) * uy) + f6 * (uy * square(uz + 1.0) + (ux * ux) * uy) + f19 * (square(ux - 1.0) * (uy - 1.0) + (uy - 1.0) * square(uz - 1.0)) + f20 * (square(ux + 1.0) * (uy + 1.0) + (uy + 1.0) * square(uz + 1.0)) + f21 * (square(ux - 1.0) * (uy - 1.0) + (uy - 1.0) * square(uz + 1.0)) + f22 * (square(ux + 1.0) * (uy + 1.0) + (uy + 1.0) * square(uz - 1.0)) + f23 * (square(ux - 1.0) * (uy + 1.0) + (uy + 1.0) * square(uz - 1.0)) + f24 * (square(ux + 1.0) * (uy - 1.0) + (uy - 1.0) * square(uz + 1.0)) + f25 * (square(ux + 1.0) * (uy - 1.0) + (uy - 1.0) * square(uz - 1.0)) + f26 * (square(ux - 1.0) * (uy + 1.0) + (uy + 1.0) * square(uz + 1.0)));
		T f_t12 = -r4 * (f5 * ((ux * ux) * (uz - 1.0) + (uy * uy) * (uz - 1.0)) + f6 * ((ux * ux) * (uz + 1.0) + (uy * uy) * (uz + 1.0)) + f7 * (uz * square(ux - 1.0) + uz * square(uy - 1.0)) + f8 * (uz * square(ux + 1.0) + uz * square(uy + 1.0)) + f13 * (uz * square(ux - 1.0) + uz * square(uy + 1.0)) + f14 * (uz * square(ux + 1.0) + uz * square(uy - 1.0)) + f0 * ((ux * ux) * uz + (uy * uy) * uz) + f9 * ((uy * uy) * (uz - 1.0) + square(ux - 1.0) * (uz - 1.0)) + f10 * ((uy * uy) * (uz + 1.0) + square(ux + 1.0) * (uz + 1.0)) + f11 * ((ux * ux) * (uz - 1.0) + square(uy - 1.0) * (uz - 1.0)) + f12 * ((ux * ux) * (uz + 1.0) + square(uy + 1.0) * (uz + 1.0)) + f15 * ((uy * uy) * (uz + 1.0) + square(ux - 1.0) * (uz + 1.0)) + f16 * ((uy * uy) * (uz - 1.0) + square(ux + 1.0) * (uz - 1.0)) + f17 * ((ux * ux) * (uz + 1.0) + square(uy - 1.0) * (uz + 1.0)) + f18 * ((ux * ux) * (uz - 1.0) + square(uy + 1.0) * (uz - 1.0)) + f1 * (uz * square(ux - 1.0) + (uy * uy) * uz) + f2 * (uz * square(ux + 1.0) + (uy * uy) * uz) + f3 * (uz * square(uy - 1.0) + (ux * ux) * uz) + f4 * (uz * square(uy + 1.0) + (ux * ux) * uz) + f19 * (square(ux - 1.0) * (uz - 1.0) + square(uy - 1.0) * (uz - 1.0)) + f20 * (square(ux + 1.0) * (uz + 1.0) + square(uy + 1.0) * (uz + 1.0)) + f21 * (square(ux - 1.0) * (uz + 1.0) + square(uy - 1.0) * (uz + 1.0)) + f22 * (square(ux + 1.0) * (uz - 1.0) + square(uy + 1.0) * (uz - 1.0)) + f23 * (square(ux - 1.0) * (uz - 1.0) + square(uy + 1.0) * (uz - 1.0)) + f24 * (square(ux + 1.0) * (uz + 1.0) + square(uy - 1.0) * (uz + 1.0)) + f25 * (square(ux + 1.0) * (uz - 1.0) + square(uy - 1.0) * (uz - 1.0)) + f26 * (square(ux - 1.0) * (uz + 1.0) + square(uy + 1.0) * (uz + 1.0)));
		T f_t13 = -r4 * (f1 * ((uy * uy) * (ux - 1.0) - (uz * uz) * (ux - 1.0)) + f2 * ((uy * uy) * (ux + 1.0) - (uz * uz) * (ux + 1.0)) + f11 * (ux * square(uy - 1.0) - ux * square(uz - 1.0)) + f12 * (ux * square(uy + 1.0) - ux * square(uz + 1.0)) + f17 * (ux * square(uy - 1.0) - ux * square(uz + 1.0)) + f18 * (ux * square(uy + 1.0) - ux * square(uz - 1.0)) + f0 * (ux * (uy * uy) - ux * (uz * uz)) - f7 * ((uz * uz) * (ux - 1.0) - (ux - 1.0) * square(uy - 1.0)) - f8 * ((uz * uz) * (ux + 1.0) - (ux + 1.0) * square(uy + 1.0)) + f9 * ((uy * uy) * (ux - 1.0) - (ux - 1.0) * square(uz - 1.0)) + f10 * ((uy * uy) * (ux + 1.0) - (ux + 1.0) * square(uz + 1.0)) - f13 * ((uz * uz) * (ux - 1.0) - (ux - 1.0) * square(uy + 1.0)) - f14 * ((uz * uz) * (ux + 1.0) - (ux + 1.0) * square(uy - 1.0)) + f15 * ((uy * uy) * (ux - 1.0) - (ux - 1.0) * square(uz + 1.0)) + f16 * ((uy * uy) * (ux + 1.0) - (ux + 1.0) * square(uz - 1.0)) + f3 * (ux * square(uy - 1.0) - ux * (uz * uz)) + f4 * (ux * square(uy + 1.0) - ux * (uz * uz)) - f5 * (ux * square(uz - 1.0) - ux * (uy * uy)) - f6 * (ux * square(uz + 1.0) - ux * (uy * uy)) + f19 * ((ux - 1.0) * square(uy - 1.0) - (ux - 1.0) * square(uz - 1.0)) + f20 * ((ux + 1.0) * square(uy + 1.0) - (ux + 1.0) * square(uz + 1.0)) + f21 * ((ux - 1.0) * square(uy - 1.0) - (ux - 1.0) * square(uz + 1.0)) + f22 * ((ux + 1.0) * square(uy + 1.0) - (ux + 1.0) * square(uz - 1.0)) + f23 * ((ux - 1.0) * square(uy + 1.0) - (ux - 1.0) * square(uz - 1.0)) + f24 * ((ux + 1.0) * square(uy - 1.0) - (ux + 1.0) * square(uz + 1.0)) + f25 * ((ux + 1.0) * square(uy - 1.0) - (ux + 1.0) * square(uz - 1.0)) + f26 * ((ux - 1.0) * square(uy + 1.0) - (ux - 1.0) * square(uz + 1.0)));
		T f_t14 = -r4 * (f3 * ((ux * ux) * (uy - 1.0) - (uz * uz) * (uy - 1.0)) + f4 * ((ux * ux) * (uy + 1.0) - (uz * uz) * (uy + 1.0)) + f9 * (uy * square(ux - 1.0) - uy * square(uz - 1.0)) + f10 * (uy * square(ux + 1.0) - uy * square(uz + 1.0)) + f15 * (uy * square(ux - 1.0) - uy * square(uz + 1.0)) + f16 * (uy * square(ux + 1.0) - uy * square(uz - 1.0)) + f0 * ((ux * ux) * uy - uy * (uz * uz)) - f7 * ((uz * uz) * (uy - 1.0) - square(ux - 1.0) * (uy - 1.0)) - f8 * ((uz * uz) * (uy + 1.0) - square(ux + 1.0) * (uy + 1.0)) + f11 * ((ux * ux) * (uy - 1.0) - (uy - 1.0) * square(uz - 1.0)) + f12 * ((ux * ux) * (uy + 1.0) - (uy + 1.0) * square(uz + 1.0)) - f13 * ((uz * uz) * (uy + 1.0) - square(ux - 1.0) * (uy + 1.0)) - f14 * ((uz * uz) * (uy - 1.0) - square(ux + 1.0) * (uy - 1.0)) + f17 * ((ux * ux) * (uy - 1.0) - (uy - 1.0) * square(uz + 1.0)) + f18 * ((ux * ux) * (uy + 1.0) - (uy + 1.0) * square(uz - 1.0)) + f1 * (uy * square(ux - 1.0) - uy * (uz * uz)) + f2 * (uy * square(ux + 1.0) - uy * (uz * uz)) - f5 * (uy * square(uz - 1.0) - (ux * ux) * uy) - f6 * (uy * square(uz + 1.0) - (ux * ux) * uy) + f19 * (square(ux - 1.0) * (uy - 1.0) - (uy - 1.0) * square(uz - 1.0)) + f20 * (square(ux + 1.0) * (uy + 1.0) - (uy + 1.0) * square(uz + 1.0)) + f21 * (square(ux - 1.0) * (uy - 1.0) - (uy - 1.0) * square(uz + 1.0)) + f22 * (square(ux + 1.0) * (uy + 1.0) - (uy + 1.0) * square(uz - 1.0)) + f23 * (square(ux - 1.0) * (uy + 1.0) - (uy + 1.0) * square(uz - 1.0)) + f24 * (square(ux + 1.0) * (uy - 1.0) - (uy - 1.0) * square(uz + 1.0)) + f25 * (square(ux + 1.0) * (uy - 1.0) - (uy - 1.0) * square(uz - 1.0)) + f26 * (square(ux - 1.0) * (uy + 1.0) - (uy + 1.0) * square(uz + 1.0)));
		T f_t15 = -r4 * (f5 * ((ux * ux) * (uz - 1.0) - (uy * uy) * (uz - 1.0)) + f6 * ((ux * ux) * (uz + 1.0) - (uy * uy) * (uz + 1.0)) + f7 * (uz * square(ux - 1.0) - uz * square(uy - 1.0)) + f8 * (uz * square(ux + 1.0) - uz * square(uy + 1.0)) + f13 * (uz * square(ux - 1.0) - uz * square(uy + 1.0)) + f14 * (uz * square(ux + 1.0) - uz * square(uy - 1.0)) + f0 * ((ux * ux) * uz - (uy * uy) * uz) - f9 * ((uy * uy) * (uz - 1.0) - square(ux - 1.0) * (uz - 1.0)) - f10 * ((uy * uy) * (uz + 1.0) - square(ux + 1.0) * (uz + 1.0)) + f11 * ((ux * ux) * (uz - 1.0) - square(uy - 1.0) * (uz - 1.0)) + f12 * ((ux * ux) * (uz + 1.0) - square(uy + 1.0) * (uz + 1.0)) - f15 * ((uy * uy) * (uz + 1.0) - square(ux - 1.0) * (uz + 1.0)) - f16 * ((uy * uy) * (uz - 1.0) - square(ux + 1.0) * (uz - 1.0)) + f17 * ((ux * ux) * (uz + 1.0) - square(uy - 1.0) * (uz + 1.0)) + f18 * ((ux * ux) * (uz - 1.0) - square(uy + 1.0) * (uz - 1.0)) + f1 * (uz * square(ux - 1.0) - (uy * uy) * uz) + f2 * (uz * square(ux + 1.0) - (uy * uy) * uz) - f3 * (uz * square(uy - 1.0) - (ux * ux) * uz) - f4 * (uz * square(uy + 1.0) - (ux * ux) * uz) + f19 * (square(ux - 1.0) * (uz - 1.0) - square(uy - 1.0) * (uz - 1.0)) + f20 * (square(ux + 1.0) * (uz + 1.0) - square(uy + 1.0) * (uz + 1.0)) + f21 * (square(ux - 1.0) * (uz + 1.0) - square(uy - 1.0) * (uz + 1.0)) + f22 * (square(ux + 1.0) * (uz - 1.0) - square(uy + 1.0) * (uz - 1.0)) + f23 * (square(ux - 1.0) * (uz - 1.0) - square(uy + 1.0) * (uz - 1.0)) + f24 * (square(ux + 1.0) * (uz + 1.0) - square(uy - 1.0) * (uz + 1.0)) + f25 * (square(ux + 1.0) * (uz - 1.0) - square(uy - 1.0) * (uz - 1.0)) + f26 * (square(ux - 1.0) * (uz + 1.0) - square(uy + 1.0) * (uz + 1.0)));
		T f_t16 = -r4 * (f0 * ux * uy * uz + f19 * (ux - 1.0) * (uy - 1.0) * (uz - 1.0) + f20 * (ux + 1.0) * (uy + 1.0) * (uz + 1.0) + f21 * (ux - 1.0) * (uy - 1.0) * (uz + 1.0) + f22 * (ux + 1.0) * (uy + 1.0) * (uz - 1.0) + f23 * (ux - 1.0) * (uy + 1.0) * (uz - 1.0) + f24 * (ux + 1.0) * (uy - 1.0) * (uz + 1.0) + f25 * (ux + 1.0) * (uy - 1.0) * (uz - 1.0) + f26 * (ux - 1.0) * (uy + 1.0) * (uz + 1.0) + f1 * uy * uz * (ux - 1.0) + f2 * uy * uz * (ux + 1.0) + f3 * ux * uz * (uy - 1.0) + f4 * ux * uz * (uy + 1.0) + f5 * ux * uy * (uz - 1.0) + f6 * ux * uy * (uz + 1.0) + f7 * uz * (ux - 1.0) * (uy - 1.0) + f8 * uz * (ux + 1.0) * (uy + 1.0) + f9 * uy * (ux - 1.0) * (uz - 1.0) + f10 * uy * (ux + 1.0) * (uz + 1.0) + f11 * ux * (uy - 1.0) * (uz - 1.0) + f12 * ux * (uy + 1.0) * (uz + 1.0) + f13 * uz * (ux - 1.0) * (uy + 1.0) + f14 * uz * (ux + 1.0) * (uy - 1.0) + f15 * uy * (ux - 1.0) * (uz + 1.0) + f16 * uy * (ux + 1.0) * (uz - 1.0) + f17 * ux * (uy - 1.0) * (uz + 1.0) + f18 * ux * (uy + 1.0) * (uz - 1.0));
		T f_t17 = r5 * (rho * (-1.0 / 3.0) + f0 * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) + (uy * uy) * (uz * uz)) + f19 * (square(ux - 1.0) * square(uy - 1.0) + square(ux - 1.0) * square(uz - 1.0) + square(uy - 1.0) * square(uz - 1.0)) + f20 * (square(ux + 1.0) * square(uy + 1.0) + square(ux + 1.0) * square(uz + 1.0) + square(uy + 1.0) * square(uz + 1.0)) + f21 * (square(ux - 1.0) * square(uy - 1.0) + square(ux - 1.0) * square(uz + 1.0) + square(uy - 1.0) * square(uz + 1.0)) + f22 * (square(ux + 1.0) * square(uy + 1.0) + square(ux + 1.0) * square(uz - 1.0) + square(uy + 1.0) * square(uz - 1.0)) + f23 * (square(ux - 1.0) * square(uy + 1.0) + square(ux - 1.0) * square(uz - 1.0) + square(uy + 1.0) * square(uz - 1.0)) + f24 * (square(ux + 1.0) * square(uy - 1.0) + square(ux + 1.0) * square(uz + 1.0) + square(uy - 1.0) * square(uz + 1.0)) + f25 * (square(ux + 1.0) * square(uy - 1.0) + square(ux + 1.0) * square(uz - 1.0) + square(uy - 1.0) * square(uz - 1.0)) + f26 * (square(ux - 1.0) * square(uy + 1.0) + square(ux - 1.0) * square(uz + 1.0) + square(uy + 1.0) * square(uz + 1.0)) + f7 * ((uz * uz) * square(ux - 1.0) + (uz * uz) * square(uy - 1.0) + square(ux - 1.0) * square(uy - 1.0)) + f8 * ((uz * uz) * square(ux + 1.0) + (uz * uz) * square(uy + 1.0) + square(ux + 1.0) * square(uy + 1.0)) + f9 * ((uy * uy) * square(ux - 1.0) + (uy * uy) * square(uz - 1.0) + square(ux - 1.0) * square(uz - 1.0)) + f10 * ((uy * uy) * square(ux + 1.0) + (uy * uy) * square(uz + 1.0) + square(ux + 1.0) * square(uz + 1.0)) + f11 * ((ux * ux) * square(uy - 1.0) + (ux * ux) * square(uz - 1.0) + square(uy - 1.0) * square(uz - 1.0)) + f12 * ((ux * ux) * square(uy + 1.0) + (ux * ux) * square(uz + 1.0) + square(uy + 1.0) * square(uz + 1.0)) + f13 * ((uz * uz) * square(ux - 1.0) + (uz * uz) * square(uy + 1.0) + square(ux - 1.0) * square(uy + 1.0)) + f14 * ((uz * uz) * square(ux + 1.0) + (uz * uz) * square(uy - 1.0) + square(ux + 1.0) * square(uy - 1.0)) + f15 * ((uy * uy) * square(ux - 1.0) + (uy * uy) * square(uz + 1.0) + square(ux - 1.0) * square(uz + 1.0)) + f16 * ((uy * uy) * square(ux + 1.0) + (uy * uy) * square(uz - 1.0) + square(ux + 1.0) * square(uz - 1.0)) + f17 * ((ux * ux) * square(uy - 1.0) + (ux * ux) * square(uz + 1.0) + square(uy - 1.0) * square(uz + 1.0)) + f18 * ((ux * ux) * square(uy + 1.0) + (ux * ux) * square(uz - 1.0) + square(uy + 1.0) * square(uz - 1.0)) + f1 * ((uy * uy) * (uz * uz) + (uy * uy) * square(ux - 1.0) + (uz * uz) * square(ux - 1.0)) + f2 * ((uy * uy) * (uz * uz) + (uy * uy) * square(ux + 1.0) + (uz * uz) * square(ux + 1.0)) + f3 * ((ux * ux) * (uz * uz) + (ux * ux) * square(uy - 1.0) + (uz * uz) * square(uy - 1.0)) + f4 * ((ux * ux) * (uz * uz) + (ux * ux) * square(uy + 1.0) + (uz * uz) * square(uy + 1.0)) + f5 * ((ux * ux) * (uy * uy) + (ux * ux) * square(uz - 1.0) + (uy * uy) * square(uz - 1.0)) + f6 * ((ux * ux) * (uy * uy) + (ux * ux) * square(uz + 1.0) + (uy * uy) * square(uz + 1.0)));
		T f_t18 = r5 * (rho * (-1.0 / 9.0) + f0 * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) - (uy * uy) * (uz * uz)) + f19 * (square(ux - 1.0) * square(uy - 1.0) + square(ux - 1.0) * square(uz - 1.0) - square(uy - 1.0) * square(uz - 1.0)) + f20 * (square(ux + 1.0) * square(uy + 1.0) + square(ux + 1.0) * square(uz + 1.0) - square(uy + 1.0) * square(uz + 1.0)) + f21 * (square(ux - 1.0) * square(uy - 1.0) + square(ux - 1.0) * square(uz + 1.0) - square(uy - 1.0) * square(uz + 1.0)) + f22 * (square(ux + 1.0) * square(uy + 1.0) + square(ux + 1.0) * square(uz - 1.0) - square(uy + 1.0) * square(uz - 1.0)) + f23 * (square(ux - 1.0) * square(uy + 1.0) + square(ux - 1.0) * square(uz - 1.0) - square(uy + 1.0) * square(uz - 1.0)) + f24 * (square(ux + 1.0) * square(uy - 1.0) + square(ux + 1.0) * square(uz + 1.0) - square(uy - 1.0) * square(uz + 1.0)) + f25 * (square(ux + 1.0) * square(uy - 1.0) + square(ux + 1.0) * square(uz - 1.0) - square(uy - 1.0) * square(uz - 1.0)) + f26 * (square(ux - 1.0) * square(uy + 1.0) + square(ux - 1.0) * square(uz + 1.0) - square(uy + 1.0) * square(uz + 1.0)) + f7 * ((uz * uz) * square(ux - 1.0) - (uz * uz) * square(uy - 1.0) + square(ux - 1.0) * square(uy - 1.0)) + f8 * ((uz * uz) * square(ux + 1.0) - (uz * uz) * square(uy + 1.0) + square(ux + 1.0) * square(uy + 1.0)) + f9 * ((uy * uy) * square(ux - 1.0) - (uy * uy) * square(uz - 1.0) + square(ux - 1.0) * square(uz - 1.0)) + f10 * ((uy * uy) * square(ux + 1.0) - (uy * uy) * square(uz + 1.0) + square(ux + 1.0) * square(uz + 1.0)) + f11 * ((ux * ux) * square(uy - 1.0) + (ux * ux) * square(uz - 1.0) - square(uy - 1.0) * square(uz - 1.0)) + f12 * ((ux * ux) * square(uy + 1.0) + (ux * ux) * square(uz + 1.0) - square(uy + 1.0) * square(uz + 1.0)) + f13 * ((uz * uz) * square(ux - 1.0) - (uz * uz) * square(uy + 1.0) + square(ux - 1.0) * square(uy + 1.0)) + f14 * ((uz * uz) * square(ux + 1.0) - (uz * uz) * square(uy - 1.0) + square(ux + 1.0) * square(uy - 1.0)) + f15 * ((uy * uy) * square(ux - 1.0) - (uy * uy) * square(uz + 1.0) + square(ux - 1.0) * square(uz + 1.0)) + f16 * ((uy * uy) * square(ux + 1.0) - (uy * uy) * square(uz - 1.0) + square(ux + 1.0) * square(uz - 1.0)) + f17 * ((ux * ux) * square(uy - 1.0) + (ux * ux) * square(uz + 1.0) - square(uy - 1.0) * square(uz + 1.0)) + f18 * ((ux * ux) * square(uy + 1.0) + (ux * ux) * square(uz - 1.0) - square(uy + 1.0) * square(uz - 1.0)) + f1 * (-(uy * uy) * (uz * uz) + (uy * uy) * square(ux - 1.0) + (uz * uz) * square(ux - 1.0)) + f2 * (-(uy * uy) * (uz * uz) + (uy * uy) * square(ux + 1.0) + (uz * uz) * square(ux + 1.0)) + f3 * ((ux * ux) * (uz * uz) + (ux * ux) * square(uy - 1.0) - (uz * uz) * square(uy - 1.0)) + f4 * ((ux * ux) * (uz * uz) + (ux * ux) * square(uy + 1.0) - (uz * uz) * square(uy + 1.0)) + f5 * ((ux * ux) * (uy * uy) + (ux * ux) * square(uz - 1.0) - (uy * uy) * square(uz - 1.0)) + f6 * ((ux * ux) * (uy * uy) + (ux * ux) * square(uz + 1.0) - (uy * uy) * square(uz + 1.0)));
		T f_t19 = r5 * (f0 * ((ux * ux) * (uy * uy) - (ux * ux) * (uz * uz)) - f7 * ((uz * uz) * square(ux - 1.0) - square(ux - 1.0) * square(uy - 1.0)) - f8 * ((uz * uz) * square(ux + 1.0) - square(ux + 1.0) * square(uy + 1.0)) + f9 * ((uy * uy) * square(ux - 1.0) - square(ux - 1.0) * square(uz - 1.0)) + f10 * ((uy * uy) * square(ux + 1.0) - square(ux + 1.0) * square(uz + 1.0)) - f13 * ((uz * uz) * square(ux - 1.0) - square(ux - 1.0) * square(uy + 1.0)) - f14 * ((uz * uz) * square(ux + 1.0) - square(ux + 1.0) * square(uy - 1.0)) + f15 * ((uy * uy) * square(ux - 1.0) - square(ux - 1.0) * square(uz + 1.0)) + f16 * ((uy * uy) * square(ux + 1.0) - square(ux + 1.0) * square(uz - 1.0)) - f3 * ((ux * ux) * (uz * uz) - (ux * ux) * square(uy - 1.0)) - f4 * ((ux * ux) * (uz * uz) - (ux * ux) * square(uy + 1.0)) + f5 * ((ux * ux) * (uy * uy) - (ux * ux) * square(uz - 1.0)) + f6 * ((ux * ux) * (uy * uy) - (ux * ux) * square(uz + 1.0)) + f19 * (square(ux - 1.0) * square(uy - 1.0) - square(ux - 1.0) * square(uz - 1.0)) + f20 * (square(ux + 1.0) * square(uy + 1.0) - square(ux + 1.0) * square(uz + 1.0)) + f21 * (square(ux - 1.0) * square(uy - 1.0) - square(ux - 1.0) * square(uz + 1.0)) + f22 * (square(ux + 1.0) * square(uy + 1.0) - square(ux + 1.0) * square(uz - 1.0)) + f23 * (square(ux - 1.0) * square(uy + 1.0) - square(ux - 1.0) * square(uz - 1.0)) + f24 * (square(ux + 1.0) * square(uy - 1.0) - square(ux + 1.0) * square(uz + 1.0)) + f25 * (square(ux + 1.0) * square(uy - 1.0) - square(ux + 1.0) * square(uz - 1.0)) + f26 * (square(ux - 1.0) * square(uy + 1.0) - square(ux - 1.0) * square(uz + 1.0)) + f1 * ((uy * uy) * square(ux - 1.0) - (uz * uz) * square(ux - 1.0)) + f2 * ((uy * uy) * square(ux + 1.0) - (uz * uz) * square(ux + 1.0)) + f11 * ((ux * ux) * square(uy - 1.0) - (ux * ux) * square(uz - 1.0)) + f12 * ((ux * ux) * square(uy + 1.0) - (ux * ux) * square(uz + 1.0)) + f17 * ((ux * ux) * square(uy - 1.0) - (ux * ux) * square(uz + 1.0)) + f18 * ((ux * ux) * square(uy + 1.0) - (ux * ux) * square(uz - 1.0)));
		T f_t20 = r5 * (f7 * uz * square(ux - 1.0) * (uy - 1.0) + f8 * uz * square(ux + 1.0) * (uy + 1.0) + f9 * uy * square(ux - 1.0) * (uz - 1.0) + f10 * uy * square(ux + 1.0) * (uz + 1.0) + f11 * (ux * ux) * (uy - 1.0) * (uz - 1.0) + f12 * (ux * ux) * (uy + 1.0) * (uz + 1.0) + f13 * uz * square(ux - 1.0) * (uy + 1.0) + f14 * uz * square(ux + 1.0) * (uy - 1.0) + f15 * uy * square(ux - 1.0) * (uz + 1.0) + f16 * uy * square(ux + 1.0) * (uz - 1.0) + f17 * (ux * ux) * (uy - 1.0) * (uz + 1.0) + f18 * (ux * ux) * (uy + 1.0) * (uz - 1.0) + f0 * (ux * ux) * uy * uz + f19 * square(ux - 1.0) * (uy - 1.0) * (uz - 1.0) + f20 * square(ux + 1.0) * (uy + 1.0) * (uz + 1.0) + f21 * square(ux - 1.0) * (uy - 1.0) * (uz + 1.0) + f22 * square(ux + 1.0) * (uy + 1.0) * (uz - 1.0) + f23 * square(ux - 1.0) * (uy + 1.0) * (uz - 1.0) + f24 * square(ux + 1.0) * (uy - 1.0) * (uz + 1.0) + f25 * square(ux + 1.0) * (uy - 1.0) * (uz - 1.0) + f26 * square(ux - 1.0) * (uy + 1.0) * (uz + 1.0) + f1 * uy * uz * square(ux - 1.0) + f2 * uy * uz * square(ux + 1.0) + f3 * (ux * ux) * uz * (uy - 1.0) + f4 * (ux * ux) * uz * (uy + 1.0) + f5 * (ux * ux) * uy * (uz - 1.0) + f6 * (ux * ux) * uy * (uz + 1.0));
		T f_t21 = r5 * (f7 * uz * (ux - 1.0) * square(uy - 1.0) + f8 * uz * (ux + 1.0) * square(uy + 1.0) + f9 * (uy * uy) * (ux - 1.0) * (uz - 1.0) + f10 * (uy * uy) * (ux + 1.0) * (uz + 1.0) + f11 * ux * square(uy - 1.0) * (uz - 1.0) + f12 * ux * square(uy + 1.0) * (uz + 1.0) + f13 * uz * (ux - 1.0) * square(uy + 1.0) + f14 * uz * (ux + 1.0) * square(uy - 1.0) + f15 * (uy * uy) * (ux - 1.0) * (uz + 1.0) + f16 * (uy * uy) * (ux + 1.0) * (uz - 1.0) + f17 * ux * square(uy - 1.0) * (uz + 1.0) + f18 * ux * square(uy + 1.0) * (uz - 1.0) + f0 * ux * (uy * uy) * uz + f19 * (ux - 1.0) * square(uy - 1.0) * (uz - 1.0) + f20 * (ux + 1.0) * square(uy + 1.0) * (uz + 1.0) + f21 * (ux - 1.0) * square(uy - 1.0) * (uz + 1.0) + f22 * (ux + 1.0) * square(uy + 1.0) * (uz - 1.0) + f23 * (ux - 1.0) * square(uy + 1.0) * (uz - 1.0) + f24 * (ux + 1.0) * square(uy - 1.0) * (uz + 1.0) + f25 * (ux + 1.0) * square(uy - 1.0) * (uz - 1.0) + f26 * (ux - 1.0) * square(uy + 1.0) * (uz + 1.0) + f1 * (uy * uy) * uz * (ux - 1.0) + f2 * (uy * uy) * uz * (ux + 1.0) + f3 * ux * uz * square(uy - 1.0) + f4 * ux * uz * square(uy + 1.0) + f5 * ux * (uy * uy) * (uz - 1.0) + f6 * ux * (uy * uy) * (uz + 1.0));
		T f_t22 = r5 * (f7 * (uz * uz) * (ux - 1.0) * (uy - 1.0) + f8 * (uz * uz) * (ux + 1.0) * (uy + 1.0) + f9 * uy * (ux - 1.0) * square(uz - 1.0) + f10 * uy * (ux + 1.0) * square(uz + 1.0) + f11 * ux * (uy - 1.0) * square(uz - 1.0) + f12 * ux * (uy + 1.0) * square(uz + 1.0) + f13 * (uz * uz) * (ux - 1.0) * (uy + 1.0) + f14 * (uz * uz) * (ux + 1.0) * (uy - 1.0) + f15 * uy * (ux - 1.0) * square(uz + 1.0) + f16 * uy * (ux + 1.0) * square(uz - 1.0) + f17 * ux * (uy - 1.0) * square(uz + 1.0) + f18 * ux * (uy + 1.0) * square(uz - 1.0) + f0 * ux * uy * (uz * uz) + f19 * (ux - 1.0) * (uy - 1.0) * square(uz - 1.0) + f20 * (ux + 1.0) * (uy + 1.0) * square(uz + 1.0) + f21 * (ux - 1.0) * (uy - 1.0) * square(uz + 1.0) + f22 * (ux + 1.0) * (uy + 1.0) * square(uz - 1.0) + f23 * (ux - 1.0) * (uy + 1.0) * square(uz - 1.0) + f24 * (ux + 1.0) * (uy - 1.0) * square(uz + 1.0) + f25 * (ux + 1.0) * (uy - 1.0) * square(uz - 1.0) + f26 * (ux - 1.0) * (uy + 1.0) * square(uz + 1.0) + f1 * uy * (uz * uz) * (ux - 1.0) + f2 * uy * (uz * uz) * (ux + 1.0) + f3 * ux * (uz * uz) * (uy - 1.0) + f4 * ux * (uz * uz) * (uy + 1.0) + f5 * ux * uy * square(uz - 1.0) + f6 * ux * uy * square(uz + 1.0));
		T f_t23 = -r6 * (f19 * (ux - 1.0) * square(uy - 1.0) * square(uz - 1.0) + f20 * (ux + 1.0) * square(uy + 1.0) * square(uz + 1.0) + f21 * (ux - 1.0) * square(uy - 1.0) * square(uz + 1.0) + f22 * (ux + 1.0) * square(uy + 1.0) * square(uz - 1.0) + f23 * (ux - 1.0) * square(uy + 1.0) * square(uz - 1.0) + f24 * (ux + 1.0) * square(uy - 1.0) * square(uz + 1.0) + f25 * (ux + 1.0) * square(uy - 1.0) * square(uz - 1.0) + f26 * (ux - 1.0) * square(uy + 1.0) * square(uz + 1.0) + f1 * (uy * uy) * (uz * uz) * (ux - 1.0) + f2 * (uy * uy) * (uz * uz) * (ux + 1.0) + f3 * ux * (uz * uz) * square(uy - 1.0) + f4 * ux * (uz * uz) * square(uy + 1.0) + f5 * ux * (uy * uy) * square(uz - 1.0) + f6 * ux * (uy * uy) * square(uz + 1.0) + f7 * (uz * uz) * (ux - 1.0) * square(uy - 1.0) + f8 * (uz * uz) * (ux + 1.0) * square(uy + 1.0) + f9 * (uy * uy) * (ux - 1.0) * square(uz - 1.0) + f10 * (uy * uy) * (ux + 1.0) * square(uz + 1.0) + f11 * ux * square(uy - 1.0) * square(uz - 1.0) + f12 * ux * square(uy + 1.0) * square(uz + 1.0) + f13 * (uz * uz) * (ux - 1.0) * square(uy + 1.0) + f14 * (uz * uz) * (ux + 1.0) * square(uy - 1.0) + f15 * (uy * uy) * (ux - 1.0) * square(uz + 1.0) + f16 * (uy * uy) * (ux + 1.0) * square(uz - 1.0) + f17 * ux * square(uy - 1.0) * square(uz + 1.0) + f18 * ux * square(uy + 1.0) * square(uz - 1.0) + f0 * ux * (uy * uy) * (uz * uz));
		T f_t24 = -r6 * (f19 * square(ux - 1.0) * (uy - 1.0) * square(uz - 1.0) + f20 * square(ux + 1.0) * (uy + 1.0) * square(uz + 1.0) + f21 * square(ux - 1.0) * (uy - 1.0) * square(uz + 1.0) + f22 * square(ux + 1.0) * (uy + 1.0) * square(uz - 1.0) + f23 * square(ux - 1.0) * (uy + 1.0) * square(uz - 1.0) + f24 * square(ux + 1.0) * (uy - 1.0) * square(uz + 1.0) + f25 * square(ux + 1.0) * (uy - 1.0) * square(uz - 1.0) + f26 * square(ux - 1.0) * (uy + 1.0) * square(uz + 1.0) + f1 * uy * (uz * uz) * square(ux - 1.0) + f2 * uy * (uz * uz) * square(ux + 1.0) + f3 * (ux * ux) * (uz * uz) * (uy - 1.0) + f4 * (ux * ux) * (uz * uz) * (uy + 1.0) + f5 * (ux * ux) * uy * square(uz - 1.0) + f6 * (ux * ux) * uy * square(uz + 1.0) + f7 * (uz * uz) * square(ux - 1.0) * (uy - 1.0) + f8 * (uz * uz) * square(ux + 1.0) * (uy + 1.0) + f9 * uy * square(ux - 1.0) * square(uz - 1.0) + f10 * uy * square(ux + 1.0) * square(uz + 1.0) + f11 * (ux * ux) * (uy - 1.0) * square(uz - 1.0) + f12 * (ux * ux) * (uy + 1.0) * square(uz + 1.0) + f13 * (uz * uz) * square(ux - 1.0) * (uy + 1.0) + f14 * (uz * uz) * square(ux + 1.0) * (uy - 1.0) + f15 * uy * square(ux - 1.0) * square(uz + 1.0) + f16 * uy * square(ux + 1.0) * square(uz - 1.0) + f17 * (ux * ux) * (uy - 1.0) * square(uz + 1.0) + f18 * (ux * ux) * (uy + 1.0) * square(uz - 1.0) + f0 * (ux * ux) * uy * (uz * uz));
		T f_t25 = -r6 * (f19 * square(ux - 1.0) * square(uy - 1.0) * (uz - 1.0) + f20 * square(ux + 1.0) * square(uy + 1.0) * (uz + 1.0) + f21 * square(ux - 1.0) * square(uy - 1.0) * (uz + 1.0) + f22 * square(ux + 1.0) * square(uy + 1.0) * (uz - 1.0) + f23 * square(ux - 1.0) * square(uy + 1.0) * (uz - 1.0) + f24 * square(ux + 1.0) * square(uy - 1.0) * (uz + 1.0) + f25 * square(ux + 1.0) * square(uy - 1.0) * (uz - 1.0) + f26 * square(ux - 1.0) * square(uy + 1.0) * (uz + 1.0) + f1 * (uy * uy) * uz * square(ux - 1.0) + f2 * (uy * uy) * uz * square(ux + 1.0) + f3 * (ux * ux) * uz * square(uy - 1.0) + f4 * (ux * ux) * uz * square(uy + 1.0) + f5 * (ux * ux) * (uy * uy) * (uz - 1.0) + f6 * (ux * ux) * (uy * uy) * (uz + 1.0) + f7 * uz * square(ux - 1.0) * square(uy - 1.0) + f8 * uz * square(ux + 1.0) * square(uy + 1.0) + f9 * (uy * uy) * square(ux - 1.0) * (uz - 1.0) + f10 * (uy * uy) * square(ux + 1.0) * (uz + 1.0) + f11 * (ux * ux) * square(uy - 1.0) * (uz - 1.0) + f12 * (ux * ux) * square(uy + 1.0) * (uz + 1.0) + f13 * uz * square(ux - 1.0) * square(uy + 1.0) + f14 * uz * square(ux + 1.0) * square(uy - 1.0) + f15 * (uy * uy) * square(ux - 1.0) * (uz + 1.0) + f16 * (uy * uy) * square(ux + 1.0) * (uz - 1.0) + f17 * (ux * ux) * square(uy - 1.0) * (uz + 1.0) + f18 * (ux * ux) * square(uy + 1.0) * (uz - 1.0) + f0 * (ux * ux) * (uy * uy) * uz);
		T f_t26 = r7 * (rho * (-1.0 / 2.7E+1) + f7 * (uz * uz) * square(ux - 1.0) * square(uy - 1.0) + f8 * (uz * uz) * square(ux + 1.0) * square(uy + 1.0) + f9 * (uy * uy) * square(ux - 1.0) * square(uz - 1.0) + f10 * (uy * uy) * square(ux + 1.0) * square(uz + 1.0) + f11 * (ux * ux) * square(uy - 1.0) * square(uz - 1.0) + f12 * (ux * ux) * square(uy + 1.0) * square(uz + 1.0) + f13 * (uz * uz) * square(ux - 1.0) * square(uy + 1.0) + f14 * (uz * uz) * square(ux + 1.0) * square(uy - 1.0) + f15 * (uy * uy) * square(ux - 1.0) * square(uz + 1.0) + f16 * (uy * uy) * square(ux + 1.0) * square(uz - 1.0) + f17 * (ux * ux) * square(uy - 1.0) * square(uz + 1.0) + f18 * (ux * ux) * square(uy + 1.0) * square(uz - 1.0) + f0 * (ux * ux) * (uy * uy) * (uz * uz) + f19 * square(ux - 1.0) * square(uy - 1.0) * square(uz - 1.0) + f20 * square(ux + 1.0) * square(uy + 1.0) * square(uz + 1.0) + f21 * square(ux - 1.0) * square(uy - 1.0) * square(uz + 1.0) + f22 * square(ux + 1.0) * square(uy + 1.0) * square(uz - 1.0) + f23 * square(ux - 1.0) * square(uy + 1.0) * square(uz - 1.0) + f24 * square(ux + 1.0) * square(uy - 1.0) * square(uz + 1.0) + f25 * square(ux + 1.0) * square(uy - 1.0) * square(uz - 1.0) + f26 * square(ux - 1.0) * square(uy + 1.0) * square(uz + 1.0) + f1 * (uy * uy) * (uz * uz) * square(ux - 1.0) + f2 * (uy * uy) * (uz * uz) * square(ux + 1.0) + f3 * (ux * ux) * (uz * uz) * square(uy - 1.0) + f4 * (ux * ux) * (uz * uz) * square(uy + 1.0) + f5 * (ux * ux) * (uy * uy) * square(uz - 1.0) + f6 * (ux * ux) * (uy * uy) * square(uz + 1.0));

		T Omega_0 = f_t26 + f_t23 * ux * 2.0 + f_t24 * uy * 2.0 + f_t25 * uz * 2.0 + f_t10 * (ux * -2.0 + ux * (uy * uy) + ux * (uz * uz)) + f_t11 * (uy * -2.0 + (ux * ux) * uy + uy * (uz * uz)) + f_t12 * (uz * -2.0 + (ux * ux) * uz + (uy * uy) * uz) - f_t13 * (ux * (uy * uy) - ux * (uz * uz)) - f_t14 * ((ux * ux) * uy - uy * (uz * uz)) - f_t15 * ((ux * ux) * uz - (uy * uy) * uz) - f_t19 * ((uy * uy) / 2.0 - (uz * uz) / 2.0) - f_t4 * (ux * uy * 4.0 - ux * uy * (uz * uz) * 4.0) - f_t5 * (ux * uz * 4.0 - ux * (uy * uy) * uz * 4.0) - f_t6 * (uy * uz * 4.0 - (ux * ux) * uy * uz * 4.0) - f_t0 * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) + (uy * uy) * (uz * uz) - ux * ux - uy * uy - uz * uz - (ux * ux) * (uy * uy) * (uz * uz) + 1.0) + f_t17 * ((ux * ux) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0) + f_t18 * ((ux * ux) * (-1.0 / 2.0) + (uy * uy) / 4.0 + (uz * uz) / 4.0) + f_t1 * (ux * 2.0 - ux * (uy * uy) * 2.0 - ux * (uz * uz) * 2.0 + ux * (uy * uy) * (uz * uz) * 2.0) + f_t2 * (uy * 2.0 - (ux * ux) * uy * 2.0 - uy * (uz * uz) * 2.0 + (ux * ux) * uy * (uz * uz) * 2.0) + f_t3 * (uz * 2.0 - (ux * ux) * uz * 2.0 - (uy * uy) * uz * 2.0 + (ux * ux) * (uy * uy) * uz * 2.0) + f_t7 * (((ux * ux) * (uy * uy)) / 3.0 - (ux * ux) * (uz * uz) * (2.0 / 3.0) + ((uy * uy) * (uz * uz)) / 3.0 + (ux * ux) / 3.0 - (uy * uy) * (2.0 / 3.0) + (uz * uz) / 3.0) + f_t8 * ((ux * ux) * (uy * uy) * (-2.0 / 3.0) + ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 3.0 + (ux * ux) / 3.0 + (uy * uy) / 3.0 - (uz * uz) * (2.0 / 3.0)) + f_t9 * (((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 3.0 - (ux * ux) * (2.0 / 3.0) - (uy * uy) * (2.0 / 3.0) - (uz * uz) * (2.0 / 3.0) + 1.0) + f_t22 * ux * uy * 4.0 + f_t21 * ux * uz * 4.0 + f_t20 * uy * uz * 4.0 + f_t16 * ux * uy * uz * 8.0;
		T Omega_1 = f_t26 * (-1.0 / 2.0) - f_t6 * ((ux * ux) * uy * uz * 2.0 + ux * uy * uz * 2.0) - f_t7 * (ux / 6.0 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 3.0 + (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t8 * (ux / 6.0 - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 3.0 + (ux * (uz * uz)) / 6.0 + (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t9 * (ux * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t16 * (uy * uz * 2.0 + ux * uy * uz * 4.0) - f_t24 * uy - f_t25 * uz + f_t19 * ((uy * uy) / 4.0 - (uz * uz) / 4.0) - f_t17 * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0) + f_t18 * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 4.0) - f_t11 * (uy * (-1.0 / 2.0) + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) + f_t14 * (uy / 2.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) - f_t12 * (uz * (-1.0 / 2.0) + (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) + f_t15 * (uz / 2.0 + (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) - f_t23 * (ux + 1.0 / 2.0) + f_t4 * (uy + ux * uy * 2.0 - uy * (uz * uz) - ux * uy * (uz * uz) * 2.0) + f_t5 * (uz + ux * uz * 2.0 - (uy * uy) * uz - ux * (uy * uy) * uz * 2.0) - f_t10 * (-ux + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) + f_t2 * (ux * uy + (ux * ux) * uy - ux * uy * (uz * uz) - (ux * ux) * uy * (uz * uz)) + f_t3 * (ux * uz + (ux * ux) * uz - ux * (uy * uy) * uz - (ux * ux) * (uy * uy) * uz) - f_t1 * (ux + ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) - (uy * uy) / 2.0 - (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) + 1.0 / 2.0) + f_t13 * ((ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 - (uz * uz) / 4.0) - f_t22 * (uy + ux * uy * 2.0) - f_t21 * (uz + ux * uz * 2.0) - f_t0 * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t20 * uy * uz * 2.0;
		T Omega_2 = f_t26 * (-1.0 / 2.0) - f_t6 * ((ux * ux) * uy * uz * 2.0 - ux * uy * uz * 2.0) + f_t7 * (ux / 6.0 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 3.0 - ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 3.0 - (ux * ux) / 6.0 + (uy * uy) / 6.0 + (uz * uz) / 6.0 - 1.0 / 6.0) + f_t8 * (ux / 6.0 + ((ux * ux) * (uy * uy)) / 3.0 - ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 3.0 + (ux * (uz * uz)) / 6.0 - (ux * ux) / 6.0 + (uy * uy) / 6.0 + (uz * uz) / 6.0 - 1.0 / 6.0) - f_t9 * (ux / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + f_t16 * (uy * uz * 2.0 - ux * uy * uz * 4.0) - f_t24 * uy - f_t25 * uz + f_t10 * (ux - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) + f_t19 * ((uy * uy) / 4.0 - (uz * uz) / 4.0) - f_t17 * (ux * (-1.0 / 4.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0) - f_t18 * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0) + f_t11 * (uy / 2.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) + f_t14 * (uy / 2.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) + f_t12 * (uz / 2.0 + (ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) + f_t15 * (uz / 2.0 - (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) - f_t23 * (ux - 1.0 / 2.0) - f_t4 * (uy - ux * uy * 2.0 - uy * (uz * uz) + ux * uy * (uz * uz) * 2.0) - f_t5 * (uz - ux * uz * 2.0 - (uy * uy) * uz + ux * (uy * uy) * uz * 2.0) - f_t2 * (ux * uy - (ux * ux) * uy - ux * uy * (uz * uz) + (ux * ux) * uy * (uz * uz)) - f_t3 * (ux * uz - (ux * ux) * uz - ux * (uy * uy) * uz + (ux * ux) * (uy * uy) * uz) - f_t1 * (ux - ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) + (uy * uy) / 2.0 + (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) - 1.0 / 2.0) + f_t13 * ((ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 - (uy * uy) / 4.0 + (uz * uz) / 4.0) + f_t22 * (uy - ux * uy * 2.0) + f_t21 * (uz - ux * uz * 2.0) + f_t0 * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 - (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t20 * uy * uz * 2.0;
		T Omega_3 = f_t26 * (-1.0 / 2.0) - f_t5 * (ux * (uy * uy) * uz * 2.0 + ux * uy * uz * 2.0) - f_t7 * (uy * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uy) / 6.0 + (uy * (uz * uz)) / 6.0 + (ux * ux) / 3.0 - (uy * uy) / 3.0 + (uz * uz) / 3.0 - 1.0 / 3.0) - f_t8 * (uy / 6.0 - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 3.0 + (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 + (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t9 * (uy * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uy) / 6.0 + (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t16 * (ux * uz * 2.0 + ux * uy * uz * 4.0) - f_t23 * ux - f_t25 * uz + f_t19 * (uy / 4.0 + (uy * uy) / 4.0 - (uz * uz) / 4.0 + 1.0 / 4.0) - f_t18 * (uy / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uy / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) - f_t10 * (ux * (-1.0 / 2.0) + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) + f_t13 * (ux / 2.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) - f_t12 * (uz * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) - f_t15 * (uz / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) - f_t24 * (uy + 1.0 / 2.0) + f_t4 * (ux + ux * uy * 2.0 - ux * (uz * uz) - ux * uy * (uz * uz) * 2.0) + f_t6 * (uz + uy * uz * 2.0 - (ux * ux) * uz - (ux * ux) * uy * uz * 2.0) - f_t11 * (-uy + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) + f_t1 * (ux * uy + ux * (uy * uy) - ux * uy * (uz * uz) - ux * (uy * uy) * (uz * uz)) + f_t3 * (uy * uz + (uy * uy) * uz - (ux * ux) * uy * uz - (ux * ux) * (uy * uy) * uz) - f_t2 * (uy + ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) - (ux * ux) / 2.0 - (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) + 1.0 / 2.0) + f_t14 * (((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 - (uz * uz) / 4.0) - f_t22 * (ux + ux * uy * 2.0) - f_t20 * (uz + uy * uz * 2.0) - f_t0 * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t21 * ux * uz * 2.0;
		T Omega_4 = f_t26 * (-1.0 / 2.0) - f_t5 * (ux * (uy * uy) * uz * 2.0 - ux * uy * uz * 2.0) - f_t7 * (uy / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 6.0 - (uy * (uz * uz)) / 6.0 + (ux * ux) / 3.0 - (uy * uy) / 3.0 + (uz * uz) / 3.0 - 1.0 / 3.0) + f_t8 * (uy / 6.0 + ((ux * ux) * (uy * uy)) / 3.0 - ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 3.0 + (uy * (uz * uz)) / 6.0 + (ux * ux) / 6.0 - (uy * uy) / 6.0 + (uz * uz) / 6.0 - 1.0 / 6.0) - f_t9 * (uy / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 6.0 - (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + f_t16 * (ux * uz * 2.0 - ux * uy * uz * 4.0) - f_t23 * ux - f_t25 * uz + f_t11 * (uy - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) - f_t19 * (uy / 4.0 - (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 4.0) - f_t18 * (uy * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uy * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) + f_t10 * (ux / 2.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) + f_t13 * (ux / 2.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) + f_t12 * (uz / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) - f_t15 * (uz / 2.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) - f_t24 * (uy - 1.0 / 2.0) - f_t4 * (ux - ux * uy * 2.0 - ux * (uz * uz) + ux * uy * (uz * uz) * 2.0) - f_t6 * (uz - uy * uz * 2.0 - (ux * ux) * uz + (ux * ux) * uy * uz * 2.0) - f_t1 * (ux * uy - ux * (uy * uy) - ux * uy * (uz * uz) + ux * (uy * uy) * (uz * uz)) - f_t3 * (uy * uz - (uy * uy) * uz - (ux * ux) * uy * uz + (ux * ux) * (uy * uy) * uz) - f_t2 * (uy - ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) + (ux * ux) / 2.0 + (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) - 1.0 / 2.0) + f_t14 * (((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 - (ux * ux) / 4.0 + (uz * uz) / 4.0) + f_t22 * (ux - ux * uy * 2.0) + f_t20 * (uz - uy * uz * 2.0) + f_t0 * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 - (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t21 * ux * uz * 2.0;
		T Omega_5 = f_t26 * (-1.0 / 2.0) - f_t4 * (ux * uy * (uz * uz) * 2.0 + ux * uy * uz * 2.0) - f_t8 * (uz * (-1.0 / 3.0) - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 6.0 + (ux * ux) / 3.0 + (uy * uy) / 3.0 - (uz * uz) / 3.0 - 1.0 / 3.0) - f_t7 * (uz / 6.0 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 3.0 + ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 + (uz * uz) / 6.0 + 1.0 / 6.0) - f_t9 * (uz * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) - f_t16 * (ux * uy * 2.0 + ux * uy * uz * 4.0) - f_t23 * ux - f_t24 * uy - f_t19 * (uz / 4.0 - (uy * uy) / 4.0 + (uz * uz) / 4.0 + 1.0 / 4.0) - f_t18 * (uz / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uz / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) - f_t10 * (ux * (-1.0 / 2.0) + (ux * uz) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) - f_t13 * (ux / 2.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) - f_t11 * (uy * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) - f_t14 * (uy / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) - f_t25 * (uz + 1.0 / 2.0) + f_t5 * (ux + ux * uz * 2.0 - ux * (uy * uy) - ux * (uy * uy) * uz * 2.0) + f_t6 * (uy + uy * uz * 2.0 - (ux * ux) * uy - (ux * ux) * uy * uz * 2.0) - f_t12 * (-uz + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) + f_t1 * (ux * uz + ux * (uz * uz) - ux * (uy * uy) * uz - ux * (uy * uy) * (uz * uz)) + f_t2 * (uy * uz + uy * (uz * uz) - (ux * ux) * uy * uz - (ux * ux) * uy * (uz * uz)) - f_t3 * (uz + ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz - (ux * ux) / 2.0 - (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz + 1.0 / 2.0) + f_t15 * (((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0) - f_t21 * (ux + ux * uz * 2.0) - f_t20 * (uy + uy * uz * 2.0) - f_t0 * (uz / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t22 * ux * uy * 2.0;
		T Omega_6 = f_t26 * (-1.0 / 2.0) - f_t4 * (ux * uy * (uz * uz) * 2.0 - ux * uy * uz * 2.0) - f_t8 * (uz / 3.0 - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 6.0 + (ux * ux) / 3.0 + (uy * uy) / 3.0 - (uz * uz) / 3.0 - 1.0 / 3.0) + f_t7 * (uz / 6.0 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 3.0 - ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 3.0 + ((uy * uy) * uz) / 6.0 + (ux * ux) / 6.0 + (uy * uy) / 6.0 - (uz * uz) / 6.0 - 1.0 / 6.0) - f_t9 * (uz / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) + f_t16 * (ux * uy * 2.0 - ux * uy * uz * 4.0) - f_t23 * ux - f_t24 * uy + f_t12 * (uz - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) + f_t19 * (uz / 4.0 + (uy * uy) / 4.0 - (uz * uz) / 4.0 - 1.0 / 4.0) - f_t18 * (uz * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uz * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) + f_t10 * (ux / 2.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) - f_t13 * (ux / 2.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) + f_t11 * (uy / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) - f_t14 * (uy / 2.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) - f_t25 * (uz - 1.0 / 2.0) - f_t5 * (ux - ux * uz * 2.0 - ux * (uy * uy) + ux * (uy * uy) * uz * 2.0) - f_t6 * (uy - uy * uz * 2.0 - (ux * ux) * uy + (ux * ux) * uy * uz * 2.0) - f_t1 * (ux * uz - ux * (uz * uz) - ux * (uy * uy) * uz + ux * (uy * uy) * (uz * uz)) - f_t2 * (uy * uz - uy * (uz * uz) - (ux * ux) * uy * uz + (ux * ux) * uy * (uz * uz)) - f_t3 * (uz - ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz + (ux * ux) / 2.0 + (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz - 1.0 / 2.0) + f_t15 * (((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0) + f_t21 * (ux - ux * uz * 2.0) + f_t20 * (uy - uy * uz * 2.0) + f_t0 * (uz / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t22 * ux * uy * 2.0;
		T Omega_7 = f_t26 / 4.0 + f_t12 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) - f_t15 * ((ux * uz) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) + f_t22 * (ux / 2.0 + uy / 2.0 + ux * uy + 1.0 / 4.0) + (f_t25 * uz) / 2.0 + f_t21 * (uz / 2.0 + ux * uz) + f_t20 * (uz / 2.0 + uy * uz) + f_t7 * (ux / 6.0 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 6.0 - (uy * uy) / 1.2E+1) - f_t8 * (ux / 1.2E+1 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 + ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) + f_t9 * (ux * (-1.0 / 1.2E+1) - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t16 * (uz / 2.0 + ux * uz + uy * uz + ux * uy * uz * 2.0) - f_t19 * (uy / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) - f_t2 * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t3 * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (ux / 2.0 + uy / 2.0 + ux * uy - (ux * (uz * uz)) / 2.0 - (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 - ux * uy * (uz * uz) + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) + f_t10 * (ux * (-1.0 / 4.0) + uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t11 * (ux / 8.0 - uy / 4.0 + (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t13 * (ux / 4.0 + uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t14 * (ux / 8.0 + uy / 4.0 + (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t6 * ((ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) * uy * uz + ux * uy * uz) + f_t5 * ((uy * uz) / 2.0 + ((uy * uy) * uz) / 2.0 + ux * (uy * uy) * uz + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t24 * (uy / 2.0 + 1.0 / 4.0);
		T Omega_8 = f_t26 / 4.0 - f_t12 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) + f_t15 * ((ux * uz) / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) - f_t22 * (ux / 2.0 + uy / 2.0 - ux * uy - 1.0 / 4.0) + (f_t25 * uz) / 2.0 - f_t21 * (uz / 2.0 - ux * uz) - f_t20 * (uz / 2.0 - uy * uz) - f_t7 * (ux / 6.0 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 6.0 + (uy * uy) / 1.2E+1) + f_t8 * (ux / 1.2E+1 + uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 + ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t9 * (ux / 1.2E+1 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t16 * (uz / 2.0 - ux * uz - uy * uz + ux * uy * uz * 2.0) + f_t19 * (uy / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t17 * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 - uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t2 * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t3 * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (ux / 2.0 + uy / 2.0 - ux * uy - (ux * (uz * uz)) / 2.0 - (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 + ux * uy * (uz * uz) - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 - uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t11 * (ux / 8.0 - uy / 4.0 - (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t13 * (ux / 4.0 + uy / 8.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t14 * (ux / 8.0 + uy / 4.0 - (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t6 * ((ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) * uy * uz - ux * uy * uz) + f_t5 * ((uy * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + ux * (uy * uy) * uz - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t24 * (uy / 2.0 - 1.0 / 4.0);
		T Omega_9 = f_t26 / 4.0 + f_t11 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) - f_t14 * ((ux * uy) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) + f_t21 * (ux / 2.0 + uz / 2.0 + ux * uz + 1.0 / 4.0) + (f_t24 * uy) / 2.0 + f_t22 * (uy / 2.0 + ux * uy) + f_t20 * (uy / 2.0 + uy * uz) - f_t7 * (ux / 1.2E+1 + uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) + f_t8 * (ux / 6.0 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 6.0 - (uz * uz) / 1.2E+1) + f_t9 * (ux * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (uy / 2.0 + ux * uy + uy * uz + ux * uy * uz * 2.0) + f_t19 * (uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) - f_t3 * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t1 * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t2 * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (ux / 2.0 + uz / 2.0 + ux * uz - (ux * (uy * uy)) / 2.0 - ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 - ux * (uy * uy) * uz + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) + f_t10 * (ux * (-1.0 / 4.0) + uz / 8.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (ux / 8.0 - uz / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t13 * (ux / 4.0 + uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t15 * (ux / 8.0 + uz / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t6 * ((ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (ux * ux) * uy * uz + ux * uy * uz) + f_t4 * ((uy * uz) / 2.0 + (uy * (uz * uz)) / 2.0 + ux * uy * (uz * uz) + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
		T Omega_10 = f_t26 / 4.0 - f_t11 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) + f_t14 * ((ux * uy) / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) - f_t21 * (ux / 2.0 + uz / 2.0 - ux * uz - 1.0 / 4.0) + (f_t24 * uy) / 2.0 - f_t22 * (uy / 2.0 - ux * uy) - f_t20 * (uy / 2.0 - uy * uz) + f_t7 * (ux / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t8 * (ux / 6.0 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (ux / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (uy / 2.0 - ux * uy - uy * uz + ux * uy * uz * 2.0) - f_t19 * (uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) - f_t17 * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t3 * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t1 * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t2 * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (ux / 2.0 + uz / 2.0 - ux * uz - (ux * (uy * uy)) / 2.0 - ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 + ux * (uy * uy) * uz - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 - uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (ux / 8.0 - uz / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t13 * (ux / 4.0 + uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) - f_t15 * (ux / 8.0 + uz / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t6 * ((ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 + (ux * ux) * uy * uz - ux * uy * uz) + f_t4 * ((uy * uz) / 2.0 - (uy * (uz * uz)) / 2.0 + ux * uy * (uz * uz) - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
		T Omega_11 = f_t26 / 4.0 + f_t10 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) - f_t13 * ((ux * uy) / 4.0 - (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) + f_t20 * (uy / 2.0 + uz / 2.0 + uy * uz + 1.0 / 4.0) - f_t19 * (uy / 8.0 - uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 + f_t22 * (ux / 2.0 + ux * uy) + f_t21 * (ux / 2.0 + ux * uz) + f_t7 * (uy * (-1.0 / 1.2E+1) + uz / 6.0 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 6.0 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 + (uz * uz) / 6.0) + f_t8 * (uy / 6.0 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 6.0 - (uz * uz) / 1.2E+1) + f_t9 * (uy * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (ux / 2.0 + ux * uy + ux * uz + ux * uy * uz * 2.0) + f_t17 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0) + f_t18 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0) - f_t3 * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t2 * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t6 * (uy / 2.0 + uz / 2.0 + uy * uz - ((ux * ux) * uy) / 2.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 - (ux * ux) * uy * uz + 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) + f_t11 * (uy * (-1.0 / 4.0) + uz / 8.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (uy / 8.0 - uz / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t14 * (uy / 4.0 + uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t15 * (uy / 8.0 + uz / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 + 1.0 / 8.0) + f_t5 * ((ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + ux * (uy * uy) * uz + ux * uy * uz) + f_t4 * ((ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 + ux * uy * (uz * uz) + ux * uy * uz) + f_t24 * (uy / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
		T Omega_12 = f_t26 / 4.0 - f_t10 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) + f_t13 * ((ux * uy) / 4.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) - f_t20 * (uy / 2.0 + uz / 2.0 - uy * uz - 1.0 / 4.0) + f_t19 * (uy / 8.0 - uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 - f_t22 * (ux / 2.0 - ux * uy) - f_t21 * (ux / 2.0 - ux * uz) + f_t7 * (uy / 1.2E+1 - uz / 6.0 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 6.0 - (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 + (uz * uz) / 6.0) - f_t8 * (uy / 6.0 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (uy / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (ux / 2.0 - ux * uy - ux * uz + ux * uy * uz * 2.0) - f_t17 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 8.0) - f_t18 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 - 1.0 / 8.0) - f_t3 * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t2 * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t6 * (uy / 2.0 + uz / 2.0 - uy * uz - ((ux * ux) * uy) / 2.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 + (ux * ux) * uy * uz - 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t11 * (uy / 4.0 - uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (uy / 8.0 - uz / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t14 * (uy / 4.0 + uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) + f_t15 * (uy / 8.0 + uz / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 - 1.0 / 8.0) + f_t5 * ((ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 + ux * (uy * uy) * uz - ux * uy * uz) + f_t4 * ((ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 + ux * uy * (uz * uz) - ux * uy * uz) + f_t24 * (uy / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
		T Omega_13 = f_t26 / 4.0 + f_t12 * ((ux * uz) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) - f_t15 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) - f_t22 * (ux / 2.0 - uy / 2.0 - ux * uy + 1.0 / 4.0) + (f_t25 * uz) / 2.0 + f_t21 * (uz / 2.0 + ux * uz) - f_t20 * (uz / 2.0 - uy * uz) + f_t7 * (ux / 6.0 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 - (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 6.0 - (uy * uy) / 1.2E+1) - f_t8 * (ux / 1.2E+1 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 - ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) - f_t9 * (ux / 1.2E+1 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) - f_t16 * (uz / 2.0 + ux * uz - uy * uz - ux * uy * uz * 2.0) + f_t19 * (uy / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t17 * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t2 * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t3 * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (ux / 2.0 - uy / 2.0 - ux * uy - (ux * (uz * uz)) / 2.0 + (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 + ux * uy * (uz * uz) + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t11 * (ux / 8.0 + uy / 4.0 - (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t13 * (ux / 4.0 - uy / 8.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t14 * (ux / 8.0 - uy / 4.0 - (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t6 * ((ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) * uy * uz - ux * uy * uz) - f_t5 * ((uy * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - ux * (uy * uy) * uz + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t24 * (uy / 2.0 - 1.0 / 4.0);
		T Omega_14 = f_t26 / 4.0 + f_t12 * (ux * uz * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) + f_t15 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) + f_t22 * (ux / 2.0 - uy / 2.0 + ux * uy - 1.0 / 4.0) + (f_t25 * uz) / 2.0 - f_t21 * (uz / 2.0 - ux * uz) + f_t20 * (uz / 2.0 + uy * uz) - f_t7 * (ux / 6.0 + uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 6.0 + (uy * uy) / 1.2E+1) + f_t8 * (ux / 1.2E+1 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 - ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t9 * (ux / 1.2E+1 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) - f_t16 * (uz / 2.0 - ux * uz + uy * uz - ux * uy * uz * 2.0) - f_t19 * (uy / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux * (-1.0 / 8.0) + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t2 * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t3 * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (ux / 2.0 - uy / 2.0 + ux * uy - (ux * (uz * uz)) / 2.0 + (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 - ux * uy * (uz * uz) - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uy / 8.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t11 * (ux / 8.0 + uy / 4.0 + (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t13 * (ux / 4.0 - uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t14 * (ux / 8.0 - uy / 4.0 + (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t6 * ((ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) * uy * uz + ux * uy * uz) - f_t5 * ((uy * uz) / 2.0 + ((uy * uy) * uz) / 2.0 - ux * (uy * uy) * uz - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t24 * (uy / 2.0 + 1.0 / 4.0);
		T Omega_15 = f_t26 / 4.0 + f_t11 * ((ux * uy) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) - f_t14 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) - f_t21 * (ux / 2.0 - uz / 2.0 - ux * uz + 1.0 / 4.0) + (f_t24 * uy) / 2.0 + f_t22 * (uy / 2.0 + ux * uy) - f_t20 * (uy / 2.0 - uy * uz) - f_t7 * (ux / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) + f_t8 * (ux / 6.0 + uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 6.0 - (uz * uz) / 1.2E+1) - f_t9 * (ux / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) - f_t16 * (uy / 2.0 + ux * uy - uy * uz - ux * uy * uz * 2.0) - f_t19 * (uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) + f_t17 * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t3 * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t1 * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t2 * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (ux / 2.0 - uz / 2.0 - ux * uz - (ux * (uy * uy)) / 2.0 + ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 + ux * (uy * uy) * uz + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t12 * (ux / 8.0 + uz / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t13 * (ux / 4.0 - uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t15 * (ux / 8.0 - uz / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) - f_t6 * ((ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * ux) * uy * uz - ux * uy * uz) - f_t4 * ((uy * uz) / 2.0 - (uy * (uz * uz)) / 2.0 - ux * uy * (uz * uz) + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
		T Omega_16 = f_t26 / 4.0 + f_t11 * (ux * uy * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) + f_t14 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) + f_t21 * (ux / 2.0 - uz / 2.0 + ux * uz - 1.0 / 4.0) + (f_t24 * uy) / 2.0 - f_t22 * (uy / 2.0 - ux * uy) + f_t20 * (uy / 2.0 + uy * uz) + f_t7 * (ux / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t8 * (ux / 6.0 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (ux / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t16 * (uy / 2.0 - ux * uy + uy * uz - ux * uy * uz * 2.0) + f_t19 * (uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux * (-1.0 / 8.0) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t3 * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t1 * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t2 * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (ux / 2.0 - uz / 2.0 + ux * uz - (ux * (uy * uy)) / 2.0 + ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 - ux * (uy * uy) * uz - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t12 * (ux / 8.0 + uz / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t13 * (ux / 4.0 - uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) + f_t15 * (ux / 8.0 - uz / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) - f_t6 * ((ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * ux) * uy * uz + ux * uy * uz) - f_t4 * ((uy * uz) / 2.0 + (uy * (uz * uz)) / 2.0 - ux * uy * (uz * uz) - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
		T Omega_17 = f_t26 / 4.0 + f_t10 * ((ux * uy) / 4.0 - (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) - f_t13 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) - f_t20 * (uy / 2.0 - uz / 2.0 - uy * uz + 1.0 / 4.0) - f_t19 * (uy / 8.0 + uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 + f_t22 * (ux / 2.0 + ux * uy) - f_t21 * (ux / 2.0 - ux * uz) - f_t7 * (uy / 1.2E+1 + uz / 6.0 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 6.0 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 1.2E+1 - (uz * uz) / 6.0) + f_t8 * (uy / 6.0 + uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 6.0 - (uz * uz) / 1.2E+1) - f_t9 * (uy / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 1.2E+1 + (uz * uz) / 1.2E+1) - f_t16 * (ux / 2.0 + ux * uy - ux * uz - ux * uy * uz * 2.0) + f_t17 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0) + f_t18 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0) + f_t3 * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t2 * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t6 * (uy / 2.0 - uz / 2.0 - uy * uz - ((ux * ux) * uy) / 2.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 + (ux * ux) * uy * uz + 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t11 * (uy / 4.0 + uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t12 * (uy / 8.0 + uz / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t14 * (uy / 4.0 - uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t15 * (uy / 8.0 - uz / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 + 1.0 / 8.0) - f_t5 * ((ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - ux * (uy * uy) * uz - ux * uy * uz) - f_t4 * ((ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ux * uy * (uz * uz) + ux * uy * uz) + f_t24 * (uy / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
		T Omega_18 = f_t26 / 4.0 + f_t10 * (ux * uy * (-1.0 / 4.0) + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) + f_t13 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) + f_t20 * (uy / 2.0 - uz / 2.0 + uy * uz - 1.0 / 4.0) + f_t19 * (uy / 8.0 + uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 - f_t22 * (ux / 2.0 - ux * uy) + f_t21 * (ux / 2.0 + ux * uz) + f_t7 * (uy / 1.2E+1 + uz / 6.0 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 6.0 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 + (uz * uz) / 6.0) - f_t8 * (uy / 6.0 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (uy / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t16 * (ux / 2.0 - ux * uy + ux * uz - ux * uy * uz * 2.0) + f_t17 * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0) + f_t18 * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0) + f_t3 * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t2 * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t6 * (uy / 2.0 - uz / 2.0 + uy * uz - ((ux * ux) * uy) / 2.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 - (ux * ux) * uy * uz - 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t11 * (uy / 4.0 + uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t12 * (uy / 8.0 + uz / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t14 * (uy / 4.0 - uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) - f_t15 * (uy / 8.0 - uz / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 - 1.0 / 8.0) - f_t5 * ((ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - ux * (uy * uy) * uz + ux * uy * uz) - f_t4 * ((ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ux * uy * (uz * uz) - ux * uy * uz) + f_t24 * (uy / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
		T Omega_19 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 + uy / 4.0 + (ux * uy) / 2.0 + 1.0 / 8.0) - f_t21 * (ux / 4.0 + uz / 4.0 + (ux * uz) / 2.0 + 1.0 / 8.0) - f_t20 * (uy / 4.0 + uz / 4.0 + (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t8 * ((ux * ux) * (uy * uy) * (-1.0 / 1.2E+1) + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 + uy / 4.0 + uz / 4.0 + (ux * uy) / 2.0 + (ux * uz) / 2.0 + (uy * uz) / 2.0 + ux * uy * uz + 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 + (ux * uy) / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 + (ux * uy) / 4.0 + (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 + (ux * uz) / 4.0 + (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
		T Omega_20 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 + uy / 4.0 - (ux * uy) / 2.0 - 1.0 / 8.0) + f_t21 * (ux / 4.0 + uz / 4.0 - (ux * uz) / 2.0 - 1.0 / 8.0) + f_t20 * (uy / 4.0 + uz / 4.0 - (uy * uz) / 2.0 - 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t17 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 + uy / 4.0 + uz / 4.0 - (ux * uy) / 2.0 - (ux * uz) / 2.0 - (uy * uz) / 2.0 + ux * uy * uz - 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 - (ux * uy) / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 - (ux * uy) / 4.0 - (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 - (ux * uz) / 4.0 - (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);
		T Omega_21 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 + uy / 4.0 + (ux * uy) / 2.0 + 1.0 / 8.0) + f_t21 * (ux / 4.0 - uz / 4.0 - (ux * uz) / 2.0 + 1.0 / 8.0) + f_t20 * (uy / 4.0 - uz / 4.0 - (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t18 * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 + uy / 4.0 - uz / 4.0 + (ux * uy) / 2.0 - (ux * uz) / 2.0 - (uy * uz) / 2.0 - ux * uy * uz + 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 + (ux * uy) / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 + (ux * uy) / 4.0 - (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 + (ux * uz) / 4.0 + (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);
		T Omega_22 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 + uy / 4.0 - (ux * uy) / 2.0 - 1.0 / 8.0) - f_t21 * (ux / 4.0 - uz / 4.0 + (ux * uz) / 2.0 - 1.0 / 8.0) - f_t20 * (uy / 4.0 - uz / 4.0 + (uy * uz) / 2.0 - 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t17 * (ux * (-1.0 / 1.6E+1) - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 + uy / 4.0 - uz / 4.0 - (ux * uy) / 2.0 + (ux * uz) / 2.0 + (uy * uz) / 2.0 - ux * uy * uz - 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 - (ux * uy) / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 - (ux * uy) / 4.0 + (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 - (ux * uz) / 4.0 - (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
		T Omega_23 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 - uy / 4.0 - (ux * uy) / 2.0 + 1.0 / 8.0) - f_t21 * (ux / 4.0 + uz / 4.0 + (ux * uz) / 2.0 + 1.0 / 8.0) - f_t20 * (uy / 4.0 - uz / 4.0 + (uy * uz) / 2.0 - 1.0 / 8.0) + f_t7 * ((ux * ux) * (uy * uy) * (-1.0 / 2.4E+1) + ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t8 * ((ux * ux) * (uy * uy) * (-1.0 / 1.2E+1) + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 - uy / 4.0 + uz / 4.0 - (ux * uy) / 2.0 + (ux * uz) / 2.0 - (uy * uz) / 2.0 - ux * uy * uz + 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 - (ux * uy) / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 + (ux * uy) / 4.0 + (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 + (ux * uz) / 4.0 - (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
		T Omega_24 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 - uy / 4.0 + (ux * uy) / 2.0 - 1.0 / 8.0) + f_t21 * (ux / 4.0 + uz / 4.0 - (ux * uz) / 2.0 - 1.0 / 8.0) + f_t20 * (uy / 4.0 - uz / 4.0 - (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 1.2E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 1.2E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t10 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 - uy / 4.0 + uz / 4.0 + (ux * uy) / 2.0 - (ux * uz) / 2.0 + (uy * uz) / 2.0 - ux * uy * uz - 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 + (ux * uy) / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 - (ux * uy) / 4.0 - (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 - (ux * uz) / 4.0 + (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);
		T Omega_25 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 - uy / 4.0 + (ux * uy) / 2.0 - 1.0 / 8.0) - f_t21 * (ux / 4.0 - uz / 4.0 + (ux * uz) / 2.0 - 1.0 / 8.0) - f_t20 * (uy / 4.0 + uz / 4.0 + (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t8 * ((ux * ux) * (uy * uy) * (-1.0 / 1.2E+1) + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t11 * (ux * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t12 * (ux * (-1.0 / 1.6E+1) + uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 - uy / 4.0 - uz / 4.0 + (ux * uy) / 2.0 + (ux * uz) / 2.0 - (uy * uz) / 2.0 + ux * uy * uz - 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 + (ux * uy) / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 - (ux * uy) / 4.0 + (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 - (ux * uz) / 4.0 + (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
		T Omega_26 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 - uy / 4.0 - (ux * uy) / 2.0 + 1.0 / 8.0) + f_t21 * (ux / 4.0 - uz / 4.0 - (ux * uz) / 2.0 + 1.0 / 8.0) + f_t20 * (uy / 4.0 + uz / 4.0 - (uy * uz) / 2.0 - 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t11 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t12 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 - uy / 4.0 - uz / 4.0 - (ux * uy) / 2.0 - (ux * uz) / 2.0 + (uy * uz) / 2.0 + ux * uy * uz + 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 - (ux * uy) / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 + (ux * uy) / 4.0 - (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 + (ux * uz) / 4.0 - (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);

		T G_0 = Fx * ux * (r6 / 2.0 - 1.0) * (2.0 / 9.0) + Fy * uy * (r6 / 2.0 - 1.0) * (2.0 / 9.0) + Fz * uz * (r6 / 2.0 - 1.0) * (2.0 / 9.0) + Fx * (r4 / 2.0 - 1.0) * (ux * -2.0 + ux * (uy * uy) + ux * (uz * uz)) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (uy * -2.0 + (ux * ux) * uy + uy * (uz * uz)) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (uz * -2.0 + (ux * ux) * uz + (uy * uy) * uz) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * 2.0 - ux * (uy * uy) * 2.0 - ux * (uz * uz) * 2.0 + ux * (uy * uy) * (uz * uz) * 2.0) + Fy * (r1 / 2.0 - 1.0) * (uy * 2.0 - (ux * ux) * uy * 2.0 - uy * (uz * uz) * 2.0 + (ux * ux) * uy * (uz * uz) * 2.0) + Fz * (r1 / 2.0 - 1.0) * (uz * 2.0 - (ux * ux) * uz * 2.0 - (uy * uy) * uz * 2.0 + (ux * ux) * (uy * uy) * uz * 2.0);
		T G_1 = Fy * uy * (r6 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r6 / 2.0 - 1.0)) / 9.0 - Fy * (r4 / 2.0 - 1.0) * (uy * (-1.0 / 2.0) + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (uz * (-1.0 / 2.0) + (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fx * (r6 / 2.0 - 1.0) * (ux + 1.0 / 2.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (-ux + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fy * (r1 / 2.0 - 1.0) * (ux * uy + (ux * ux) * uy - ux * uy * (uz * uz) - (ux * ux) * uy * (uz * uz)) + Fz * (r1 / 2.0 - 1.0) * (ux * uz + (ux * ux) * uz - ux * (uy * uy) * uz - (ux * ux) * (uy * uy) * uz) - Fx * (r1 / 2.0 - 1.0) * (ux + ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) - (uy * uy) / 2.0 - (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) + 1.0 / 2.0);
		T G_2 = Fy * uy * (r6 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r6 / 2.0 - 1.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * (ux - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (uy / 2.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (uz / 2.0 + (ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fx * (r6 / 2.0 - 1.0) * (ux - 1.0 / 2.0)) / 9.0 - Fy * (r1 / 2.0 - 1.0) * (ux * uy - (ux * ux) * uy - ux * uy * (uz * uz) + (ux * ux) * uy * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (ux * uz - (ux * ux) * uz - ux * (uy * uy) * uz + (ux * ux) * (uy * uy) * uz) - Fx * (r1 / 2.0 - 1.0) * (ux - ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) + (uy * uy) / 2.0 + (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) - 1.0 / 2.0);
		T G_3 = Fx * ux * (r6 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r6 / 2.0 - 1.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (ux * (-1.0 / 2.0) + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (uz * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fy * (r6 / 2.0 - 1.0) * (uy + 1.0 / 2.0)) / 9.0 - Fy * (r4 / 2.0 - 1.0) * (-uy + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * uy + ux * (uy * uy) - ux * uy * (uz * uz) - ux * (uy * uy) * (uz * uz)) + Fz * (r1 / 2.0 - 1.0) * (uy * uz + (uy * uy) * uz - (ux * ux) * uy * uz - (ux * ux) * (uy * uy) * uz) - Fy * (r1 / 2.0 - 1.0) * (uy + ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) - (ux * ux) / 2.0 - (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) + 1.0 / 2.0);
		T G_4 = Fx * ux * (r6 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r6 / 2.0 - 1.0)) / 9.0 + Fy * (r4 / 2.0 - 1.0) * (uy - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r4 / 2.0 - 1.0) * (ux / 2.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (uz / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fy * (r6 / 2.0 - 1.0) * (uy - 1.0 / 2.0)) / 9.0 - Fx * (r1 / 2.0 - 1.0) * (ux * uy - ux * (uy * uy) - ux * uy * (uz * uz) + ux * (uy * uy) * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (uy * uz - (uy * uy) * uz - (ux * ux) * uy * uz + (ux * ux) * (uy * uy) * uz) - Fy * (r1 / 2.0 - 1.0) * (uy - ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) + (ux * ux) / 2.0 + (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) - 1.0 / 2.0);
		T G_5 = Fx * ux * (r6 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fy * uy * (r6 / 2.0 - 1.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (ux * (-1.0 / 2.0) + (ux * uz) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (uy * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) - (Fz * (r6 / 2.0 - 1.0) * (uz + 1.0 / 2.0)) / 9.0 - Fz * (r4 / 2.0 - 1.0) * (-uz + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * uz + ux * (uz * uz) - ux * (uy * uy) * uz - ux * (uy * uy) * (uz * uz)) + Fy * (r1 / 2.0 - 1.0) * (uy * uz + uy * (uz * uz) - (ux * ux) * uy * uz - (ux * ux) * uy * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (uz + ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz - (ux * ux) / 2.0 - (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz + 1.0 / 2.0);
		T G_6 = Fx * ux * (r6 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fy * uy * (r6 / 2.0 - 1.0)) / 9.0 + Fz * (r4 / 2.0 - 1.0) * (uz - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r4 / 2.0 - 1.0) * (ux / 2.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (uy / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) - (Fz * (r6 / 2.0 - 1.0) * (uz - 1.0 / 2.0)) / 9.0 - Fx * (r1 / 2.0 - 1.0) * (ux * uz - ux * (uz * uz) - ux * (uy * uy) * uz + ux * (uy * uy) * (uz * uz)) - Fy * (r1 / 2.0 - 1.0) * (uy * uz - uy * (uz * uz) - (ux * ux) * uy * uz + (ux * ux) * uy * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (uz - ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz + (ux * ux) / 2.0 + (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz - 1.0 / 2.0);
		T G_7 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + Fz * (r4 / 2.0 - 1.0) * ((ux * uz) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r6 / 2.0 - 1.0)) / 1.8E+1 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) + Fx * (r4 / 2.0 - 1.0) * (ux * (-1.0 / 4.0) + uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (ux / 8.0 - uy / 4.0 + (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
		T G_8 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 - Fz * (r4 / 2.0 - 1.0) * ((ux * uz) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r6 / 2.0 - 1.0)) / 1.8E+1 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r4 / 2.0 - 1.0) * (ux / 4.0 - uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (ux / 8.0 - uy / 4.0 - (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
		T G_9 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fy * (r4 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r6 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) + Fx * (r4 / 2.0 - 1.0) * (ux * (-1.0 / 4.0) + uz / 8.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (ux / 8.0 - uz / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
		T G_10 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 - Fy * (r4 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r6 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r4 / 2.0 - 1.0) * (ux / 4.0 - uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (ux / 8.0 - uz / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
		T G_11 = (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r6 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) + Fy * (r4 / 2.0 - 1.0) * (uy * (-1.0 / 4.0) + uz / 8.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (uy / 8.0 - uz / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
		T G_12 = (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r6 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fy * (r4 / 2.0 - 1.0) * (uy / 4.0 - uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (uy / 8.0 - uz / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
		T G_13 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 + Fz * (r4 / 2.0 - 1.0) * ((ux * uz) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r6 / 2.0 - 1.0)) / 1.8E+1 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r4 / 2.0 - 1.0) * (ux / 4.0 + uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux / 8.0 + uy / 4.0 - (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
		T G_14 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + Fz * (r4 / 2.0 - 1.0) * (ux * uz * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r6 / 2.0 - 1.0)) / 1.8E+1 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - Fx * (r4 / 2.0 - 1.0) * (ux / 4.0 + uy / 8.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux / 8.0 + uy / 4.0 + (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
		T G_15 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 + Fy * (r4 / 2.0 - 1.0) * ((ux * uy) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r6 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r4 / 2.0 - 1.0) * (ux / 4.0 + uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux / 8.0 + uz / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
		T G_16 = (Fx * (r6 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fy * (r4 / 2.0 - 1.0) * (ux * uy * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r6 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - Fx * (r4 / 2.0 - 1.0) * (ux / 4.0 + uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux / 8.0 + uz / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
		T G_17 = (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * ((ux * uy) / 4.0 - (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r6 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fy * (r4 / 2.0 - 1.0) * (uy / 4.0 + uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (uy / 8.0 + uz / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
		T G_18 = (Fy * (r6 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r6 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * (ux * uy * (-1.0 / 4.0) + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r6 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - Fy * (r4 / 2.0 - 1.0) * (uy / 4.0 + uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (uy / 8.0 + uz / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
		T G_19 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);
		T G_20 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
		T G_21 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
		T G_22 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);
		T G_23 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 - Fx * (r4 / 2.0 - 1.0) * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
		T G_24 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);
		T G_25 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r4 / 2.0 - 1.0) * (ux * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r4 / 2.0 - 1.0) * (ux * (-1.0 / 1.6E+1) + uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
		T G_26 = Fx * (r6 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r6 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r6 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 + Fx * (r4 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fy * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fz * (r4 / 2.0 - 1.0) * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);


		f_star[dataLayout(27, alpha, pos, 0)] = f0 + Omega_0 + G_0;
		f_star[dataLayout(27, alpha, pos, 1)] = f1 + Omega_1 + G_1;
		f_star[dataLayout(27, alpha, pos, 2)] = f2 + Omega_2 + G_2;
		f_star[dataLayout(27, alpha, pos, 3)] = f3 + Omega_3 + G_3;
		f_star[dataLayout(27, alpha, pos, 4)] = f4 + Omega_4 + G_4;
		f_star[dataLayout(27, alpha, pos, 5)] = f5 + Omega_5 + G_5;
		f_star[dataLayout(27, alpha, pos, 6)] = f6 + Omega_6 + G_6;
		f_star[dataLayout(27, alpha, pos, 7)] = f7 + Omega_7 + G_7;
		f_star[dataLayout(27, alpha, pos, 8)] = f8 + Omega_8 + G_8;
		f_star[dataLayout(27, alpha, pos, 9)] = f9 + Omega_9 + G_9;
		f_star[dataLayout(27, alpha, pos, 10)] = f10 + Omega_10 + G_10;
		f_star[dataLayout(27, alpha, pos, 11)] = f11 + Omega_11 + G_11;
		f_star[dataLayout(27, alpha, pos, 12)] = f12 + Omega_12 + G_12;
		f_star[dataLayout(27, alpha, pos, 13)] = f13 + Omega_13 + G_13;
		f_star[dataLayout(27, alpha, pos, 14)] = f14 + Omega_14 + G_14;
		f_star[dataLayout(27, alpha, pos, 15)] = f15 + Omega_15 + G_15;
		f_star[dataLayout(27, alpha, pos, 16)] = f16 + Omega_16 + G_16;
		f_star[dataLayout(27, alpha, pos, 17)] = f17 + Omega_17 + G_17;
		f_star[dataLayout(27, alpha, pos, 18)] = f18 + Omega_18 + G_18;
		f_star[dataLayout(27, alpha, pos, 19)] = f19 + Omega_19 + G_19;
		f_star[dataLayout(27, alpha, pos, 20)] = f20 + Omega_20 + G_20;
		f_star[dataLayout(27, alpha, pos, 21)] = f21 + Omega_21 + G_21;
		f_star[dataLayout(27, alpha, pos, 22)] = f22 + Omega_22 + G_22;
		f_star[dataLayout(27, alpha, pos, 23)] = f23 + Omega_23 + G_23;
		f_star[dataLayout(27, alpha, pos, 24)] = f24 + Omega_24 + G_24;
		f_star[dataLayout(27, alpha, pos, 25)] = f25 + Omega_25 + G_25;
		f_star[dataLayout(27, alpha, pos, 26)] = f26 + Omega_26 + G_26;
	}

}

template<typename T, int D, int Q>
__global__ void updateMoments_kernel(T* rho_L, vec<T, D>* u_L, T *secondMom_L, T* uMag, T *zerothMomSum, T *firstMomSum, T *secondMomSum, vec<T, D>* F_ext_L, T* f)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int alpha = sd_dev<T, D, Q>.alpha;

	if (pos < sd_dev<T, D, Q>.maxNodeCount)
	{
		CustomVecLength<T, D> vecLength;
		CustomAtomicMax<T> atomMax;

		rho_L[pos] = 0;
		u_L[pos] = {};
		vec<T, D* D> pi_mom{};

		for (int i = 0; i < Q; ++i)
		{
			rho_L[pos] += f[dataLayout(Q, alpha, pos, i)];
			u_L[pos] += sd_dev<T, D, Q>.c[i] * f[dataLayout(Q, alpha, pos, i)];

			for(int m=0; m < D; ++m)
				for (int n = 0; n < D; ++n)
				{
					pi_mom[m * D + n] += sd_dev<T, D, Q>.c[i][m] * sd_dev<T, D, Q>.c[i][n] * f[dataLayout(Q, alpha, pos, i)];
				}

		}

		u_L[pos] = (u_L[pos] + (T)0.5 * F_ext_L[pos]) / rho_L[pos];

		if (sd_dev<T, D, Q>.localRelaxation)
		{
			CustomAtomicAdd<T> atomicAdd;
			CustomVecLength<T, D*D> vecLength_2;
			atomicAdd.AtomicAdd(zerothMomSum, rho_L[pos]);
			atomicAdd.AtomicAdd(firstMomSum, vecLength.length(rho_L[pos] * u_L[pos]));

			T sMom = vecLength_2.length(pi_mom);
			secondMom_L[pos] = sMom;
			atomicAdd.AtomicAdd(secondMomSum, sMom);
		}

		atomMax.atomicMax(uMag, vecLength.length(u_L[pos]));
	}
}

template<typename T, int D, int Q>
__global__ void updateMoments_kernel(T* rho_L, vec<T, D>* u_unc_L, vec<T, D>* u_L, T* secondMom_L, T *uMag, T *zerothMomSum, T *firstMomSum, T *secondMomSum, vec<T, D>* F_ext_L, T* f)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int alpha = sd_dev<T, D, Q>.alpha;
	
	if (pos < sd_dev<T, D, Q>.maxNodeCount)
	{
		CustomVecLength<T, D> vecLength;
		CustomAtomicMax<T> atomMax;

		rho_L[pos] = 0;
		u_unc_L[pos] = {};
		vec<T, D* D> pi_mom{};

		for (int i = 0; i < Q; ++i)
		{
			rho_L[pos] += f[dataLayout(Q, alpha, pos, i)];
			u_unc_L[pos] += sd_dev<T, D, Q>.c[i] * f[dataLayout(Q, alpha, pos, i)];

			for (int m = 0; m < D; ++m)
				for (int n = 0; n < D; ++n)
				{
					pi_mom[m * D + n] += sd_dev<T, D, Q>.c[i][m] * sd_dev<T, D, Q>.c[i][n] * f[dataLayout(Q, alpha, pos, i)];
				}
		}

		u_unc_L[pos] /= rho_L[pos];
		u_L[pos] = u_unc_L[pos] + (T)0.5 * F_ext_L[pos] / rho_L[pos];
		F_ext_L[pos] = {};

		if (sd_dev<T, D, Q>.localRelaxation)
		{
			CustomAtomicAdd<T> atomicAdd;
			CustomVecLength<T, D*D> vecLength_2;
			atomicAdd.AtomicAdd(zerothMomSum, rho_L[pos]);
			atomicAdd.AtomicAdd(firstMomSum, vecLength.length(rho_L[pos] * u_L[pos]));
			T sMom = vecLength_2.length(pi_mom);
			secondMom_L[pos] = sMom;
			atomicAdd.AtomicAdd(secondMomSum, sMom);
		}

		atomMax.atomicMax(uMag, vecLength.length(u_L[pos]));
	}
}

template<typename T, int D, int Q>
__global__ void rescaleMoments_BGK_kernel(T* rho_L, vec<T,D> *u_L, vec<T, D>* u_unc_L, T* f, vec<T, D>* F_ext_L, T scale_u, T scale_F)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int alpha = sd_dev<T, D, Q>.alpha;

	if (pos < sd_dev<T, D, Q>.maxNodeCount)
	{
		vec<T, D> u_L_old = u_L[pos];
		u_L[pos] *= scale_u;
		u_unc_L[pos] *= scale_u;
		F_ext_L[pos] *= scale_F;

		for (int i = 0; i < Q; ++i)
		{
			T f_old = f[dataLayout(Q, alpha, pos, i)];
			T f_eq_old = sd_dev<T, D, Q>.w[i] * rho_L[pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L_old) / cs_sq<T> + dot(sd_dev<T, D, Q>.c[i], u_L_old) *
				dot(sd_dev<T, D, Q>.c[i], u_L_old) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L_old, u_L_old) / ((T)2.0 * cs_sq<T>));

			f[dataLayout(Q, alpha, pos, i)] = sd_dev<T, D, Q>.w[i] * rho_L[pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L[pos]) / cs_sq<T> +dot(sd_dev<T, D, Q>.c[i], u_L[pos]) *
				dot(sd_dev<T, D, Q>.c[i], u_L[pos]) / ((T)2.0 * cs_sq<T> * cs_sq<T>) - dot(u_L[pos], u_L[pos]) / ((T)2.0 * cs_sq<T>));

			f[dataLayout(Q, alpha, pos, i)] += ((f_eq_old - f_old) / f_eq_old) * f[dataLayout(Q, alpha, pos, i)];
		}

	}
}

template<typename T, int D, int Q>
__global__ void rescaleMoments_CM_kernel(T* rho_L, vec<T, D> *u_L, vec<T, D>* u_unc_L, T* f, vec<T,D>* F_ext_L, T scale_u, T scale_F)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int alpha = sd_dev<T, D, Q>.alpha;

	if (pos < sd_dev<T, D, Q>.maxNodeCount)
	{
		vec<T, D> u_L_old = u_L[pos];
		u_L[pos] *= scale_u;
		u_unc_L[pos] *= scale_u;
		F_ext_L[pos] *= scale_F;

		for (int i = 0; i < Q; ++i)
		{
			T f_old = f[dataLayout(Q, alpha, pos, i)];
			T f_eq_old = calc_f_eq_CM<T>(i, rho_L[pos], u_L_old);

			f[dataLayout(Q, alpha, pos, i)] = calc_f_eq_CM<T>(i, rho_L[pos], u_L[pos]);
			f[dataLayout(Q, alpha, pos, i)] += ((f_eq_old - f_old) / f_eq_old) * f[dataLayout(Q, alpha, pos, i)];
		}
	}
}

#pragma region solver_kernels_2D
//2D
//----------------------
template<typename T>
__device__ int calcPos2D(int idx, int x, int y)
{
	return (y + static_cast<int>(sd_dev<T, 2, 9>.c[idx][1])) * sd_dev<T, 2, 9>.gridDim_L[0] + x + static_cast<int>(sd_dev<T, 2, 9>.c[idx][0]);
}

template<typename T>
__global__ void streaming2D_kernel(T* f, T* f_star, vec<T, 2>* u_L, T* rho_L, T C_u, T C_p)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < sd_dev<T, 2, 9>.gridDim_L[0] && y < sd_dev<T, 2, 9>.gridDim_L[1])
	{
		int pos = y * sd_dev<T, 2, 9>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 2, 9>.alpha;

		T rhoVB_border = rho_L[pos];
		T rhoPB_border = 1.0;
		vec<T, 2> uwVB{ 0.0, 0.0 }, uwPB{ 0.0, 0.0 };
		bool velBound_defined = false;
		bool pressBound_defined = false;

		for (int vbIdx = 0; vbIdx < sd_dev<T, 2, 9>.velBound_Count; ++vbIdx)
		{
			auto& velBound = sd_dev<T, 2, 9>.velBounds[vbIdx];

			if (x == 0 && velBound.side == left)	//left
			{
				if (y < sd_dev<T, 2, 9>.gridDim_L[1])
				{
					velBound_defined = true;
					uwVB[0] = velBound.u_w / C_u;
					uwVB[1] = 0.0;
					break;
				}
			}
			if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 && velBound.side == right)	//right
			{
				if (y < sd_dev<T, 2, 9>.gridDim_L[1])
				{
					velBound_defined = true;
					uwVB[0] = -velBound.u_w / C_u;
					uwVB[1] = 0.0;
					break;
				}
			}
			if (y == 0 && velBound.side == bottom)	//bottom
			{
				if (x < sd_dev<T, 2, 9>.gridDim_L[0])
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = velBound.u_w / C_u;
					break;
				}
			}
			if (y == sd_dev<T, 2, 9>.gridDim_L[1] - 1 && velBound.side == top)	//top
			{
				if (x < sd_dev<T, 2, 9>.gridDim_L[0])
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = -velBound.u_w / C_u;
					break;
				}
			}
		}

		for (int pbIdx = 0; pbIdx < sd_dev<T, 2, 9>.pressBound_Count; ++pbIdx)
		{
			auto& pressBound = sd_dev<T, 2, 9>.pressBounds[pbIdx];

			if (x == 0 && pressBound.side == left)	//left
			{
				if (y < sd_dev<T, 2, 9>.gridDim_L[1])
				{
					pressBound_defined = true;
					int pos_next = y * sd_dev<T, 2, 9>.gridDim_L[0] + x + 1;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w / C_p) / cs_sq<T> +1.0;
					break;
				}
			}
			if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 && pressBound.side == right)	//right
			{
				if (y < sd_dev<T, 2, 9>.gridDim_L[1])
				{
					pressBound_defined = true;
					int pos_next = y * sd_dev<T, 2, 9>.gridDim_L[0] + x - 1;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w / C_p) / cs_sq<T> +1.0;
					break;
				}
			}
			if (y == 0 && pressBound.side == bottom)	//bottom
			{
				if (x < sd_dev<T, 2, 9>.gridDim_L[0])
				{
					pressBound_defined = true;
					int pos_next = (y + 1) * sd_dev<T, 2, 9>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w / C_p) / cs_sq<T> +1.0;
					break;
				}
			}
			if (y == sd_dev<T, 2, 9>.gridDim_L[1] - 1 && pressBound.side == top)	//top
			{
				if (x < sd_dev<T, 2, 9>.gridDim_L[0])
				{
					pressBound_defined = true;
					int pos_next = (y - 1) * sd_dev<T, 2, 9>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = (pressBound.dp_w / C_p) / cs_sq<T> +1.0;
					break;
				}
			}
		}

		int i = 0;
		f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 1;
		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 3)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 3)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 3)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 2;
		if (y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 4)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 4)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 4)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 3;				//L
		if (x == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 1)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 1)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 1)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 4;
		if (y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 2)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 2)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 2)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 5;
		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 || y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 7)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 7)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 7)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 6;
		if (x == 0 || y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 8)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 8)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 8)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 7;						//L
		if (x == 0 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 5)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 5)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 5)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];

		i = 8;
		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, 6)] = calc_f_with_velBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, 6)] = calc_f_with_presBound<T, 2, 9>(i, f_star[dataLayout(9, alpha, pos, i)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, 6)] = f_star[dataLayout(9, alpha, pos, i)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i, x, y), i)] = f_star[dataLayout(9, alpha, pos, i)];
	}
}

template<typename T>
__global__ void streaming2D_kernel_L(T* f, T* f_star, vec<T, 2>* u_L, T* rho_L)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < sd_dev<T, 2, 9>.gridDim_L[0] && y < sd_dev<T, 2, 9>.gridDim_L[1])
	{
		int pos = y * sd_dev<T, 2, 9>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 2, 9>.alpha;

		T rhoVB_border = rho_L[pos];
		T rhoPB_border = 1.0;
		vec<T, 2> uwVB{ 0.0, 0.0 }, uwPB{ 0.0, 0.0 };
		bool velBound_defined = false;
		bool pressBound_defined = false;

		for (int i = 0; i < sd_dev<T, 2, 9>.velBound_Count; ++i)
		{
			auto& velBound = sd_dev<T, 2, 9>.velBounds[i];
			if (x == 0 && velBound.side == left)	//left
			{
				auto leftCorner = velBound.center - velBound.width / 2;
				auto rightCorner = velBound.center + velBound.width / 2;

				if (y >= leftCorner && y <= rightCorner)
				{
					velBound_defined = true;
					uwVB[0] = velBound.u_w;
					uwVB[1] = 0.0;
					break;
				}
			}
		}

		for (int i = 0; i < sd_dev<T, 2, 9>.pressBound_Count; ++i)
		{
			auto& pressBound = sd_dev<T, 2, 9>.pressBounds[i];
			if (x == 0 && pressBound.side == left)	//left
			{
				auto leftCorner = pressBound.center - pressBound.width / 2;
				auto rightCorner = pressBound.center + pressBound.width / 2;

				if (y >= leftCorner && y <= rightCorner)
				{
					pressBound_defined = true;
					int pos_next = y * sd_dev<T, 2, 9>.gridDim_L[0] + x + 1;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = pressBound.dp_w / cs_sq<T> +1.0;
					break;
				}
			}
		}

		int i_1 = 3, i_2 = 7, i_3 = 6;
		int j_1 = 1, j_2 = 5, j_3 = 8;

		f[dataLayout(9, alpha, calcPos2D<T>(0, x, y), 0)] = f_star[dataLayout(9, alpha, pos, 0)];

		if (x == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_velBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_presBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_1)] = f_star[dataLayout(9, alpha, pos, i_1)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_1, x, y), i_1)] = f_star[dataLayout(9, alpha, pos, i_1)];


		if (x == 0 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_velBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_presBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_2)] = f_star[dataLayout(9, alpha, pos, i_2)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_2, x, y), i_2)] = f_star[dataLayout(9, alpha, pos, i_2)];

		if (x == 0 || y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_velBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_presBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_3)] = f_star[dataLayout(9, alpha, pos, i_3)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_3, x, y), i_3)] = f_star[dataLayout(9, alpha, pos, i_3)];
	}

}

template<typename T>
__global__ void streaming2D_kernel_R(T* f, T* f_star, vec<T, 2>* u_L, T* rho_L)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < sd_dev<T, 2, 9>.gridDim_L[0] && y < sd_dev<T, 2, 9>.gridDim_L[1])
	{
		int pos = y * sd_dev<T, 2, 9>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 2, 9>.alpha;

		T rhoVB_border = rho_L[pos];
		T rhoPB_border = 1.0;
		vec<T, 2> uwVB{ 0.0, 0.0 }, uwPB{ 0.0, 0.0 };
		bool velBound_defined = false;
		bool pressBound_defined = false;

		for (int i = 0; i < sd_dev<T, 2, 9>.velBound_Count; ++i)
		{
			auto& velBound = sd_dev<T, 2, 9>.velBounds[i];
			if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 && velBound.side == right)	//right
			{
				auto leftCorner = velBound.center + velBound.width / 2;
				auto rightCorner = velBound.center - velBound.width / 2;

				if (y <= leftCorner && y >= rightCorner)
				{
					velBound_defined = true;
					uwVB[0] = -velBound.u_w;
					uwVB[1] = 0.0;
					break;
				}
			}
		}

		for (int i = 0; i < sd_dev<T, 2, 9>.pressBound_Count; ++i)
		{
			auto& pressBound = sd_dev<T, 2, 9>.pressBounds[i];
			if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 && pressBound.side == right)	//right
			{
				auto leftCorner = pressBound.center + pressBound.width / 2;
				auto rightCorner = pressBound.center - pressBound.width / 2;

				if (y <= leftCorner && y >= rightCorner)
				{
					pressBound_defined = true;
					int pos_next = y * sd_dev<T, 2, 9>.gridDim_L[0] + x - 1;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = pressBound.dp_w / cs_sq<T> +1.0;
					break;
				}
			}
		}

		int i_1 = 1, i_2 = 5, i_3 = 8;
		int j_1 = 3, j_2 = 7, j_3 = 6;

		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_velBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_presBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_1)] = f_star[dataLayout(9, alpha, pos, i_1)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_1, x, y), i_1)] = f_star[dataLayout(9, alpha, pos, i_1)];

		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 || y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_velBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_presBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_2)] = f_star[dataLayout(9, alpha, pos, i_2)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_2, x, y), i_2)] = f_star[dataLayout(9, alpha, pos, i_2)];


		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_velBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_presBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_3)] = f_star[dataLayout(9, alpha, pos, i_3)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_3, x, y), i_3)] = f_star[dataLayout(9, alpha, pos, i_3)];

	}
}

template<typename T>
__global__ void streaming2D_kernel_B(T* f, T* f_star, vec<T, 2>* u_L, T* rho_L)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < sd_dev<T, 2, 9>.gridDim_L[0] && y < sd_dev<T, 2, 9>.gridDim_L[1])
	{
		int pos = y * sd_dev<T, 2, 9>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 2, 9>.alpha;

		T rhoVB_border = rho_L[pos];
		T rhoPB_border = 1.0;
		vec<T, 2> uwVB{ 0.0, 0.0 }, uwPB{ 0.0, 0.0 };
		bool velBound_defined = false;
		bool pressBound_defined = false;

		for (int i = 0; i < sd_dev<T, 2, 9>.velBound_Count; ++i)
		{
			auto& velBound = sd_dev<T, 2, 9>.velBounds[i];
			if (y == 0 && velBound.side == bottom)	//bottom
			{
				auto leftCorner = velBound.center + velBound.width / 2;
				auto rightCorner = velBound.center - velBound.width / 2;

				if (x <= leftCorner && x >= rightCorner)
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = velBound.u_w;
					break;
				}
			}
		}

		for (int i = 0; i < sd_dev<T, 2, 9>.pressBound_Count; ++i)
		{
			auto& pressBound = sd_dev<T, 2, 9>.pressBounds[i];
			if (y == 0 && pressBound.side == bottom)	//bottom
			{
				auto leftCorner = pressBound.center + pressBound.width / 2;
				auto rightCorner = pressBound.center - pressBound.width / 2;

				if (x <= leftCorner && x >= rightCorner)
				{
					pressBound_defined = true;
					int pos_next = (y + 1) * sd_dev<T, 2, 9>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = pressBound.dp_w / cs_sq<T> +1.0;
					break;
				}
			}
		}

		int i_1 = 8, i_2 = 4, i_3 = 7;
		int j_1 = 6, j_2 = 2, j_3 = 5;

		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_velBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_presBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_1)] = f_star[dataLayout(9, alpha, pos, i_1)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_1, x, y), i_1)] = f_star[dataLayout(9, alpha, pos, i_1)];

		if (y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_velBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_presBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_2)] = f_star[dataLayout(9, alpha, pos, i_2)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_2, x, y), i_2)] = f_star[dataLayout(9, alpha, pos, i_2)];

		if (x == 0 || y == 0)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_velBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_presBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_3)] = f_star[dataLayout(9, alpha, pos, i_3)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_3, x, y), i_3)] = f_star[dataLayout(9, alpha, pos, i_3)];

	}
}

template<typename T>
__global__ void streaming2D_kernel_T(T* f, T* f_star, vec<T, 2>* u_L, T* rho_L)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x < sd_dev<T, 2, 9>.gridDim_L[0] && y < sd_dev<T, 2, 9>.gridDim_L[1])
	{
		int pos = y * sd_dev<T, 2, 9>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 2, 9>.alpha;

		T rhoVB_border = rho_L[pos];
		T rhoPB_border = 1.0;
		vec<T, 2> uwVB{ 0.0, 0.0 }, uwPB{ 0.0, 0.0 };
		bool velBound_defined = false;
		bool pressBound_defined = false;

		for (int i = 0; i < sd_dev<T, 2, 9>.velBound_Count; ++i)
		{
			auto& velBound = sd_dev<T, 2, 9>.velBounds[i];
			if (y == sd_dev<T, 2, 9>.gridDim_L[1] - 1 && velBound.side == top)	//top
			{
				auto leftCorner = velBound.center - velBound.width / 2;
				auto rightCorner = velBound.center + velBound.width / 2;

				if (x >= leftCorner && x <= rightCorner)
				{
					velBound_defined = true;
					uwVB[0] = 0.0;
					uwVB[1] = -velBound.u_w;
					break;
				}
			}
		}

		for (int i = 0; i < sd_dev<T, 2, 9>.pressBound_Count; ++i)
		{
			auto& pressBound = sd_dev<T, 2, 9>.pressBounds[i];
			if (y == sd_dev<T, 2, 9>.gridDim_L[1] - 1 && pressBound.side == top)	//top
			{
				auto leftCorner = pressBound.center - pressBound.width / 2;
				auto rightCorner = pressBound.center + pressBound.width / 2;

				if (x >= leftCorner && x <= rightCorner)
				{
					pressBound_defined = true;
					int pos_next = (y - 1) * sd_dev<T, 2, 9>.gridDim_L[0] + x;

					uwPB = u_L[pos] + (T)0.5 * (u_L[pos] - u_L[pos_next]);
					rhoPB_border = pressBound.dp_w / cs_sq<T> +1.0;
					break;
				}
			}
		}

		int i_1 = 6, i_2 = 2, i_3 = 5;
		int j_1 = 8, j_2 = 4, j_3 = 7;

		if (x == 0 || y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_velBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_1)] = calc_f_with_presBound<T, 2, 9>(i_1, f_star[dataLayout(9, alpha, pos, i_1)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_1)] = f_star[dataLayout(9, alpha, pos, i_1)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_1, x, y), i_1)] = f_star[dataLayout(9, alpha, pos, i_1)];

		if (y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_velBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_2)] = calc_f_with_presBound<T, 2, 9>(i_2, f_star[dataLayout(9, alpha, pos, i_2)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_2)] = f_star[dataLayout(9, alpha, pos, i_2)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_2, x, y), i_2)] = f_star[dataLayout(9, alpha, pos, i_2)];

		if (x == sd_dev<T, 2, 9>.gridDim_L[0] - 1 || y == sd_dev<T, 2, 9>.gridDim_L[1] - 1)
		{
			if (velBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_velBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoVB_border, uwVB);
			else if (pressBound_defined)
				f[dataLayout(9, alpha, pos, j_3)] = calc_f_with_presBound<T, 2, 9>(i_3, f_star[dataLayout(9, alpha, pos, i_3)], rhoPB_border, uwPB);
			else
				f[dataLayout(9, alpha, pos, j_3)] = f_star[dataLayout(9, alpha, pos, i_3)];
		}
		else
			f[dataLayout(9, alpha, calcPos2D<T>(i_3, x, y), i_3)] = f_star[dataLayout(9, alpha, pos, i_3)];
	}
}

template<typename T>
__global__ void collision2D_kernel_CM(T* rho_L,  vec<T, 2>* u_L, vec<T, 2>* F_ext_L, T* f, T* f_star, T r_vis, T zerothMomentMean = 0, T firstMomentMean = 0, T seconMomentMean = 0)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	CustomSqrt<T> customSqrt;

	if (x < sd_dev<T, 2, 9>.gridDim_L[0] && y < sd_dev<T, 2, 9>.gridDim_L[1])
	{
		int pos = y * sd_dev<T, 2, 9>.gridDim_L[0] + x;
		int alpha = sd_dev<T, 2, 9>.alpha;

		T f0 = f[dataLayout(9, alpha, pos, 0)],
			f1 = f[dataLayout(9, alpha, pos, 1)],
			f2 = f[dataLayout(9, alpha, pos, 2)],
			f3 = f[dataLayout(9, alpha, pos, 3)],
			f4 = f[dataLayout(9, alpha, pos, 4)],
			f5 = f[dataLayout(9, alpha, pos, 5)],
			f6 = f[dataLayout(9, alpha, pos, 6)],
			f7 = f[dataLayout(9, alpha, pos, 7)],
			f8 = f[dataLayout(9, alpha, pos, 8)];

		T rho = rho_L[pos];
		T ux = u_L[pos][0];
		T uy = u_L[pos][1];
		T Fx = F_ext_L[pos][0];
		T Fy = F_ext_L[pos][1];

		T r0 = sd_dev<T, 2, 9>.zerothRelaxationTime,
			r1 = sd_dev<T, 2, 9>.lowRelaxationTimes,
			r2 = r_vis,
			r3 = r_vis,
			r4, r5;

		if (sd_dev<T, 2, 9>.localRelaxation)
		{
			T r = sd_dev<T, 2, 9>.param_0 * rho / zerothMomentMean + sd_dev<T, 2, 9>.param_1 * customSqrt.sqrt(rho * rho * (ux * ux + uy * uy)) / firstMomentMean + sd_dev<T, 2, 9>.param_2;
			r4 = r;
			r5 = r;
		}
		else
		{
			r4 = sd_dev<T, 2, 9>.r_3;
			r5 = sd_dev<T, 2, 9>.r_3;
		}

		T f_t0 = r0 * (f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 - rho);
		T f_t1 = -r1 * (f0 * ux + f2 * ux + f4 * ux + f1 * (ux - 1.0) + f3 * (ux + 1.0) + f5 * (ux - 1.0) + f6 * (ux + 1.0) + f7 * (ux + 1.0) + f8 * (ux - 1.0));
		T f_t2 = -r1 * (f0 * uy + f1 * uy + f3 * uy + f2 * (uy - 1.0) + f4 * (uy + 1.0) + f5 * (uy - 1.0) + f6 * (uy - 1.0) + f7 * (uy + 1.0) + f8 * (uy + 1.0));
		T f_t3 = r2 * (rho * (-2.0 / 3.0) + f1 * (square(ux - 1.0) + uy * uy) + f2 * (square(uy - 1.0) + ux * ux) + f3 * (square(ux + 1.0) + uy * uy) + f4 * (square(uy + 1.0) + ux * ux) + f5 * (square(ux - 1.0) + square(uy - 1.0)) + f6 * (square(ux + 1.0) + square(uy - 1.0)) + f7 * (square(ux + 1.0) + square(uy + 1.0)) + f8 * (square(ux - 1.0) + square(uy + 1.0)) + f0 * (ux * ux + uy * uy));
		T f_t4 = r2 * (f0 * (ux * ux - uy * uy) + f1 * (square(ux - 1.0) - uy * uy) - f2 * (square(uy - 1.0) - ux * ux) + f3 * (square(ux + 1.0) - uy * uy) - f4 * (square(uy + 1.0) - ux * ux) + f5 * (square(ux - 1.0) - square(uy - 1.0)) + f6 * (square(ux + 1.0) - square(uy - 1.0)) + f7 * (square(ux + 1.0) - square(uy + 1.0)) + f8 * (square(ux - 1.0) - square(uy + 1.0)));
		T f_t5 = r3 * (f1 * uy * (ux - 1.0) + f2 * ux * (uy - 1.0) + f3 * uy * (ux + 1.0) + f4 * ux * (uy + 1.0) + f5 * (ux - 1.0) * (uy - 1.0) + f6 * (ux + 1.0) * (uy - 1.0) + f7 * (ux + 1.0) * (uy + 1.0) + f8 * (ux - 1.0) * (uy + 1.0) + f0 * ux * uy);
		T f_t6 = -r4 * (f0 * (ux * ux) * uy + f1 * uy * square(ux - 1.0) + f2 * (ux * ux) * (uy - 1.0) + f3 * uy * square(ux + 1.0) + f4 * (ux * ux) * (uy + 1.0) + f5 * square(ux - 1.0) * (uy - 1.0) + f6 * square(ux + 1.0) * (uy - 1.0) + f7 * square(ux + 1.0) * (uy + 1.0) + f8 * square(ux - 1.0) * (uy + 1.0));
		T f_t7 = -r4 * (f0 * ux * (uy * uy) + f1 * (uy * uy) * (ux - 1.0) + f2 * ux * square(uy - 1.0, 2.0) + f3 * (uy * uy) * (ux + 1.0) + f4 * ux * square(uy + 1.0) + f5 * (ux - 1.0) * square(uy - 1.0) + f6 * (ux + 1.0) * square(uy - 1.0) + f7 * (ux + 1.0) * square(uy + 1.0) + f8 * (ux - 1.0) * square(uy + 1.0));
		T f_t8 = r5 * (rho * (-1.0 / 9.0) + f5 * square(ux - 1.0) * square(uy - 1.0) + f6 * square(ux + 1.0) * square(uy - 1.0) + f7 * square(ux + 1.0) * square(uy + 1.0) + f8 * square(ux - 1.0) * square(uy + 1.0) + f0 * (ux * ux) * (uy * uy) + f1 * (uy * uy) * square(ux - 1.0) + f2 * (ux * ux) * square(uy - 1.0) + f3 * (uy * uy) * square(ux + 1.0) + f4 * (ux * ux) * square(uy + 1.0));

		T Omega_0 = -f_t8 - f_t7 * ux * 2.0 - f_t6 * uy * 2.0 - f_t3 * ((ux * ux) / 2.0 + (uy * uy) / 2.0 - 1.0) + f_t4 * ((ux * ux) / 2.0 - (uy * uy) / 2.0) - f_t0 * ((ux * ux) * (uy * uy) - ux * ux - uy * uy + 1.0) + f_t1 * (ux * 2.0 - ux * (uy * uy) * 2.0) + f_t2 * (uy * 2.0 - (ux * ux) * uy * 2.0) - f_t5 * ux * uy * 4.0;
		T Omega_1 = f_t8 / 2.0 + f_t2 * (ux * uy + (ux * ux) * uy) + f_t6 * uy + f_t3 * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - f_t4 * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) - f_t1 * (ux - ux * (uy * uy) - (uy * uy) / 2.0 + 1.0 / 2.0) + f_t7 * (ux + 1.0 / 2.0) - f_t0 * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * ux) / 2.0) + f_t5 * (uy + ux * uy * 2.0);
		T Omega_2 = f_t8 / 2.0 + f_t1 * (ux * uy + ux * (uy * uy)) + f_t7 * ux + f_t3 * (uy / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) + f_t4 * (uy / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0 + 1.0 / 4.0) - f_t2 * (uy - (ux * ux) * uy - (ux * ux) / 2.0 + 1.0 / 2.0) + f_t6 * (uy + 1.0 / 2.0) - f_t0 * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * uy) / 2.0) + f_t5 * (ux + ux * uy * 2.0);
		T Omega_3 = f_t8 / 2.0 - f_t2 * (ux * uy - (ux * ux) * uy) + f_t6 * uy - f_t3 * (ux / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) + f_t4 * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - f_t1 * (ux - ux * (uy * uy) + (uy * uy) / 2.0 - 1.0 / 2.0) + f_t7 * (ux - 1.0 / 2.0) + f_t0 * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * ux) / 2.0) - f_t5 * (uy - ux * uy * 2.0);
		T Omega_4 = f_t8 / 2.0 - f_t1 * (ux * uy - ux * (uy * uy)) + f_t7 * ux - f_t3 * (uy / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) - f_t4 * (uy / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0 - 1.0 / 4.0) - f_t2 * (uy - (ux * ux) * uy + (ux * ux) / 2.0 - 1.0 / 2.0) + f_t6 * (uy - 1.0 / 2.0) + f_t0 * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * uy) / 2.0) - f_t5 * (ux - ux * uy * 2.0);
		T Omega_5 = f_t8 * (-1.0 / 4.0) - f_t5 * (ux / 2.0 + uy / 2.0 + ux * uy + 1.0 / 4.0) - f_t2 * (ux / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) - f_t1 * (uy / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0) - f_t3 * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) + f_t4 * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 + 1.0 / 4.0) - f_t6 * (uy / 2.0 + 1.0 / 4.0);
		T Omega_6 = f_t8 * (-1.0 / 4.0) - f_t5 * (ux / 2.0 - uy / 2.0 + ux * uy - 1.0 / 4.0) + f_t2 * (ux / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) + f_t1 * (uy / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0) - f_t3 * (ux * (-1.0 / 8.0) + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) - f_t4 * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 - 1.0 / 4.0) - f_t6 * (uy / 2.0 + 1.0 / 4.0);
		T Omega_7 = f_t8 * (-1.0 / 4.0) + f_t5 * (ux / 2.0 + uy / 2.0 - ux * uy - 1.0 / 4.0) - f_t2 * (ux / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) - f_t1 * (uy / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * uy) / 4.0) + f_t3 * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0) - f_t4 * (ux / 8.0 - uy / 8.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 - 1.0 / 4.0) - f_t6 * (uy / 2.0 - 1.0 / 4.0);
		T Omega_8 = f_t8 * (-1.0 / 4.0) + f_t2 * (ux / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) - f_t3 * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) + f_t4 * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 + 1.0 / 4.0) - f_t6 * (uy / 2.0 - 1.0 / 4.0) + (f_t1 * (uy - uy * uy) * (ux * 2.0 + 1.0)) / 4.0 - (f_t5 * (ux * 2.0 + 1.0) * (uy * 2.0 - 1.0)) / 4.0;

		T G_0 = Fx * uy * (r4 / 2.0 - 1.0) * (-2.0 / 3.0) - Fy * ux * (r4 / 2.0 - 1.0) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * 2.0 - ux * (uy * uy) * 2.0) + Fy * (r1 / 2.0 - 1.0) * (uy * 2.0 - (ux * ux) * uy * 2.0);
		T G_1 = Fy * (r1 / 2.0 - 1.0) * (ux * uy + (ux * ux) * uy) + (Fx * uy * (r4 / 2.0 - 1.0)) / 3.0 - Fx * (r1 / 2.0 - 1.0) * (ux - ux * (uy * uy) - (uy * uy) / 2.0 + 1.0 / 2.0) + (Fy * (r4 / 2.0 - 1.0) * (ux + 1.0 / 2.0)) / 3.0;
		T G_2 = Fx * (r1 / 2.0 - 1.0) * (ux * uy + ux * (uy * uy)) + (Fy * ux * (r4 / 2.0 - 1.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (uy - (ux * ux) * uy - (ux * ux) / 2.0 + 1.0 / 2.0) + (Fx * (r4 / 2.0 - 1.0) * (uy + 1.0 / 2.0)) / 3.0;
		T G_3 = -Fy * (r1 / 2.0 - 1.0) * (ux * uy - (ux * ux) * uy) + (Fx * uy * (r4 / 2.0 - 1.0)) / 3.0 - Fx * (r1 / 2.0 - 1.0) * (ux - ux * (uy * uy) + (uy * uy) / 2.0 - 1.0 / 2.0) + (Fy * (r4 / 2.0 - 1.0) * (ux - 1.0 / 2.0)) / 3.0;
		T G_4 = -Fx * (r1 / 2.0 - 1.0) * (ux * uy - ux * (uy * uy)) + (Fy * ux * (r4 / 2.0 - 1.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (uy - (ux * ux) * uy + (ux * ux) / 2.0 - 1.0 / 2.0) + (Fx * (r4 / 2.0 - 1.0) * (uy - 1.0 / 2.0)) / 3.0;
		T G_5 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0);
		T G_6 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 3.0 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0);
		T G_7 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * uy) / 4.0);
		T G_8 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 3.0 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) + (Fx * (uy - uy * uy) * (r1 / 2.0 - 1.0) * (ux * 2.0 + 1.0)) / 4.0;

		f_star[dataLayout(9, alpha, pos, 0)] = f0 + Omega_0 + G_0;
		f_star[dataLayout(9, alpha, pos, 1)] = f1 + Omega_1 + G_1;
		f_star[dataLayout(9, alpha, pos, 2)] = f2 + Omega_2 + G_2;
		f_star[dataLayout(9, alpha, pos, 3)] = f3 + Omega_3 + G_3;
		f_star[dataLayout(9, alpha, pos, 4)] = f4 + Omega_4 + G_4;
		f_star[dataLayout(9, alpha, pos, 5)] = f5 + Omega_5 + G_5;
		f_star[dataLayout(9, alpha, pos, 6)] = f6 + Omega_6 + G_6;
		f_star[dataLayout(9, alpha, pos, 7)] = f7 + Omega_7 + G_7;
		f_star[dataLayout(9, alpha, pos, 8)] = f8 + Omega_8 + G_8;
	}
}


#pragma endregion
#endif