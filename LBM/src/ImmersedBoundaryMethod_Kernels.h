#ifndef IMB_KERNELS
#define IMB_KERNELS
#include "ImmersedBody.h"
#include "SimDomain.h"
#include <cuda_runtime_api.h>
#include "LBM_Types.h"
#include "CustomCudaExtensions.h"


template<typename T, int D, int Q>
__constant__ SimDomain_dev<T, D, Q> sd_dev;

//3D
template<typename T>
__device__ vec<T, 3> calc_uf_lag3D(vec<T, 3> sample, vec<T, 3>* u_L, int kernelSize, unsigned char sharpness)
{
	int x{}, y{}, z{};
	vec<T, 3> uf_lag{};


	for (int zk = 0; zk < kernelSize; ++zk)
	{
		for (int yk = 0; yk < kernelSize; ++yk)
		{
			for (int xk = 0; xk < kernelSize; ++xk)
			{
				int r = kernelSize / 2;
				x = roundf(sample[0]) - r + xk;
				y = roundf(sample[1]) - r + yk;
				z = roundf(sample[2]) - r + zk;

				if (x >= 0 && x < sd_dev<T, 3, 27>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 3, 27>.gridDim_L[1] && z >= 0 && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					T delta = deltaFunc(vec<T, 3>{(T)x, (T)y, (T)z}, sample, sharpness);
					uf_lag += u_L[(z * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x] * delta;
				}
			}
		}
	}

	return uf_lag;
}

template<typename T>
__device__ T calc_rho_lag3D(vec<T, 3> sample, T* rho_L, int kernelSize, unsigned char sharpness)
{
	int x{}, y{}, z{};
	T rho_lag = 0.0;

	for (int zk = 0; zk < kernelSize; ++zk)
	{
		for (int yk = 0; yk < kernelSize; ++yk)
		{
			for (int xk = 0; xk < kernelSize; ++xk)
			{
				int r = kernelSize / 2;
				x = roundf(sample[0]) - r + xk;
				y = roundf(sample[1]) - r + yk;
				z = roundf(sample[2]) - r + zk;

				if (x >= 0 && x < sd_dev<T, 3, 27>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 3, 27>.gridDim_L[1] && z >= 0 && z < sd_dev<T, 3, 27>.gridDim_L[2])
				{
					T delta = deltaFunc(vec<T, 3>{(T)x, (T)y, (T)z}, sample, sharpness);
					rho_lag += rho_L[(z * sd_dev<T, 3, 27>.gridDim_L[1] + y) * sd_dev<T, 3, 27>.gridDim_L[0] + x] * delta;
				}
			}
		}
	}

	return rho_lag;
}

template<typename T, int D, int Q>
__device__ void collision_ghost_kernel_CM(T* rho_L, vec<T, D>* u_L, int x_S_pos, int x_B_pos, int x_F_pos, vec<T,D> velocity, T q, T* f, T* f_star)
{
	int alpha = sd_dev<T, D, Q>.alpha;

	for (int i = 0; i < Q; ++i)
	{
		vec<T, D> u_solid;

		if (q >= (T)3.0 / 4.0)
		{
			u_solid = ((q - (T)1.0) * u_L[x_B_pos] + velocity) / q;
		}
		else
		{
			u_solid = (q - (T)1.0) * u_L[x_B_pos] + velocity + ((T)1.0 - q) * ((q - (T)1.0) * u_L[x_F_pos] + (T)2.0 * velocity) / ((T)1.0 + q);
		}

		T f_s_eq = calc_f_eq_CM<T>(i, rho_L[x_B_pos], u_solid);

		T f_s_neq;

		if (q > (T)3.0 / 4.0)
		{
			T f_eq_B = calc_f_eq_CM<T>(i, rho_L[x_B_pos], u_L[x_B_pos]);

			f_s_neq = f[dataLayout(Q, alpha, x_B_pos, i)] - f_eq_B;
		}
		else
		{
			T f_eq_B = calc_f_eq_CM<T>(i, rho_L[x_B_pos], u_L[x_B_pos]);

			T f_eq_F = calc_f_eq_CM<T>(i, rho_L[x_F_pos], u_L[x_F_pos]);

			f_s_neq = q * (f[dataLayout(Q, alpha, x_B_pos, i)] - f_eq_B) + ((T)1.0 - q) * (f[dataLayout(Q, alpha, x_F_pos, i)] - f_eq_F);
		}

		f_star[dataLayout(Q, alpha, x_S_pos, i)] = f_s_neq + f_s_eq;
	}
}

template<typename T, int D, int Q>
__device__ void collision_ghost_kernel_BGK(T* rho_L, vec<T, D>* u_L, int x_S_pos, int x_B_pos, int x_F_pos, vec<T, D> velocity, T q, T* f, T* f_star)
{
	int alpha = sd_dev<T, D, Q>.alpha;

	for (int i = 0; i < Q; ++i)
	{
		vec<T, D> u_solid;

		if (q >= (T)3.0 / 4.0)
		{
			u_solid = ((q - (T)1.0) * u_L[x_B_pos] + velocity) / q;
		}
		else
		{
			u_solid = (q - (T)1.0) * u_L[x_B_pos] + velocity + ((T)1.0 - q) * ((q - (T)1.0) * u_L[x_F_pos] + (T)2.0 * velocity) / ((T)1.0 + q);
		}

		T f_s_eq = sd_dev<T, D, Q>.w[i] * rho_L[x_B_pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_solid) / cs_sq<T> +dot(sd_dev<T, D, Q>.c[i], u_solid) *
			dot(sd_dev<T, D, Q>.c[i], u_solid) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_solid, u_solid) / ((T)2.0 * cs_sq<T>));


		T f_s_neq;

		if (q > (T)3.0 / 4.0)
		{
			T f_eq_B = sd_dev<T, D, Q>.w[i] * rho_L[x_B_pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L[x_B_pos]) / cs_sq<T> +dot(sd_dev<T, D, Q>.c[i], u_L[x_B_pos]) *
				dot(sd_dev<T, D, Q>.c[i], u_L[x_B_pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L[x_B_pos], u_L[x_B_pos]) / ((T)2.0 * cs_sq<T>));

			f_s_neq = f[dataLayout(Q, alpha, x_B_pos, i)] - f_eq_B;
		}
		else
		{
			T f_eq_B = sd_dev<T, D, Q>.w[i] * rho_L[x_B_pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L[x_B_pos]) / cs_sq<T> +dot(sd_dev<T, D, Q>.c[i], u_L[x_B_pos]) *
				dot(sd_dev<T, D, Q>.c[i], u_L[x_B_pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L[x_B_pos], u_L[x_B_pos]) / ((T)2.0 * cs_sq<T>));

			T f_eq_F = sd_dev<T, D, Q>.w[i] * rho_L[x_F_pos] * ((T)1.0 + dot(sd_dev<T, D, Q>.c[i], u_L[x_F_pos]) / cs_sq<T> +dot(sd_dev<T, D, Q>.c[i], u_L[x_F_pos]) *
				dot(sd_dev<T, D, Q>.c[i], u_L[x_F_pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L[x_F_pos], u_L[x_F_pos]) / ((T)2.0 * cs_sq<T>));

			f_s_neq = q * (f[dataLayout(Q, alpha, x_B_pos, i)] - f_eq_B) + ((T)1.0 - q) * (f[dataLayout(Q, alpha, x_F_pos, i)] - f_eq_F);
		}

		f_star[dataLayout(Q, alpha, x_S_pos, i)] = f_s_neq + f_s_eq;
	}
}


template<typename T>
__global__ void calcInlet3D_kernel(T* rho_L, vec<T, 3>* u_L, T* f, T* f_star, vec<T, 3>* samples, vec<T, 3>* velocities, vec<T, 3>* normals, int sampleCount, bool BGK_used)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;

	if (sampID < sampleCount)
	{
		int x = roundf(samples[sampID][0]);
		int y = roundf(samples[sampID][1]);
		int z = roundf(samples[sampID][2]);
		int x_loc, y_loc, z_loc;

		if (x >= 0 && x < sd_dev<T, 3, 27>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 3, 27>.gridDim_L[1] && z >= 0 && z < sd_dev<T, 3, 27>.gridDim_L[2])
		{
			vec<T, 3> node_S{ (T)x,(T)y, (T)z };

			if (dot(normals[sampID], (node_S - samples[sampID])) > 0)
				node_S -= normals[sampID];

			node_S[0] = roundf(node_S[0]);
			node_S[1] = roundf(node_S[1]);
			node_S[2] = roundf(node_S[2]);

			for (int k = 1; k < 27; ++k)
			{
				int x_B = roundf(node_S[0] + sd_dev<T, 3, 27>.c[k][0]);
				int y_B = roundf(node_S[1] + sd_dev<T, 3, 27>.c[k][1]);
				int z_B = roundf(node_S[2] + sd_dev<T, 3, 27>.c[k][2]);
				vec<T, 3> node_B{ (T)x_B, (T)y_B, (T)z_B };

				if (dot(normals[sampID], (node_B - samples[sampID])) > 0)
				{
					if (dot(normals[sampID], (node_B - sd_dev<T, 3, 27>.c[k] - samples[sampID])) <= 0)
					{
						node_S[0] = roundf(node_B[0] - sd_dev<T, 3, 27>.c[k][0]);
						node_S[1] = roundf(node_B[1] - sd_dev<T, 3, 27>.c[k][1]);
						node_S[2] = roundf(node_B[2] - sd_dev<T, 3, 27>.c[k][2]);

						int node_F0 = roundf((T)x_B + sd_dev<T, 3, 27>.c[k][0]);
						int node_F1 = roundf((T)y_B + sd_dev<T, 3, 27>.c[k][1]);
						int node_F2 = roundf((T)z_B + sd_dev<T, 3, 27>.c[k][2]);

						int pos_solid = (node_S[2] * sd_dev<T, 3, 27>.gridDim_L[1] + node_S[1] ) * sd_dev<T, 3, 27>.gridDim_L[0] + node_S[0];
						int pos_bound = (z_B * sd_dev<T, 3, 27>.gridDim_L[1] + y_B) * sd_dev<T, 3, 27>.gridDim_L[0] + x_B;
						int pos_fluid = (node_F2 * sd_dev<T, 3, 27>.gridDim_L[1] + node_F1) * sd_dev<T, 3, 27>.gridDim_L[0] + node_F0;

						T denominator = dot(normals[sampID], sd_dev<T, 3, 27>.c[k]);
						if (abs(denominator) > 0)
						{
							T q = abs(dot(normals[sampID], node_B - samples[sampID]) / denominator);

							if (BGK_used)
								collision_ghost_kernel_BGK<T, 3, 27>(rho_L, u_L, pos_solid, pos_bound, pos_fluid, velocities[sampID], q, f, f_star);
							else
								collision_ghost_kernel_CM<T, 3, 27>(rho_L, u_L, pos_solid, pos_bound, pos_fluid, velocities[sampID], q, f, f_star);
						}

					}
				}
			}
		}
	}
}


template<typename T>
__global__ void calcFs3D_kernel(vec<T, 3>* u_unc_L, vec<T, 3>* F_ext_L, T* rho_L, vec<T, 3>* samples, vec<T, 3>* velocities, vec<T,3> *Fsg, int kernelSize, int sampleCount, unsigned char sharpness)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomAtomicAdd<T> atomAdd;
	if (sampID < sampleCount)
	{
		int x = roundf(samples[sampID][0]);
		int y = roundf(samples[sampID][1]);
		int z = roundf(samples[sampID][2]);
		int x_loc, y_loc, z_loc;


		if (x >= 0 && x < sd_dev<T, 3, 27>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 3, 27>.gridDim_L[1] && z >= 0 && z < sd_dev<T, 3, 27>.gridDim_L[2])
		{
			vec<T, 3> Fs_lag = calc_rho_lag3D(samples[sampID], rho_L, kernelSize, sharpness)
				* (velocities[sampID] - calc_uf_lag3D(samples[sampID], u_unc_L, kernelSize, sharpness));

			atomAdd.AtomicAdd(&(*Fsg)[0], Fs_lag[0]);
			atomAdd.AtomicAdd(&(*Fsg)[1], Fs_lag[1]);
			atomAdd.AtomicAdd(&(*Fsg)[2], Fs_lag[2]);
			for (int zk = 0; zk < kernelSize; ++zk)
			{
				for (int yk = 0; yk < kernelSize; ++yk)
				{
					for (int xk = 0; xk < kernelSize; ++xk)
					{
						int r = kernelSize / 2;
						x_loc = x - r + xk;
						y_loc = y - r + yk;
						z_loc = z - r + zk;

						if (x_loc >= 0 && x_loc < sd_dev<T, 3, 27>.gridDim_L[0] && y_loc >= 0 && y_loc < sd_dev<T, 3, 27>.gridDim_L[1] && z_loc >= 0 && z_loc < sd_dev<T, 3, 27>.gridDim_L[2])
						{
							int pos_L = (z_loc * sd_dev<T, 3, 27>.gridDim_L[1] + y_loc) * sd_dev<T, 3, 27>.gridDim_L[0] + x_loc;
							T delta = deltaFunc(samples[sampID], { (T)x_loc, (T)y_loc, (T)z_loc }, sharpness);
							vec<T, 3> Fs_eul_loc = Fs_lag * delta;

							if (sum(Fs_eul_loc * Fs_eul_loc) > (T)0.0)
							{
								atomAdd.AtomicAdd(&(F_ext_L[pos_L][0]), Fs_eul_loc[0]);
								atomAdd.AtomicAdd(&(F_ext_L[pos_L][1]), Fs_eul_loc[1]);
								atomAdd.AtomicAdd(&(F_ext_L[pos_L][2]), Fs_eul_loc[2]);
							}
						}
					}
				}
			}
		}
	}
}

//template<typename T>
//__global__ void calcFs3D_Inlet_kernel(vec<T, 3>* u_unc_L, vec<T, 3>* F_ext_L, T* rho_L, vec<T, 3>* samples, vec<T, 3>* normals, vec<T, 3>* velocities, int kernelSize, int sampleCount, unsigned char sharpness)
//{
//	int sampID = blockIdx.x * blockDim.x + threadIdx.x;
//	CustomAtomicAdd<T> atomAdd;
//	if (sampID < sampleCount)
//	{
//		int x = roundf(samples[sampID][0]);
//		int y = roundf(samples[sampID][1]);
//		int z = roundf(samples[sampID][2]);
//		int x_loc, y_loc, z_loc;
//
//
//		if (x >= 0 && x < sd_dev<T, 3, 27>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 3, 27>.gridDim_L[1] && z >= 0 && z < sd_dev<T, 3, 27>.gridDim_L[2])
//		{
//
//			vec<T, 3> node_S{ (T)x,(T)y, (T)z };
//
//			if (dot(normals[sampID], (node_S - samples[sampID])) > 0)
//				node_S -= normals[sampID];
//
//			node_S[0] = roundf(node_S[0]);
//			node_S[1] = roundf(node_S[1]);
//			node_S[2] = roundf(node_S[2]);
//			vec<T, 3> node_B = node_S + normals[sampID];
//
//			node_B[0] = roundf(node_B[0]);
//			node_B[1] = roundf(node_B[1]);
//			node_B[2] = roundf(node_B[2]);
//
//			int pos_B = (node_B[2] * sd_dev<T, 3, 27>.gridDim_L[1] + node_B[1]) * sd_dev<T, 3, 27>.gridDim_L[0] + node_B[0];
//			int pos_S = (node_S[2] * sd_dev<T, 3, 27>.gridDim_L[1] + node_S[1]) * sd_dev<T, 3, 27>.gridDim_L[0] + node_S[0];
//			vec<T, 3> Fs_eul_loc = rho_L[pos_B] * dot((velocities[sampID] - u_unc_L[pos_B] ),normals[sampID])*normals[sampID];
//
//			//printf("rho %2.2f\n", rho_L[pos_B]);
//			//printf("Normals {%2.2f, %2.2f, %2.2f}\n", normals[sampID][0], normals[sampID][1], normals[sampID][2]);
//			//printf("u_unc_L {%2.2f, %2.2f, %2.2f}\n", u_unc_L[pos_B][0], u_unc_L[pos_B][1], u_unc_L[pos_B][2]);
//			//printf("velocities {%2.2f, %2.2f, %2.2f}\n", velocities[sampID][0], velocities[sampID][1], velocities[sampID][2]);
//			//printf("Fs_eul_loc {%2.2f, %2.2f, %2.2f}\n", Fs_eul_loc[0], Fs_eul_loc[1], Fs_eul_loc[2]);
//
//			atomAdd.AtomicAdd(&(F_ext_L[pos_B][0]), Fs_eul_loc[0]);
//			atomAdd.AtomicAdd(&(F_ext_L[pos_B][1]), Fs_eul_loc[1]);
//			atomAdd.AtomicAdd(&(F_ext_L[pos_B][2]), Fs_eul_loc[2]);
//
//		}
//	}
//}


template<typename T, int D>
__global__ void updatePosAndVel_kernel(vec<T, D>* velocities, vec<T, D>* samples, vec<T, D>* normals, vec<T, D*D> R, vec<T, D> center_old, vec<T, D> center, int sampleSize)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;

	if (sampID < sampleSize)
	{
		vec<T, D> oldSample = samples[sampID];
		samples[sampID] = R * (samples[sampID] - center_old) + center;
		normals[sampID] = R * normals[sampID];
		velocities[sampID] = samples[sampID]-oldSample;
	}
}

template<typename T, int D>
__global__ void updatePosAndVel_kernel(vec<T, D>* velocities, vec<T, D>* samples, vec<T, D> *normals, vec<T, D> inlet_velocity, vec<T, D* D> R, vec<T, D> center_old, vec<T, D> center, int sampleSize)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;

	if (sampID < sampleSize)
	{
		vec<T, D> oldSample = samples[sampID];
		samples[sampID] = R * (samples[sampID] - center_old) + center;
		normals[sampID] = R * normals[sampID];
		velocities[sampID] = samples[sampID] - oldSample + dot(inlet_velocity, normals[sampID]) * normals[sampID];
	}
}

template<typename T, int D>
__global__ void updatePosAndVel_kernel(vec<T, D>* velocities, vec<T, D>* samples, vec<T, D>* normals, char* is_inlet, vec<T, D> inlet_velocity, vec<T, D* D> R, vec<T, D> center_old, vec<T, D> center, int sampleSize)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;

	if (sampID < sampleSize)
	{
		vec<T, D> oldSample = samples[sampID];
		samples[sampID] = R * (samples[sampID] - center_old) + center;
		normals[sampID] = R * normals[sampID];

		if(is_inlet[sampID])
			velocities[sampID] = samples[sampID] - oldSample + dot(inlet_velocity, normals[sampID]) * normals[sampID];
		else
			velocities[sampID] = samples[sampID] - oldSample;
	}
}

template<typename T, int D>
__global__ void rescaleVelocities_kernel(vec<T, D > * velocities, T scale_u, int sampleCount)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;

	if (sampID < sampleCount)
	{
		velocities[sampID] *= scale_u;
	}
}

#pragma region IMB_Kernels_2D
//2D
template<typename T>
__device__ vec<T, 2> calc_uf_lag2D(vec<T, 2> sample, vec<T, 2>* u_L, int kernelSize, unsigned char sharpness)
{
	int x{}, y{};
	vec<T, 2> uf_lag{};

	for (int yk = 0; yk < kernelSize; ++yk)
	{
		for (int xk = 0; xk < kernelSize; ++xk)
		{
			int r = kernelSize / 2;
			x = roundf(sample[0]) - r + xk;
			y = roundf(sample[1]) - r + yk;

			if (x >= 0 && x < sd_dev<T, 2, 9>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 2, 9>.gridDim_L[1])
			{
				T delta = deltaFunc(vec<T, 2>{(T)x, (T)y}, sample, sharpness);
				uf_lag += u_L[y * sd_dev<T, 2, 9>.gridDim_L[0] + x] * delta;
			}
		}
	}

	return uf_lag;
}

template<typename T>
__device__ T calc_rho_lag2D(vec<T, 2> sample, T* rho_L, int kernelSize, unsigned char sharpness)
{
	int x{}, y{};
	T rho_lag = 0.0;

	for (int yk = 0; yk < kernelSize; ++yk)
	{
		for (int xk = 0; xk < kernelSize; ++xk)
		{
			int r = kernelSize / 2;
			x = roundf(sample[0]) - r + xk;
			y = roundf(sample[1]) - r + yk;

			if (x >= 0 && x < sd_dev<T, 2, 9>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 2, 9>.gridDim_L[1])
			{
				T delta = deltaFunc(vec<T, 2>{(T)x, (T)y}, sample, sharpness);
				rho_lag += rho_L[y * sd_dev<T, 2, 9>.gridDim_L[0] + x] * delta;
			}
		}
	}

	return rho_lag;
}

template<typename T>
__global__ void calcInlet2D_kernel(T* rho_L, vec<T, 2>* u_L, T* f, T* f_star, vec<T, 2>* samples, vec<T, 2>* velocities, vec<T, 2>* normals, int sampleCount, bool BGK_used)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;

	if (sampID < sampleCount)
	{
		int x = roundf(samples[sampID][0]);
		int y = roundf(samples[sampID][1]);
		int x_loc, y_loc;

		if (x >= 0 && x < sd_dev<T, 2, 9>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 2, 9>.gridDim_L[1])
		{
			vec<T, 2> node_S{ (T)x,(T)y };

			if (dot(normals[sampID], (node_S - samples[sampID])) > 0)
				node_S -= normals[sampID];

			node_S[0] = roundf(node_S[0]);
			node_S[1] = roundf(node_S[1]);

			for (int k = 1; k < 9; ++k)
			{
				int x_B = roundf(node_S[0] + sd_dev<T, 2, 9>.c[k][0]);
				int y_B = roundf(node_S[1] + sd_dev<T, 2, 9>.c[k][1]);
				vec<T, 2> node_B{ (T)x_B, (T)y_B };

				if (dot(normals[sampID], (node_B - samples[sampID])) > 0)
				{
					if (dot(normals[sampID], (node_B - sd_dev<T, 2, 9>.c[k] - samples[sampID])) <= 0)
					{
						node_S[0] = roundf(node_B[0] - sd_dev<T, 2, 9>.c[k][0]);
						node_S[1] = roundf(node_B[1] - sd_dev<T, 2, 9>.c[k][1]);

						int node_F0 = roundf((T)x_B + sd_dev<T, 2, 9>.c[k][0]);
						int node_F1 = roundf((T)y_B + sd_dev<T, 2, 9>.c[k][1]);

						int pos_solid = node_S[1] * sd_dev<T, 2, 9>.gridDim_L[0] + node_S[0];
						int pos_bound = y_B * sd_dev<T, 2, 9>.gridDim_L[0] + x_B;
						int pos_fluid = node_F1 * sd_dev<T, 2, 9>.gridDim_L[0] + node_F0;

						T denominator = dot(normals[sampID], sd_dev<T, 2, 9>.c[k]);
						if (abs(denominator) > 0)
						{
							T q = abs(dot(normals[sampID], node_B - samples[sampID]) / denominator);

							if (BGK_used)
								collision_ghost_kernel_BGK<T, 2, 9>(rho_L, u_L, pos_solid, pos_bound, pos_fluid, velocities[sampID], q, f, f_star);
							else
								collision_ghost_kernel_CM<T, 2, 9>(rho_L, u_L, pos_solid, pos_bound, pos_fluid, velocities[sampID], q, f, f_star);
						}

					}
				}
			}
		}
	}
}

template<typename T>
__global__ void calcFs2D_kernel(vec<T, 2>* u_unc_L, vec<T, 2>* F_ext_L, T* rho_L, vec<T, 2>* samples, vec<T, 2>* velocities, int kernelSize, int sampleCount, unsigned char sharpness)
{
	int sampID = blockIdx.x * blockDim.x + threadIdx.x;
	CustomAtomicAdd<T> atomAdd;
	if (sampID < sampleCount)
	{
		int x = roundf(samples[sampID][0]);
		int y = roundf(samples[sampID][1]);
		int x_loc, y_loc;

		if (x >= 0 && x < sd_dev<T, 2, 9>.gridDim_L[0] && y >= 0 && y < sd_dev<T, 2, 9>.gridDim_L[1])
		{
			vec<T, 2> Fs_lag = calc_rho_lag2D(samples[sampID], rho_L, kernelSize, sharpness)
				* (velocities[sampID] - calc_uf_lag2D(samples[sampID], u_unc_L, kernelSize, sharpness));

			for (int yk = 0; yk < kernelSize; ++yk)
			{
				for (int xk = 0; xk < kernelSize; ++xk)
				{
					int r = kernelSize / 2;
					x_loc = x - r + xk;
					y_loc = y - r + yk;

					if (x_loc >= 0 && x_loc < sd_dev<T, 2, 9>.gridDim_L[0] && y_loc >= 0 && y_loc < sd_dev<T, 2, 9>.gridDim_L[1])
					{
						int pos_L = y_loc * sd_dev<T, 2, 9>.gridDim_L[0] + x_loc;
						T delta = deltaFunc(samples[sampID], { (T)x_loc, (T)y_loc }, sharpness);
						vec<T, 2> Fs_eul_loc = Fs_lag * delta;

						if (sum(Fs_eul_loc * Fs_eul_loc) > (T)0.0)
						{
							atomAdd.AtomicAdd(&(F_ext_L[pos_L][0]), Fs_eul_loc[0]);
							atomAdd.AtomicAdd(&(F_ext_L[pos_L][1]), Fs_eul_loc[1]);
						}
					}
				}
			}
		}
	}
}
#pragma endregion
#endif