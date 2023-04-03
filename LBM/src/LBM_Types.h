#ifndef LBM_TYP
#define LBM_TYP
#include <memory>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "location_ptr/array.h"
#include "location_ptr/locationPtr.h"
#include "location_ptr/arrayMath.h"

#define L_UNIT_OUTPUT 1
#define L_UNIT_POS 2
#define L_UNIT_TIME 4

constexpr bool normilized_parameter = false;
#define init_R_3D 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
#define init_R_2D 1.0, 0.0, 0.0, 1.0

#define SHARP 0
#define	MEDIUM_SMOOTH 1
#define SMOOTH 2

#define AoS 
//#define SoA 


#if defined(AoS) && ! defined(SoA)
#define dataLayout(Q, alpha, pos, i)(Q * pos + i)
#endif

#if defined(SoA) && ! defined(AoS)
#define dataLayout(Q, alpha, pos, i)(alpha * i + pos)
#endif

#if ! defined(SoA) && ! defined(AoS) || defined(SoA) && defined(AoS)
#define dataLayout(Q, alpha, pos, i)(Q * alpha * (pos / alpha) + alpha * i + pos % alpha)
#endif

#define square(a) ((a) * (a))

using grid_size_t = unsigned int;

template<typename T>
constexpr T cs_sq =  1.0 / 3.0;

template<typename T>
constexpr T cs_sq_init =  1.0 / 3.0;

template<typename T>
constexpr T PI = 3.14159265358979323846;

template<typename T>
constexpr T minMass = 0.0000001;

template<typename T, size_t D>
using vec = array<T,D>;

//fpr short and static Vector Fields like c_i
template<typename T, size_t D, size_t Q>
using vec_set = vec<vec<T, D>, Q>;

template<typename location,typename T, size_t D>
using vec_field = location_ptr<location, vec<T, D>>;

template<typename location, typename T>
using scal_field = location_ptr<location, T>;

template<typename T>
vec<T, 9> invert3x3Mat(const vec<T, 9>& M)
{
	vec<T,9> M_inv;
	M_inv[0] = (M[4] * M[8] - M[5] * M[7]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[1] = -(M[1] * M[8] - M[2] * M[7]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[2] = (M[1] * M[5] - M[2] * M[4]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[3] = -(M[3] * M[8] - M[5] * M[6]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[4] = (M[0] * M[8] - M[2] * M[6]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[5] = -(M[0] * M[5] - M[2] * M[3]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[6] = (M[3] * M[7] - M[4] * M[6]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[7] = -(M[0] * M[7] - M[1] * M[6]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	M_inv[8] = (M[0] * M[4] - M[1] * M[3]) / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
	
	return M_inv;
}

template<typename T>
vec<T, 4> invert2x2Mat(const vec<T, 4>& M)
{
	vec<T, 4> M_inv;
	M_inv[0] = M[3] / (M[0] * M[3] - M[1] * M[2]);
	M_inv[1] = -M[1] / (M[0] * M[3] - M[1] * M[2]);
	M_inv[2] = -M[2] / (M[0] * M[3] - M[1] * M[2]);
	M_inv[3] = M[0] / (M[0] * M[3] - M[1] * M[2]);

	return M_inv;
}

template <typename T, int D>
struct CustomVecLength
{
	__device__ __host__ T length(vec<T, D> v)
	{
		extern __device__ __host__ void error(void);
		error(); // Ensure that we won't compile any un-specialized types
		return NULL;
	}
};

template <int D>
struct CustomVecLength <float, D>
{
	__device__ __host__ float length(vec<float, D> v)
	{
		return std::sqrtf(sum(v * v));
	}
};

template <int D>
struct CustomVecLength <double, D>
{
	__device__ __host__ double length(vec<double, D> v)
	{
		return std::sqrt(sum(v * v));
	}
};

template <typename T>
struct CustomSqrt
{
	__device__ __host__ T sqrt(T vel)
	{
		extern __device__ __host__ void error(void);
		error(); // Ensure that we won't compile any un-specialized types
		return NULL;
	}
};

template <>
struct CustomSqrt<float>
{
	__device__ __host__ float sqrt(float vel)
	{
		return std::sqrtf(vel);
	}
};

template <>
struct CustomSqrt<double>
{
	__device__ __host__ double sqrt(double vel)
	{
		return std::sqrt(vel);
	}
};



template<typename T, size_t D>
__device__ __host__ vec<T, D> operator*(vec<T, D* D> M, vec<T, D> v)
{
	vec<T, D> v_out{};

	for (int i = 0; i < D; ++i)
	{
		for (int k = 0; k < D; ++k)
		{
			v_out[i] += M[i * D + k] * v[k];
		}
	}

	return v_out;
}

template<typename T>
__device__ __host__ vec<T, 2> bilinearVelInterpolation(vec<T, 2> partPos, vec<T, 2>* u, int gridDimX_L)
{
	int x_pos = static_cast<int>(partPos[0]);
	int y_pos = static_cast<int>(partPos[1]);
	T x = x_pos;
	T y = y_pos;

	vec<T, 2> int_u = (y + (T)1.0 - partPos[1])*((x + (T)1.0 - partPos[0]) * u[y_pos * gridDimX_L + x_pos] + (partPos[0] - x) * u[y_pos * gridDimX_L + x_pos + 1])
		+ (partPos[1] - y)*((x + (T)1.0 - partPos[0]) * u[(y_pos + 1) * gridDimX_L + x_pos] + (partPos[0] - x) * u[(y_pos + 1) * gridDimX_L + x_pos + 1]);
	return int_u;
}

template<typename T>
__device__ __host__ T bilinearDensInterpolation(vec<T, 2> partPos, T* rho, int gridDimX_L)
{
	int x_pos = static_cast<int>(partPos[0]);
	int y_pos = static_cast<int>(partPos[1]);
	T x = x_pos;
	T y = y_pos;

	T int_rho = (y + (T)1.0 - partPos[1]) * ((x + (T)1.0 - partPos[0]) * rho[y_pos * gridDimX_L + x_pos]	+ (partPos[0] - x) * rho[y_pos * gridDimX_L + x_pos + 1])
		+ (partPos[1] - y)*((x + (T)1.0 - partPos[0]) * rho[(y_pos + 1) * gridDimX_L + x_pos] + (partPos[0] - x) * rho[(y_pos + 1) * gridDimX_L + x_pos + 1]);
	return int_rho;
}

template<typename T>
__device__ __host__ vec<T, 3> trilinearVelInterpolation(vec<T, 3> partPos, vec<T, 3>* u, int gridDimX_L, int gridDimY_L, int gridDimZ_L)
{
	int x_pos = static_cast<int>(partPos[0]);
	int y_pos = static_cast<int>(partPos[1]);
	int z_pos = static_cast<int>(partPos[2]);

	T x_d = partPos[0] - x_pos;
	T y_d = partPos[1] - y_pos;
	T z_d = partPos[2] - z_pos;

	vec<T, 3> u000{}, u100{},
		u001{}, u101{},
		u010{}, u110{},
		u011{}, u111{};

	
	if(z_pos < gridDimZ_L-1)
		u000 = u[((z_pos + 1) * gridDimY_L + y_pos) * gridDimX_L + x_pos];

	if(z_pos < gridDimZ_L - 1 && x_pos < gridDimX_L - 1)
		u100 = u[((z_pos + 1) * gridDimY_L + y_pos) * gridDimX_L + x_pos + 1];

	if (z_pos < gridDimZ_L - 1 && y_pos < gridDimY_L - 1)
		u001 = u[((z_pos + 1) * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos];

	if (z_pos < gridDimZ_L - 1 && y_pos < gridDimY_L - 1 && x_pos < gridDimX_L - 1)
		u101 = u[((z_pos + 1) * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos + 1];

	u010 = u[(z_pos * gridDimY_L + y_pos) * gridDimX_L + x_pos];

	if (x_pos < gridDimX_L - 1)
		u110 = u[(z_pos * gridDimY_L + y_pos) * gridDimX_L + x_pos + 1];

	if (y_pos < gridDimY_L - 1)
		u011 = u[(z_pos * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos];

	if (x_pos < gridDimX_L - 1)
		u111 = u[(z_pos * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos + 1];

	vec<T, 3> u00 = u000 * ((T)1.0 - x_d) + u100  * x_d;
	vec<T, 3> u01 = u001 * ((T)1.0 - x_d) + u101 * x_d;
	vec<T, 3> u10 = u010 * ((T)1.0 - x_d) + u110 * x_d;
	vec<T, 3> u11 = u011 * ((T)1.0 - x_d) + u111 * x_d;

	vec<T, 3> u0 = u00 * (1 - y_d) + u10 * y_d;
	vec<T, 3> u1 = u01 * (1 - y_d) + u11 * y_d;


	return  u0* (1 - z_d) + u1 * z_d;
}

template<typename T>
__device__ __host__ T trilinearDensInterpolation(vec<T, 3> partPos, T* rho, int gridDimX_L, int gridDimY_L, int gridDimZ_L)
{
	int x_pos = static_cast<int>(partPos[0]);
	int y_pos = static_cast<int>(partPos[1]);
	int z_pos = static_cast<int>(partPos[2]);

	T x_d = partPos[0] - x_pos;
	T y_d = partPos[1] - y_pos;
	T z_d = partPos[2] - z_pos;

	T rho000{}, rho100{},
		rho001{}, rho101{},
		rho010{}, rho110{},
		rho011{}, rho111{};


	if (z_pos < gridDimZ_L - 1)
		rho000 = rho[((z_pos + 1) * gridDimY_L + y_pos) * gridDimX_L + x_pos];

	if (z_pos < gridDimZ_L - 1 && x_pos < gridDimX_L - 1)
		rho100 = rho[((z_pos + 1) * gridDimY_L + y_pos) * gridDimX_L + x_pos + 1];

	if (z_pos < gridDimZ_L - 1 && y_pos < gridDimY_L - 1)
		rho001 = rho[((z_pos + 1) * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos];

	if (z_pos < gridDimZ_L - 1 && y_pos < gridDimY_L - 1 && x_pos < gridDimX_L - 1)
		rho101 = rho[((z_pos + 1) * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos + 1];

	rho010 = rho[(z_pos * gridDimY_L + y_pos) * gridDimX_L + x_pos];

	if (x_pos < gridDimX_L - 1)
		rho110 = rho[(z_pos * gridDimY_L + y_pos) * gridDimX_L + x_pos + 1];

	if (y_pos < gridDimY_L - 1)
		rho011 = rho[(z_pos * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos];

	if (x_pos < gridDimX_L - 1)
		rho111 = rho[(z_pos * gridDimY_L + y_pos + 1) * gridDimX_L + x_pos + 1];

	T rho00 = rho000 * ((T)1.0 - x_d) + rho100 * x_d;
	T rho01 = rho001 * ((T)1.0 - x_d) + rho101 * x_d;
	T rho10 = rho010 * ((T)1.0 - x_d) + rho110 * x_d;
	T rho11 = rho011 * ((T)1.0 - x_d) + rho111 * x_d;
	
	T rho0 = rho00 * (1 - y_d) + rho10 * y_d;
	T rho1 = rho01 * (1 - y_d) + rho11 * y_d;


	return rho0 * (1 - z_d) + rho1 * z_d;
}

template<typename T, size_t D>
T skalarProd(vec<T, D* D>& M_1, vec<T, D* D>& M_2, int col1, int col2)
{
	T length{};
	for (int n = 0; n < D; ++n)
		length += M_1[D * n + col1] * M_2[D * n + col2];

	return length;
}

template<typename T, size_t D>
T getNorm(vec<T, D * D>& M, int col)
{
	CustomSqrt<T> customSqrt;
	T length{};
	for (int n = 0; n < D; ++n)
		length += M[n * D + col] * M[n * D + col];

	return customSqrt.sqrt(length);
}

template<typename T, size_t D>
vec<T, D* D> grammSchmidt(vec<T, D* D> M)
{
	vec<T, D* D> Q{};
	for (int k = 0; k < D; ++k)
	{
		for (int n = 0; n < D; ++n)
			Q[n * D + k] = M[n * D + k];

		for (int j = 0; j < k; ++j)
			for (int n = 0; n < D; ++n)
				Q[n * D + k] -= skalarProd<T, D>(Q, M, j, k) * Q[n * D + j];

		T norm = getNorm<T, D>(Q, k);
		for (int n = 0; n < D; ++n)
			Q[n * D + k] = Q[n * D + k] / norm;
	}
	return Q;
}

template <typename T, size_t D>
__device__ __host__ T  deltaFunc(vec<T, D> p1, vec<T, D> p2, unsigned char sharpness)
{
	vec<T, D> r{}, dr{};
	CustomSqrt<T> customSqrt;

	for (int i = 0; i < D; ++i)
		r[i] = abs(p1[i] - p2[i]);


	if (sharpness == SHARP)
	{
		for (int i = 0; i < D; ++i)
			if (r[i] >= 0.0 && r[i] < 1.0)
				dr[i] = 1.0 - r[i];
	}
	else if (sharpness == MEDIUM_SMOOTH)
	{
		for (int i = 0; i < D; ++i)
		{
			if (r[i] >= 0.0 && r[i] < 0.5)
				dr[i] = (1.0 / 3.0) * (1.0 + customSqrt.sqrt(-3.0 * r[i] * r[i] + 1.0));
			else if (r[i] >= 0.5 && r[i] < 1.5)
				dr[i] = (1.0 / 6.0) * (5.0 - 3.0 * r[i] - customSqrt.sqrt(-3.0 * (1.0 - r[i]) * (1.0 - r[i]) + 1.0));
		}
	}
	else
	{
		for (int i = 0; i < D; ++i)
		{
			if (r[i] >= 0.0 && r[i] < 1.0)
				dr[i] = (1.0 / 8.0) * (3.0 - 2.0 * r[i] + customSqrt.sqrt(1.0 + 4.0 * r[i] - 4.0 * r[i] * r[i]));
			else if (r[i] >= 1.0 && r[i] < 2.0)
				dr[i] = (1.0 / 8.0) * (5.0 - 2.0 * r[i] - customSqrt.sqrt(-7.0 + 12.0 * r[i] - 4.0 * r[i] * r[i]));
		}
	}

	T delVel = 1.0;

	for (int i = 0; i < D; ++i)
		delVel *= dr[i];

	return delVel;
}

template<typename T, size_t D>
__device__ __host__ vec<T, D*D> matMult(vec<T, D* D> M_1, vec<T, D * D> M_2)
{
	vec<T, D* D> M_out{};

	for (int i = 0; i < D; ++i)
	{
		for (int j = 0; j < D; ++j)
		{
			for (int k = 0; k < D; ++k)
			{
				M_out[i * D + j] += M_1[i * D + k] * M_2[k * D + j];
			}
		}
	}

	return M_out;
}

template<typename T>
__device__ T calc_f_eq_CM(int i, T rho, vec<T, 2> u)
{
	T ux = u[0], uy = u[1];
	switch (i)
	{
	case 0:
		return rho / 9.0 + rho * ((ux * ux) / 2.0 + (uy * uy) / 2.0 - 1.0) * (2.0 / 3.0) + rho * ((ux * ux) * (uy * uy) - ux * ux - uy * uy + 1.0);
	case 1:
		return rho * (-1.0 / 1.8E+1) - rho * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) * (2.0 / 3.0) + rho * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * ux) / 2.0);
	case 2:
		return rho * (-1.0 / 1.8E+1) - rho * (uy / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) * (2.0 / 3.0) + rho * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * uy) / 2.0);
	case 3:
		return rho * (-1.0 / 1.8E+1) + rho * (ux / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) * (2.0 / 3.0) - rho * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * ux) / 2.0);
	case 4:
		return rho * (-1.0 / 1.8E+1) + rho * (uy / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) * (2.0 / 3.0) - rho * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * uy) / 2.0);
	case 5:
		return rho / 3.6E+1 + rho * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0);
	case 6:
		return rho / 3.6E+1 + rho * (ux * (-1.0 / 8.0) + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0);
	case 7:
		return rho / 3.6E+1 - rho * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0);
	case 8:
		return rho / 3.6E+1 + rho * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0);
	default:
		return 0;
	}
}

template<typename T>
__device__ T calc_f_eq_CM(int i, T rho, vec<T, 3> u)
{
	T ux = u[0], uy = u[1], uz = u[2];
	switch (i)
	{
	case 0:
		return rho * (-1.0 / 2.7E+1) + rho * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) + (uy * uy) * (uz * uz) - ux * ux - uy * uy - uz * uz - (ux * ux) * (uy * uy) * (uz * uz) + 1.0) - (rho * ((ux * ux) * (-1.0 / 2.0) + (uy * uy) / 4.0 + (uz * uz) / 4.0)) / 9.0 - (rho * ((ux * ux) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0)) / 3.0 - rho * (((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 3.0 - (ux * ux) * (2.0 / 3.0) - (uy * uy) * (2.0 / 3.0) - (uz * uz) * (2.0 / 3.0) + 1.0);
	case 1:
		return rho / 5.4E+1 + rho * (ux * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - (rho * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 4.0)) / 9.0 + (rho * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0)) / 3.0 + rho * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
	case 2:
		return rho / 5.4E+1 + rho * (ux / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + (rho * (ux * (-1.0 / 4.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0)) / 3.0 + (rho * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0)) / 9.0 - rho * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 - (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
	case 3:
		return rho / 5.4E+1 + rho * (uy * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uy) / 6.0 + (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + (rho * (uy / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uy / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 + rho * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
	case 4:
		return rho / 5.4E+1 + rho * (uy / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 6.0 - (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + (rho * (uy * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uy * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 - rho * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 - (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
	case 5:
		return rho / 5.4E+1 + rho * (uz * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) + (rho * (uz / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uz / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 + rho * (uz / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
	case 6:
		return rho / 5.4E+1 + rho * (uz / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) + (rho * (uz * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uz * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 - rho * (uz / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
	case 7:
		return rho * (-1.0 / 1.08E+2) - rho * (ux * (-1.0 / 1.2E+1) - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + (rho * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 - (rho * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 8:
		return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) - (rho * (ux / 8.0 - uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + (rho * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 9:
		return rho * (-1.0 / 1.08E+2) - rho * (ux * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) + (rho * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 - (rho * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 10:
		return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (ux / 8.0 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + (rho * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 11:
		return rho * (-1.0 / 1.08E+2) - rho * (uy * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 9.0 - (rho * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 3.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 12:
		return rho * (-1.0 / 1.08E+2) - rho * (uy / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) + (rho * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 3.0 + (rho * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 9.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 13:
		return rho * (-1.0 / 1.08E+2) + rho * (ux / 1.2E+1 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) - (rho * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + (rho * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 14:
		return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) - (rho * (ux * (-1.0 / 8.0) + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 - (rho * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 15:
		return rho * (-1.0 / 1.08E+2) + rho * (ux / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) - (rho * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + (rho * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 16:
		return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (ux * (-1.0 / 8.0) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 - (rho * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 17:
		return rho * (-1.0 / 1.08E+2) + rho * (uy / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 1.2E+1 + (uz * uz) / 1.2E+1) - (rho * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 9.0 - (rho * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 3.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 18:
		return rho * (-1.0 / 1.08E+2) - rho * (uy / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 9.0 - (rho * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 3.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
	case 19:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 20:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + (rho * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 - (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 3.0 + rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 21:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - (rho * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 9.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 22:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + (rho * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 + rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 23:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 - (rho * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 24:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 25:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	case 26:
		return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + (rho * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 - (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
	default:
		return 0;
	}
}

template<typename T, size_t D>
__device__ __host__ T planeEquation(vec<T, D> P, vec<T, D> P0, vec<T, D> n)
{
	return dot(n , (P - P0));
}
#endif

