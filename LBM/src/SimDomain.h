#pragma once
#ifndef SIM_DOMAIN
#define SIM_DOMAIN
#include <stdexcept>
#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include "LBM_Types.h"
#include "LBM_DomainBoundary.h"

template<typename T, int D, int Q>
struct SimDomain_dev {
	vec_set<T, D, Q> c;		//discrete velocity-set
	vec<T, Q> w;			//discrete weights
	vec<grid_size_t, D> gridDim_L;
	T zerothRelaxationTime;
	T lowRelaxationTimes;
	T r_3, r_4, r_5, r_6;


	//local high order relaxation
	T param_0, param_1, param_2, param_3;
	bool localRelaxation;

	vec<VELOCITY_BOUND_DEV<T, D>, 2*D> velBounds;
	vec<PRESSURE_BOUND_DEV<T, D>, 2*D> pressBounds;

	int velBound_Count;
	int pressBound_Count;
	int maxNodeCount;
	int alpha;
};

template<size_t D> struct SimDom_Specialisation;

#pragma region simDom_spec_2D
template<>
struct SimDom_Specialisation<2>
{
	static constexpr size_t Q = 9;

	template<typename T>
	SimDom_Specialisation(vec_set<T, 2, Q>& c_, vec<T, Q>& w_)
	{
		w_[0] = 4.0 / 9.0;
		w_[1] = 1.0 / 9.0;
		w_[2] = 1.0 / 9.0;
		w_[3] = 1.0 / 9.0;
		w_[4] = 1.0 / 9.0;
		w_[5] = 1.0 / 36.0;
		w_[6] = 1.0 / 36.0;
		w_[7] = 1.0 / 36.0;
		w_[8] = 1.0 / 36.0;

		c_[0][0] = 0.0;
		c_[0][1] = 0.0;

		c_[1][0] = 1.0;
		c_[1][1] = 0.0;

		c_[2][0] = 0.0;
		c_[2][1] = 1.0;

		c_[3][0] = -1.0;
		c_[3][1] = 0.0;

		c_[4][0] = 0.0;
		c_[4][1] = -1.0;

		c_[5][0] = 1.0;
		c_[5][1] = 1.0;

		c_[6][0] = -1.0;
		c_[6][1] = 1.0;

		c_[7][0] = -1.0;
		c_[7][1] = -1.0;

		c_[8][0] = 1.0;
		c_[8][1] = -1.0;
	}
};
#pragma endregion

template<>
struct SimDom_Specialisation<3>
{
	static constexpr size_t Q = 27;

	template<typename T>
	SimDom_Specialisation(vec_set<T, 3, Q>& c_, vec<T, Q>& w_)
	{
		w_[0] = 8.0 / 27.0;
		w_[1] = 2.0 / 27.0;
		w_[2] = 2.0 / 27.0;
		w_[3] = 2.0 / 27.0;
		w_[4] = 2.0 / 27.0;
		w_[5] = 2.0 / 27.0;
		w_[6] = 2.0 / 27.0;
		w_[7] = 1.0 / 54.0;
		w_[8] = 1.0 / 54.0;
		w_[9] = 1.0 / 54.0;
		w_[10] = 1.0 / 54.0;
		w_[11] = 1.0 / 54.0;
		w_[12] = 1.0 / 54.0;
		w_[13] = 1.0 / 54.0;
		w_[14] = 1.0 / 54.0;
		w_[15] = 1.0 / 54.0;
		w_[16] = 1.0 / 54.0;
		w_[17] = 1.0 / 54.0;
		w_[18] = 1.0 / 54.0;
		w_[19] = 1.0 / 216.0;
		w_[20] = 1.0 / 216.0;
		w_[21] = 1.0 / 216.0;
		w_[22] = 1.0 / 216.0;
		w_[23] = 1.0 / 216.0;
		w_[24] = 1.0 / 216.0;
		w_[25] = 1.0 / 216.0;
		w_[26] = 1.0 / 216.0;

		c_[0][0] = 0.0;
		c_[0][1] = 0.0;
		c_[0][2] = 0.0;

		c_[1][0] = 1.0;
		c_[1][1] = 0.0;
		c_[1][2] = 0.0;

		c_[2][0] = -1.0;
		c_[2][1] = 0.0;
		c_[2][2] = 0.0;

		c_[3][0] = 0.0;
		c_[3][1] = 1.0;
		c_[3][2] = 0.0;

		c_[4][0] = 0.0;
		c_[4][1] = -1.0;
		c_[4][2] = 0.0;

		c_[5][0] = 0.0;
		c_[5][1] = 0.0;
		c_[5][2] = 1.0;

		c_[6][0] = 0.0;
		c_[6][1] = 0.0;
		c_[6][2] = -1.0;

		c_[7][0] = 1.0;
		c_[7][1] = 1.0;
		c_[7][2] = 0.0;

		c_[8][0] = -1.0;
		c_[8][1] = -1.0;
		c_[8][2] = 0.0;

		c_[9][0] = 1.0;
		c_[9][1] = 0.0;
		c_[9][2] = 1.0;

		c_[10][0] = -1.0;
		c_[10][1] = 0.0;
		c_[10][2] = -1.0;

		c_[11][0] = 0.0;
		c_[11][1] = 1.0;
		c_[11][2] = 1.0;

		c_[12][0] = 0.0;
		c_[12][1] = -1.0;
		c_[12][2] = -1.0;

		c_[13][0] = 1.0;
		c_[13][1] = -1.0;
		c_[13][2] = 0.0;

		c_[14][0] = -1.0;
		c_[14][1] = 1.0;
		c_[14][2] = 0.0;

		c_[15][0] = 1.0;
		c_[15][1] = 0.0;
		c_[15][2] = -1.0;

		c_[16][0] = -1.0;
		c_[16][1] = 0.0;
		c_[16][2] = 1.0;

		c_[17][0] = 0.0;
		c_[17][1] = 1.0;
		c_[17][2] = -1.0;

		c_[18][0] = 0.0;
		c_[18][1] = -1.0;
		c_[18][2] = 1.0;

		c_[19][0] = 1.0;
		c_[19][1] = 1.0;
		c_[19][2] = 1.0;

		c_[20][0] = -1.0;
		c_[20][1] = -1.0;
		c_[20][2] = -1.0;

		c_[21][0] = 1.0;
		c_[21][1] = 1.0;
		c_[21][2] = -1.0;

		c_[22][0] = -1.f;
		c_[22][1] = -1.f;
		c_[22][2] = 1.f;

		c_[23][0] = 1.0;
		c_[23][1] = -1.0;
		c_[23][2] = 1.0;

		c_[24][0] = -1.0;
		c_[24][1] = 1.0;
		c_[24][2] = -1.0;

		c_[25][0] = -1.0;
		c_[25][1] = 1.0;
		c_[25][2] = 1.0;

		c_[26][0] = 1.0;
		c_[26][1] = -1.0;
		c_[26][2] = -1.0;
	}
};

template<typename T, size_t D>
class SimDomain: protected SimDom_Specialisation<D>
{
public:
	static constexpr size_t Q = SimDom_Specialisation<D>::Q;
protected:
	//Latice Unit
	vec<grid_size_t, D> gridDim_L;
	T uRef_L = (T)0.2;
	T viscosity_L;
	T r_vis = (T)1.0;
	T zerothRelaxationTime = 1.0;
	T lowRelaxationTimes = 1.0;
	T r_3 = (T)1.94175, 
		r_4 = (T)1.91939, 
		r_5 = (T)1.89753, 
		r_6 = (T)1.8868;

	//local high order relaxation
	T param_0 = (T)0.0003, 
		param_1 = (T)(-0.00775), 
		param_2 = (T)0.00016, 
		param_3 = (T)0.0087;
	bool localRelaxation = false;


	//Physical Unit
	vec<T, D> gridDim;		//[m]				
	T dt;							//[s]
	T dh;							//[m]
	T rho;							//[kg/m^3]
	T p0;							//[Pa]
	T uRef;							//[m/s]
	T viscosity;					//[m^2/s] 

	
	//Conversion factors
	T C_u, C_p, C_f, C_F, C_vis;  //  [m/s], [Pa]=[kg/(m*s^2)], [N]=[kg*m/s^2], [N/m^3], [m^2/s] , [kg

	vec_set<T,D,Q> c_;	//discrete velocity-set
	vec<T,Q> w_;			//discrete weights

	//Inlets:
	std::vector<VELOCITY_BOUND<T,D>> velBoundList;
	//Outlets:
	std::vector<PRESSURE_BOUND<T,D>> pressBoundList;
	
public:
	SimDomain(vec<T, D> gridDim, T Viscosity, T Density, T uRef, T ReferencePressure = 1.0);
	int getD() const;
	__device__ __host__ int getQ() const;
	grid_size_t getMaxNodeCount() const;
	T getTimeStep() const;
	T getGridDim(int axis = 0) const;
	grid_size_t getGridDim_L(int axis = 0) const;
	T getGridSize() const;
	T getRelaxationConstant() const;
	T getViscosity_L() const;
	T getRho() const;
	T getC_u() const;
	T get_uRef_L() const;
	T get_uRef() const;
	T getC_p() const;
	T getC_vis() const;
	T getC_f() const;
	T getC_F() const;
	void setZerothRelaxationTime(T r);
	void setLowRelaxationTimes(T r);
	void setHighRelaxationTimes(T r_3, T r_4, T r_5, T r_6);
	void setHighRelaxationTimes(T r);
	void enableLocalRelaxation(T param_0, T param_1, T param_2, T param_3);
	void enableLocalRelaxation();
	vec<T, D> c(int i) const;
	__device__ __host__ T w(int i) const;

	void setGridDim(T GridSize, int axis);
	void setReferencePressure(T ReferencePressure);

	const VELOCITY_BOUND<T, D>& getVelBound(int idx) const;
	const PRESSURE_BOUND<T, D>& getPressBound(int idx) const;

	void addVelocityBound(VELOCITY_BOUND<T,D> velBound);
	void addPressureBound(PRESSURE_BOUND<T,D> pressBound);

	//void enableLocalRelaxation(T param_0, T param_1, T param_2, T param_3);
	bool isLocalRelaxationEnabled();


	int getVelBoundCount() const;
	int getPressBoundCount() const;

	void copyDataToStruct(SimDomain_dev<T, D, Q>& simDom_dev);
	// BGK is stable if (Krüger et al.):
	//		- tau/dt >= 0.5
	//		- |u| <=sqrt(1/3)* dx/dt for tau/dt -> 0.5	
	//		   or |u| <=sqrt(2/3)* dx/dt for tau/dt >= 1.0
	// If BKS is stable, each f_i should be > 0.0

	//in my case it works if: 
	//   -2D: |u| <= 0.51 and tau/dt >=1.0
	//   -3D: |u| <= 0.47 and tau/dt >=1.0


	bool calcConvFactors(T tau_L, int gridDimX_L);
	bool calcConvFactors(int gridDimX_L);
	void rescaleConFactors(T uMax_L, T &scale_u, T &scale_F);

};

//typedefs


template<typename T>
using SimDomain2D = SimDomain<T, 2>;

template<typename T>
using SimDomain3D = SimDomain<T, 3>;
#endif