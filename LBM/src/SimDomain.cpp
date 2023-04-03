 #include "SimDomain.h"
#include <string>
#include <iostream>

template struct SimDom_Specialisation<2>;
template struct SimDom_Specialisation<3>;
template struct SimDom_Specialisation<2>;
template struct SimDom_Specialisation<3>;
template struct SimDomain<float, 2>;
template struct SimDomain<float, 3>;
template struct SimDomain<double, 2>;
template struct SimDomain<double, 3>;

template<typename T, size_t D>
SimDomain<T,D>::SimDomain(vec<T, D> GridDim, T Viscosity, T Density, T uRef, T ReferencePressure)
	:gridDim(GridDim), viscosity(Viscosity), rho(Density), uRef(uRef), p0(ReferencePressure), SimDom_Specialisation<D>(c_, w_){}


template<typename T, size_t D>
int SimDomain<T,D>::getD() const { return D; }

template<typename T, size_t D>
int SimDomain<T,D>::getQ() const { return Q; }

template<typename T, size_t D>
__device__ __host__ vec<T, D> SimDomain<T, D>::c(int i) const { return c_[i]; }

template<typename T, size_t D>
__device__ __host__ T SimDomain<T, D>::w(int i) const { return w_[i]; }

template<typename T, size_t D>
grid_size_t SimDomain<T,D>::getMaxNodeCount() const
{
	grid_size_t nodeCount = 1;

	for (int i = 0; i < D; ++i)
		nodeCount *= gridDim_L[i];

	return nodeCount;
}

template<typename T, size_t D>
T SimDomain<T,D>::getTimeStep() const { return dt; }

template<typename T, size_t D>
T SimDomain<T, D>::get_uRef_L() const { return uRef_L; }

template<typename T, size_t D>
T SimDomain<T, D>::get_uRef() const { return uRef; }

template<typename T, size_t D>
T SimDomain<T,D>::getGridDim(int axis) const 
{ 
	if(((D == 2) && (axis < 2 && axis >= 0)) ||	((D == 3) && (axis < 3 && axis >= 0)))
		return  gridDim[axis]; 
	else 
		return 0.0;
}

template<typename T, size_t D>
grid_size_t SimDomain<T,D>::getGridDim_L(int axis) const 
{ 
	if (((D == 2) && (axis < 2 && axis >= 0)) || ((D == 3) && (axis < 3 && axis >= 0)))
		return  gridDim_L[axis];
	else if (D == 2 && axis == 2)												//useful for calculating Offset by iterating the axes
		return 1;
	else
		return 0;
}

template<typename T, size_t D>
T SimDomain<T,D>::getGridSize() const { return dh; }

template<typename T, size_t D>
T SimDomain<T,D>::getRelaxationConstant() const { return r_vis; }

template<typename T, size_t D>
T SimDomain<T, D>::getViscosity_L() const { return viscosity_L; }

template<typename T, size_t D>
T SimDomain<T,D>::getRho() const { return rho; }

template<typename T, size_t D>
T SimDomain<T,D>::getC_u() const { return C_u; }

template<typename T, size_t D>
T SimDomain<T,D>::getC_p() const { return C_p; }

template<typename T, size_t D>
T SimDomain<T,D>::getC_vis() const { return C_vis; }

template<typename T, size_t D>
T SimDomain<T,D>::getC_f() const { return C_f; }

template<typename T, size_t D>
T SimDomain<T,D>::getC_F() const { return C_F; }

template<typename T, size_t D>
void SimDomain<T, D>::setZerothRelaxationTime(T r) { this->zerothRelaxationTime = r; }

template<typename T, size_t D>
void SimDomain<T, D>::setLowRelaxationTimes(T r) { this->lowRelaxationTimes = r; }

template<typename T, size_t D>
void SimDomain<T, D>::setHighRelaxationTimes(T r_3, T r_4, T r_5, T r_6) 
{ 
	this->r_3 = r_3; 
	this->r_4 = r_4;
	this->r_5 = r_5;
	this->r_6 = r_6;
}

template<typename T, size_t D>
void SimDomain<T, D>::setHighRelaxationTimes(T r)
{
	this->r_3 = r;
	this->r_4 = r;
	this->r_5 = r;
	this->r_6 = r;
}

template<typename T, size_t D>
void SimDomain<T, D>::enableLocalRelaxation(T param_0, T param_1, T param_2, T param_3)
{
	this->param_0 = param_0;
	this->param_1 = param_1;
	this->param_2 = param_2;
	this->param_3 = param_3;
	this->localRelaxation = true;
}

template<typename T, size_t D>
void SimDomain<T, D>::enableLocalRelaxation() { this->localRelaxation = true; }

template<typename T, size_t D>
void SimDomain<T,D>::setGridDim(T GridDim, int axis) { this->gridDim[axis] = GridDim; this->gridDim_L[axis] = GridDim / dh; }

template<typename T, size_t D>
void SimDomain<T,D>::setReferencePressure(T ReferencePressure) { this->p0 = ReferencePressure; }

template<typename T, size_t D>
void SimDomain<T,D>::addVelocityBound(VELOCITY_BOUND<T, D> velBound) 
{ 
	if (normilized_parameter)
		velBound.u_w *= C_u;
	velBoundList.push_back(velBound); 
}

template<typename T, size_t D>
void SimDomain<T,D>::addPressureBound(PRESSURE_BOUND<T, D> pressBound) 
{ 
	if (normilized_parameter)
		pressBound.dp_w *= C_p;
	pressBoundList.push_back(pressBound); 
}

template<typename T, size_t D>
const VELOCITY_BOUND<T, D>& SimDomain<T, D>::getVelBound(int idx) const
{
	return velBoundList.at(idx);
}

template<typename T, size_t D>
const PRESSURE_BOUND<T, D>& SimDomain<T, D>::getPressBound(int idx) const
{
	return pressBoundList.at(idx);
}

template<typename T, size_t D>
int SimDomain<T, D>::getVelBoundCount() const { return velBoundList.size(); }

template<typename T, size_t D>
int SimDomain<T, D>::getPressBoundCount() const { return pressBoundList.size(); }

//template<typename T, size_t D>
//void SimDomain<T, D>::enableLocalRelaxation(T param_0, T param_1, T param_2, T param_3)
//{
//	this->param_0 = param_0;
//	this->param_1 = param_1;
//	this->param_2 = param_2;
//	this->localRelaxation = true;
//}

template<typename T, size_t D>
bool SimDomain<T, D>::isLocalRelaxationEnabled() { return this->localRelaxation; }

template<typename T, size_t D>
void SimDomain<T, D>::copyDataToStruct(SimDomain_dev<T,D,Q>& simDom_dev)
{
	simDom_dev.c = c_;
	simDom_dev.w = w_;
	simDom_dev.gridDim_L = gridDim_L;

	for (int i=0; i< velBoundList.size(); ++i)
	{
		velBoundList[i].copyToDev(simDom_dev.velBounds[i]);
	}

	simDom_dev.velBound_Count = velBoundList.size();

	for (int i = 0; i < pressBoundList.size(); ++i)
	{
		 pressBoundList[i].copyToDev(simDom_dev.pressBounds[i]);
	}

	simDom_dev.pressBound_Count = pressBoundList.size();
	simDom_dev.maxNodeCount = getMaxNodeCount();

	simDom_dev.zerothRelaxationTime = this->zerothRelaxationTime;
	simDom_dev.lowRelaxationTimes = this->lowRelaxationTimes;
	simDom_dev.r_3 = this->r_3;
	simDom_dev.r_4 = this->r_4;
	simDom_dev.r_5 = this->r_5;
	simDom_dev.r_6 = this->r_6;
	simDom_dev.param_0 = this->param_0;
	simDom_dev.param_1 = this->param_1;
	simDom_dev.param_2 = this->param_2;
	simDom_dev.param_3 = this->param_3;
	simDom_dev.localRelaxation = this->localRelaxation;
}

template<typename T, size_t D>
bool SimDomain<T, D>::calcConvFactors(T tau_L, int gridDimX_L)
{
	this->r_vis = (T)1.0/tau_L;
	this->gridDim_L[0] = gridDimX_L;
	dh = gridDim[0] / static_cast<T>(gridDim_L[0]);
	C_u = uRef / uRef_L;
	 
	viscosity_L = cs_sq<T> * (tau_L - 0.5);
	C_vis = dh * C_u;

	for (int i = 1; i < D; ++i)
		this->gridDim_L[i] = static_cast<grid_size_t>(gridDim[i] / dh);

	dt = dh / C_u;
	C_p = rho * dh * dh / (dt * dt);
	C_f = rho * dh * dh * dh * dh / (dt * dt);
	C_F = rho * dh / (dt * dt);

	if (tau_L <= (0.5 + 0.125 * uRef_L))
		return false;
	return true;
}

template<typename T, size_t D>
bool SimDomain<T, D>::calcConvFactors(int gridDimX_L)
{
	this->gridDim_L[0] = gridDimX_L;
	dh = gridDim[0] / static_cast<T>(gridDim_L[0]);
	C_u = uRef / uRef_L;
	dt = dh / C_u;
	T dt_min = (T)((dh / uRef) * 10e-2);
	T dt_max = (T)((0.5 * sqrtf(1.f / 3.f)) * dh / uRef);

	if (dt < dt_min || dt > dt_max)
		throw std::runtime_error("Scale out of Bound");

	C_vis = dh * C_u;

	viscosity_L = viscosity / C_vis;
	this->r_vis = (T)1.0 / ((T)3.0 * viscosity_L + (T)0.5);

	for (int i = 1; i < D; ++i)
		this->gridDim_L[i] = static_cast<grid_size_t>(gridDim[i] / dh);

	C_p = rho * dh * dh / (dt * dt);
	C_f = rho * dh * dh * dh * dh / (dt * dt);
	C_F = rho * dh / (dt * dt);

	return true;
}

//template<typename T, size_t D>
//void SimDomain<T, D>::rescaleConFactors(T uMax_L, T& scale_u, T& scale_F)
//{
//	T C_u_new = C_u * uMax_L / uRef_L;
//	if (C_u_new < uRef / uRef_L)
//		C_u_new = uRef / uRef_L;
//
//	scale_u = C_u/C_u_new;
//	C_u = C_u_new;
//	C_vis = dh * C_u;
//	dt = dh / C_u;
//	C_p = rho * dh * dh / (dt * dt);
//	C_f = rho * dh * dh * dh * dh / (dt * dt);
//	T C_F_new = rho * dh / (dt * dt);
//	scale_F = C_F / C_F_new;
//	C_F = C_F_new;
//}


template<typename T, size_t D>
void SimDomain<T, D>::rescaleConFactors(T uMax_L, T& scale_u, T& scale_F)
{
	T uRef_new = C_u * uMax_L;
	T q = 1.0;
	if (uRef_new < uRef * q)
		uRef_new = uRef * q;

	scale_u = C_u * uRef_L / uRef_new;
	C_u = uRef_new / uRef_L;
	dt = dh / C_u;
	T dt_min = (T)((dh / uRef_new) * 10e-2);
	T dt_max = (T)((0.5 * sqrtf(1.f / 3.f)) * dh / uRef_new);

	if (dt < dt_min)
	{
		dt = dt_min;
		C_u = dh / dt;
		scale_u = C_u * uRef_L / uRef_new;
	}
		
	if (dt > dt_max)
	{
		dt = dt_max;
		C_u = dh / dt;
		scale_u = C_u * uRef_L / uRef_new;
	}

	C_vis = dh * C_u;
	viscosity_L = viscosity / C_vis;
	this->r_vis = (T)1.0 / ((T)3.0 * viscosity_L + (T)0.5);


	C_p = rho * dh * dh / (dt * dt);
	C_f = rho * dh * dh * dh * dh / (dt * dt);
	T C_F_new = rho * dh / (dt * dt);
	scale_F = C_F / C_F_new;
	C_F = C_F_new;
}