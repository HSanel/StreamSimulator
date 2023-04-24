#include "SimDomain.h"

SimDomain::SimDomain(std::array <float, 3> GridDim, float Viscosity, float Density, float uRef, float ReferencePressure)
	:gridDim(GridDim), viscosity(Viscosity), rho(Density), uRef(uRef), p0(ReferencePressure)
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

unsigned int SimDomain::getMaxNodeCount() const 
{
	unsigned int nodeCount = 1;

	for (int i = 0; i < D; ++i)
		nodeCount *= gridDim_L[i];

	return nodeCount;
}

float SimDomain::getGridDim(const unsigned int axis) const
{
	if (axis < 3)
		return gridDim[axis];
	else 
		return 0.f;
}

unsigned int SimDomain::getGridDim_L(const unsigned int axis) const
{
	if (axis < 3)
		return gridDim_L[axis];
	else
		return 0;
}

float SimDomain::getGridSize() const { return dh; }

float SimDomain::getRelaxationConstant() const { return r_vis; }

float SimDomain::getViscosity_L() const { return viscosity_L; }

float SimDomain::getRho() const { return rho; }

float SimDomain::getC_u() const { return C_u; }

float SimDomain::get_uRef_L() const { return uRef_L; }

float SimDomain::get_uRef() const { return uRef; }

float SimDomain::getC_p() const { return C_p; }

float SimDomain::getC_vis() const { return C_vis; }

float SimDomain::getC_f() const { return C_f; }

float SimDomain::getC_F() const { return C_F; }

bool SimDomain::calcConvFactors(int gridDimX_L)
{
	this->gridDim_L[0] = gridDimX_L;
	dh = gridDim[0] / static_cast<float>(gridDim_L[0]);
	C_u = uRef / uRef_L;
	dt = dh / C_u;
	float dt_min = static_cast<float>((dh / uRef) * 10e-2);
	float dt_max = static_cast<float>((0.5 * sqrtf(1.f / 3.f)) * dh / uRef);

	if (dt < dt_min || dt > dt_max)
		throw std::runtime_error("Scale out of Bound");

	C_vis = dh * C_u;

	viscosity_L = viscosity / C_vis;
	this->r_vis = 1.f / (3.f * viscosity_L + 0.5f);

	for (int i = 1; i < D; ++i)
		this->gridDim_L[i] = static_cast<unsigned int>(gridDim[i] / dh);

	C_p = rho * dh * dh / (dt * dt);
	C_f = rho * dh * dh * dh * dh / (dt * dt);
	C_F = rho * dh / (dt * dt);

	return true;
}

void SimDomain::rescaleConFactors(float uMax_L, float& scale_u, float& scale_F)
{
	float uRef_new = C_u * uMax_L;
	float q = 1.0f;
	if (uRef_new < uRef * q)
		uRef_new = uRef * q;

	scale_u = C_u * uRef_L / uRef_new;
	C_u = uRef_new / uRef_L;
	dt = dh / C_u;
	float dt_min = (float)((dh / uRef_new) * 10e-2);
	float dt_max = (float)((0.5 * sqrtf(1.f / 3.f)) * dh / uRef_new);

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
	r_vis = 1.f / (3.f * viscosity_L + 0.5f);


	C_p = rho * dh * dh / (dt * dt);
	C_f = rho * dh * dh * dh * dh / (dt * dt);
	float C_F_new = rho * dh / (dt * dt);
	scale_F = C_F / C_F_new;
	C_F = C_F_new;
}