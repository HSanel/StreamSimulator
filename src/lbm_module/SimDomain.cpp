#include "SimDomain.h"

SimDomain::SimDomain(std::array <float, 3> GridDim, float Viscosity, float Density, float uRef, float ReferencePressure)
	:_gridDim(GridDim), _viscosity(Viscosity), _rho(Density), _uRef(uRef), _p0(ReferencePressure), _sd_dev(SimDomain_dev{})
{
	_w[0] = 8.0 / 27.0;
	_w[1] = 2.0 / 27.0;
	_w[2] = 2.0 / 27.0;
	_w[3] = 2.0 / 27.0;
	_w[4] = 2.0 / 27.0;
	_w[5] = 2.0 / 27.0;
	_w[6] = 2.0 / 27.0;
	_w[7] = 1.0 / 54.0;
	_w[8] = 1.0 / 54.0;
	_w[9] = 1.0 / 54.0;
	_w[10] = 1.0 / 54.0;
	_w[11] = 1.0 / 54.0;
	_w[12] = 1.0 / 54.0;
	_w[13] = 1.0 / 54.0;
	_w[14] = 1.0 / 54.0;
	_w[15] = 1.0 / 54.0;
	_w[16] = 1.0 / 54.0;
	_w[17] = 1.0 / 54.0;
	_w[18] = 1.0 / 54.0;
	_w[19] = 1.0 / 216.0;
	_w[20] = 1.0 / 216.0;
	_w[21] = 1.0 / 216.0;
	_w[22] = 1.0 / 216.0;
	_w[23] = 1.0 / 216.0;
	_w[24] = 1.0 / 216.0;
	_w[25] = 1.0 / 216.0;
	_w[26] = 1.0 / 216.0;

	_c[0][0] = 0.0;
	_c[0][1] = 0.0;
	_c[0][2] = 0.0;

	_c[1][0] = 1.0;
	_c[1][1] = 0.0;
	_c[1][2] = 0.0;

	_c[2][0] = -1.0;
	_c[2][1] = 0.0;
	_c[2][2] = 0.0;

	_c[3][0] = 0.0;
	_c[3][1] = 1.0;
	_c[3][2] = 0.0;

	_c[4][0] = 0.0;
	_c[4][1] = -1.0;
	_c[4][2] = 0.0;

	_c[5][0] = 0.0;
	_c[5][1] = 0.0;
	_c[5][2] = 1.0;

	_c[6][0] = 0.0;
	_c[6][1] = 0.0;
	_c[6][2] = -1.0;

	_c[7][0] = 1.0;
	_c[7][1] = 1.0;
	_c[7][2] = 0.0;

	_c[8][0] = -1.0;
	_c[8][1] = -1.0;
	_c[8][2] = 0.0;

	_c[9][0] = 1.0;
	_c[9][1] = 0.0;
	_c[9][2] = 1.0;

	_c[10][0] = -1.0;
	_c[10][1] = 0.0;
	_c[10][2] = -1.0;

	_c[11][0] = 0.0;
	_c[11][1] = 1.0;
	_c[11][2] = 1.0;

	_c[12][0] = 0.0;
	_c[12][1] = -1.0;
	_c[12][2] = -1.0;

	_c[13][0] = 1.0;
	_c[13][1] = -1.0;
	_c[13][2] = 0.0;

	_c[14][0] = -1.0;
	_c[14][1] = 1.0;
	_c[14][2] = 0.0;

	_c[15][0] = 1.0;
	_c[15][1] = 0.0;
	_c[15][2] = -1.0;

	_c[16][0] = -1.0;
	_c[16][1] = 0.0;
	_c[16][2] = 1.0;

	_c[17][0] = 0.0;
	_c[17][1] = 1.0;
	_c[17][2] = -1.0;

	_c[18][0] = 0.0;
	_c[18][1] = -1.0;
	_c[18][2] = 1.0;

	_c[19][0] = 1.0;
	_c[19][1] = 1.0;
	_c[19][2] = 1.0;

	_c[20][0] = -1.0;
	_c[20][1] = -1.0;
	_c[20][2] = -1.0;

	_c[21][0] = 1.0;
	_c[21][1] = 1.0;
	_c[21][2] = -1.0;

	_c[22][0] = -1.f;
	_c[22][1] = -1.f;
	_c[22][2] = 1.f;

	_c[23][0] = 1.0;
	_c[23][1] = -1.0;
	_c[23][2] = 1.0;

	_c[24][0] = -1.0;
	_c[24][1] = 1.0;
	_c[24][2] = -1.0;

	_c[25][0] = -1.0;
	_c[25][1] = 1.0;
	_c[25][2] = 1.0;

	_c[26][0] = 1.0;
	_c[26][1] = -1.0;
	_c[26][2] = -1.0;


	changeSimDomDev();
}

void SimDomain::changeSimDomDev()
{
	pack_GPU(_sd_dev.c, _c.data(), LBMDimensions::Q);
	std::memcpy(_sd_dev.w, _w.data(), LBMDimensions::Q);
	std::memcpy(_sd_dev.gridDim_L, _gridDim_L.data(), LBMDimensions::D);
	_sd_dev.maxNodeCount = _gridDim_L[0] * _gridDim_L[1] * _gridDim_L[2];
	_sd_dev.zerothRelaxationTime = _zerothRelaxationTime;
	_sd_dev.lowRelaxationTimes = _lowRelaxationTimes;
	_sd_dev.alpha = _alpha;
	_sd_dev.r_3 = _r_3;
	_sd_dev.r_4 = _r_4;
	_sd_dev.r_5 = _r_5;
	_sd_dev.r_6 = _r_6;
	_sd_dev.param_0 = _param_0;
	_sd_dev.param_1 = _param_1;
	_sd_dev.param_2 = _param_2;
	_sd_dev.param_3 = _param_3;
}

unsigned int SimDomain::getMaxNodeCount() const 
{
	unsigned int nodeCount = 1;

	for (int i = 0; i < LBMDimensions::D; ++i)
		nodeCount *= _gridDim_L[i];

	return nodeCount;
}

float SimDomain::getGridDim(const unsigned int axis) const
{
	if (axis < 3)
		return _gridDim[axis];
	else 
		return 0.f;
}

unsigned int SimDomain::getGridDim_L(const unsigned int axis) const
{
	if (axis < 3)
		return _gridDim_L[axis];
	else
		return 0;
}
std::array<unsigned int, 3> SimDomain::getGridDim_L() const
{
	return _gridDim_L;
}

float SimDomain::getGridSize() const { return _dh; }

float SimDomain::getRelaxationConstant() const { return _r_vis; }

float SimDomain::getViscosity_L() const { return _viscosity_L; }

float SimDomain::getRho() const { return _rho; }

float SimDomain::getC_u() const { return _C_u; }

float SimDomain::get_uRef_L() const { return _uRef_L; }

float SimDomain::get_uRef() const { return _uRef; }

float SimDomain::getC_p() const { return _C_p; }

float SimDomain::getC_vis() const { return _C_vis; }

float SimDomain::getC_f() const { return _C_f; }

float SimDomain::getC_F() const { return _C_F; }

bool SimDomain::calcConvFactors(int gridDimX_L)
{
	this->_gridDim_L[0] = gridDimX_L;
	_dh = _gridDim[0] / static_cast<float>(_gridDim_L[0]);
	_C_u = _uRef / _uRef_L;
	_dt = _dh / _C_u;
	float _dt_min = static_cast<float>((_dh / _uRef) * 10e-2);
	float _dt_max = static_cast<float>((0.5 * sqrtf(1.f / 3.f)) * _dh / _uRef);

	if (_dt < _dt_min || _dt > _dt_max)
		throw std::runtime_error("Scale out of Bound");

	_C_vis = _dh * _C_u;

	_viscosity_L = _viscosity / _C_vis;
	this->_r_vis = 1.f / (3.f * _viscosity_L + 0.5f);

	for (int i = 1; i < LBMDimensions::D; ++i)
		this->_gridDim_L[i] = static_cast<unsigned int>(_gridDim[i] / _dh);

	_C_p = _rho * _dh * _dh / (_dt * _dt);
	_C_f = _rho * _dh * _dh * _dh * _dh / (_dt * _dt);
	_C_F = _rho * _dh / (_dt * _dt);

	changeSimDomDev();
	return true;
}

void SimDomain::rescaleConFactors(float uMax_L, float& scale_u, float& scale_F)
{
	float _uRef_new = _C_u * uMax_L;
	float _q = 1.0f;
	if (_uRef_new < _uRef * _q)
		_uRef_new = _uRef * _q;

	scale_u = _C_u * _uRef_L / _uRef_new;
	_C_u = _uRef_new / _uRef_L;
	_dt = _dh / _C_u;
	float _dt_min = (float)((_dh / _uRef_new) * 10e-2);
	float _dt_max = (float)((0.5 * sqrtf(1.f / 3.f)) * _dh / _uRef_new);

	if (_dt < _dt_min)
	{
		_dt = _dt_min;
		_C_u = _dh / _dt;
		scale_u = _C_u * _uRef_L / _uRef_new;
	}

	if (_dt > _dt_max)
	{
		_dt = _dt_max;
		_C_u = _dh / _dt;
		scale_u = _C_u * _uRef_L / _uRef_new;
	}

	_C_vis = _dh * _C_u;
	_viscosity_L = _viscosity / _C_vis;
	_r_vis = 1.f / (3.f * _viscosity_L + 0.5f);

	_C_p = _rho * _dh * _dh / (_dt * _dt);
	_C_f = _rho * _dh * _dh * _dh * _dh / (_dt * _dt);
	float _C_F_new = _rho * _dh / (_dt * _dt);
	scale_F = _C_F / _C_F_new;
	_C_F = _C_F_new;
	changeSimDomDev();
}

SimDomain_dev SimDomain::getSimDomainStruct()
{
	return _sd_dev;
}

void SimDomain::setAlpha(unsigned int alpha) { this->_alpha = alpha; }