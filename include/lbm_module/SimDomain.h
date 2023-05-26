#ifndef SIM_DOMAIN
#define SIM_DOMAIN

#include <array>
#include <stdexcept>
#include <memory>
#include <DataStructureAlg.h>

namespace LBMDimensions {
	static constexpr unsigned int Q = 27;
	static constexpr unsigned int D = 3;
	static constexpr float cs_sq = 1.f / 3.f;
};

struct SimDomain_dev {
	float c[LBMDimensions::Q * LBMDimensions::D];
	float w[LBMDimensions::Q];
	unsigned int gridDim_L[LBMDimensions::D];
	float zerothRelaxationTime;
	float lowRelaxationTimes;
	float r_3, r_4, r_5, r_6;

	//local high order relaxation
	float param_0, param_1, param_2, param_3;

	unsigned int maxNodeCount;
	unsigned int alpha;

};

class SimDomain
{
private:
	SimDomain_dev _sd_dev;
	unsigned int _alpha = 32;
	std::array<unsigned int, 3> _gridDim_L;
	float _uRef_L = 0.2f;
	float _viscosity_L;
	float _r_vis = 1.0f;

	float _zerothRelaxationTime = 1.0f;
	float _lowRelaxationTimes = 1.0f;
	float _r_3 = 1.94175f,
		_r_4 = 1.91939f,
		_r_5 = 1.89753f,
		_r_6 = 1.8868f;

	float _param_0 = 0.0003f,
		_param_1 = -0.00775f,
		_param_2 = 0.00016f,
		_param_3 = 0.0087f;

	std::array<std::array<float, LBMDimensions::D>, LBMDimensions::Q> _c;
	std::array<float, LBMDimensions::Q> _w;

	//Physical Unit
	std::array<float, LBMDimensions::D> _gridDim;		//[m]				
	float _dt;							//[s]
	float _dh;							//[m]
	float _rho;							//[kg/m^3]
	float _p0;							//[Pa]
	float _uRef;							//[m/s]
	float _viscosity;					//[m^2/s] 

	//Conversion factors
	float _C_u, _C_p, _C_f, _C_F, _C_vis; 

	void changeSimDomDev();

public:
	SimDomain(std::array <float, LBMDimensions::D> gridDim, float viscosity, float density, float uRef, float referencePressure = 1.0);
	unsigned int getMaxNodeCount() const;
	float getGridDim(const unsigned int axis = 0) const;
	unsigned int getGridDim_L(const unsigned int axis) const;
	std::array<unsigned int,3> getGridDim_L() const;
	float getGridSize() const;
	float getRelaxationConstant() const;
	float getViscosity_L() const;
	float getRho() const;
	float getC_u() const;
	float get_uRef_L() const;
	float get_uRef() const;
	float getC_p() const;
	float getC_vis() const;
	float getC_f() const;
	float getC_F() const;
	void setAlpha(unsigned int alpha);

	bool calcConvFactors(int gridDimX_L);
	void rescaleConFactors(float uMax_L, float& scale_u, float& scale_F);

	SimDomain_dev getSimDomainStruct();

};

#endif // !SIM_DOMAIN
