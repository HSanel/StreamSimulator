#ifndef SIM_DOMAIN
#define SIM_DOMAIN

#include <array>
#include <stdexcept>
	

class SimDomain
{
public:
	static constexpr size_t Q = 27;
	static constexpr size_t D = 3;

private:
	std::array<unsigned int, 3> gridDim_L;
	float uRef_L = 0.2f;
	float viscosity_L;
	float r_vis = 1.0f;

	float zerothRelaxationTime = 1.0f;
	float lowRelaxationTimes = 1.0f;
	float r_3 = 1.94175f,
		r_4 = 1.91939f,
		r_5 = 1.89753f,
		r_6 = 1.8868f;

	float param_0 = 0.0003f,
		param_1 = -0.00775f,
		param_2 = 0.00016f,
		param_3 = 0.0087f;

	std::array<std::array<float, D>, Q> c_;	
	std::array<float, Q> w_;			

	//Physical Unit
	std::array<float, D> gridDim;		//[m]				
	float dt;							//[s]
	float dh;							//[m]
	float rho;							//[kg/m^3]
	float p0;							//[Pa]
	float uRef;							//[m/s]
	float viscosity;					//[m^2/s] 

	//Conversion factors
	float C_u, C_p, C_f, C_F, C_vis; 

public:
	SimDomain(std::array <float, 3> gridDim, float viscosity, float density, float uRef, float referencePressure = 1.0);
	unsigned int getMaxNodeCount() const;
	float getGridDim(const unsigned int axis = 0) const;
	unsigned int getGridDim_L(const unsigned int axis = 0) const;
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

	bool calcConvFactors(int gridDimX_L);
	void rescaleConFactors(float uMax_L, float& scale_u, float& scale_F);

};

#endif // !SIM_DOMAIN
