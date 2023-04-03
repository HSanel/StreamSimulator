#ifndef LBM_SOLVER_P
#define LBM_SOLVER_P
#include "SimDomain.h"
#include "SimState_P.h"
#include "LBM_Writer.h"
#include "LBM_Types.h"
#include "ImmersedBoundaryMethod_P.h"
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <chrono>
#include <fstream> 
#include "cudaErrorHandle.h"
#include <set>


template<typename T, size_t D> class LBM_Solver_Specialisation_P;

#pragma region Solver_spec_2D
template<typename T>
class LBM_Solver_Specialisation_P<T, 2>
{
protected:
	SimState<T, 2> st;
	SimState_P<T, 2> st_P;
	SimDomain<T, 2> sd;
	std::unique_ptr<ParticleGenerator_P<T, 2>> particleGenerator = nullptr;
	std::unique_ptr<IBMethod_P<T, 2>> ibm = nullptr;
	std::unique_ptr<LBM_Writer<T, 2>> writer = nullptr;

	LBM_Solver_Specialisation_P(SimDomain<T, 2> simDom, std::unique_ptr<IBMethod_P<T, 2>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 2>> writer);

	void collision_CM(T r_vis, T zerothMomSum = 0, T firstMomSum = 0, T secondMomMean = 0);
	void streaming(const T& C_u, const T& C_p);
};
#pragma endregion

template<typename T>
class LBM_Solver_Specialisation_P<T, 3>
{
protected:
	SimState<T, 3> st;
	SimState_P<T, 3> st_P;
	SimDomain<T, 3> sd;
	std::unique_ptr<ParticleGenerator_P<T, 3>> particleGenerator = nullptr;
	std::unique_ptr<IBMethod_P<T, 3>> ibm = nullptr;
	std::unique_ptr<LBM_Writer<T, 3>> writer = nullptr;

	LBM_Solver_Specialisation_P(SimDomain<T, 3> simDom, std::unique_ptr<IBMethod_P<T, 3>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 3>> writer);
	
	void collision_CM(T r_vis, T zerothMomSum = 0, T firstMomSum = 0, T secondMomMean = 0);
	void streaming(const T& C_u, const T& C_p);
};

template<typename T, size_t D>
class LBM_Solver_P : public LBM_Solver_Specialisation_P<T, D>
{
private:
	T Cd, Cl, A;
	int simulationStep = 0;
	bool staticBody = true;
	bool BGKused = false;
	bool showParticleDensity;
	bool adaptiveTimeStep = true;
	bool unitTime;
	T maxSimulationTime;
	std::ostringstream oStringStr;

	void collision_BGK(T r_vis);
	void updateMoments();
	void rescaleMoments_BGK(const T &scale_u,const  T &scale_F);
	void rescaleMoments_CM(const T& scale_u, const T& scale_F);

public:
	LBM_Solver_P(SimDomain<T, D> simDom, std::unique_ptr<IBMethod_P<T, D>> immersedBoundary, std::unique_ptr<LBM_Writer<T, D>> writer, T maxSimulationTime, SimInitialiser_P<T, D> simInit, bool unitTime = false);
	LBM_Solver_P(SimDomain<T, D> simDom, std::unique_ptr<LBM_Writer<T, D>> writer, T maxSimulationTime, SimInitialiser_P<T, D> simInit, bool unitTime = false);
	LBM_Solver_P(SimDomain<T, D> simDom, T maxSimulationTime, SimInitialiser_P<T, D> simInit, bool unitTime = false);

	void setParticleGenerator(std::unique_ptr<ParticleGenerator_P<T, D>> particleGenerator, bool showParticleDensity = true);
	void solve(T A);
	const std::string getClockOutput() const;
	void enableBGK();
	void disableAdaptiveTimeStep();
};
#endif