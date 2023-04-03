#ifndef LBM_SOLVER
#define LBM_SOLVER
#include "SimDomain.h"
#include "SimState.h"
#include "LBM_Writer.h"
#include "LBM_Types.h"
#include "ImmersedBoundaryMethod.h"
#include "ParticleGenerator.h"
#include "ParticleSource.h"
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <chrono>


template<typename T, size_t D> class LBM_Solver_Specialisation;

template<typename T> 
class LBM_Solver_Specialisation<T,2>
{
protected:
	SimState<T, 2> st;
	SimDomain<T, 2> sd;
	std::unique_ptr<ParticleGenerator<T, 2>> particleGenerator = nullptr;
	std::unique_ptr<IBMethod<T, 2>> ibm = nullptr;
	std::unique_ptr<LBM_Writer<T, 2>> writer = nullptr;

	LBM_Solver_Specialisation(SimDomain<T, 2> simDom, std::unique_ptr<IBMethod<T, 2>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 2>> writer);

	T calc_f_eq_CM(int i, T rho, vec<T, 2> u);
	void collision_CM();
	void streaming();
};

template<typename T>
class LBM_Solver_Specialisation<T, 3>
{
protected:
	SimState<T, 3> st;
	SimDomain<T, 3> sd;
	std::unique_ptr<ParticleGenerator<T, 3>> particleGenerator = nullptr;
	std::unique_ptr<IBMethod<T, 3>> ibm = nullptr;
	std::unique_ptr<LBM_Writer<T, 3>> writer = nullptr;

	LBM_Solver_Specialisation(SimDomain<T, 3> simDom, std::unique_ptr<IBMethod<T, 3>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 3>> writer);

	T calc_f_eq_CM(int i, T rho, vec<T,3> u);
	
	void collision_CM();
	void streaming();
};

template<typename T, size_t D>
class LBM_Solver: public LBM_Solver_Specialisation<T, D>
{
private:
	LBM_Solver_Specialisation<T, D>::st;
	LBM_Solver_Specialisation<T, D>::sd;
	LBM_Solver_Specialisation<T, D>::particleGenerator;
	LBM_Solver_Specialisation<T, D>::ibm;
	LBM_Solver_Specialisation<T, D>::writer;
	
	int simulationStep = 0;
	bool staticBody = true;
	bool BGKused = false;
	bool showParticleDensity;
	bool adaptiveTimeStep = true;
	bool unitTime;
	T maxSimulationTime;
	std::ostringstream oStringStr;

	void collision_BGK();
	void updateMoments();
	void collision_ghost_BGK();
	void collision_ghost_CM();
	void rescaleMoments_BGK(const T& scale_u, const  T& scale_F);
	void rescaleMoments_CM(const T& scale_u, const T& scale_F);

	void find_UMax(T& uMax_mag);

public:
	LBM_Solver(SimDomain<T, D> simDom, std::unique_ptr<IBMethod<T, D>> immersedBoundary, std::unique_ptr<LBM_Writer<T, D>> writer, double maxSimulationTime, bool unitTime = false, SimInitialiser<T, D> simInit = {});
	LBM_Solver(SimDomain<T, D> simDom, std::unique_ptr<LBM_Writer<T, D>> writer, double maxSimulationTime, bool unitTime = false, SimInitialiser<T, D> simInit = {});
	LBM_Solver(SimDomain<T, D> simDom, std::unique_ptr<IBMethod<T, D>> immersedBoundary, double maxSimulationTime, bool unitTime = false, SimInitialiser<T, D> simInit = {});
	LBM_Solver(SimDomain<T, D> simDom, double maxSimulationTime, bool unitTime = false, SimInitialiser<T, D> simInit = {});

	void setParticleGenerator(std::unique_ptr<ParticleGenerator<T, D>> particleGenerator, bool showParticleDensity = true);
	void setMaxSimulationTime(T maxSimulationTime);
	void solve();
	const std::string getClockOutput() const;
	void enableBGK();
	void disableAdaptiveTimeStep();

};

#endif