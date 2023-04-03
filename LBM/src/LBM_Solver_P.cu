#include "LBM_Solver_P.h"
#include "LBM_Solver_Kernels.h"

template struct LBM_Solver_Specialisation_P<float, 2>;
template struct LBM_Solver_Specialisation_P<float, 3>;
template struct LBM_Solver_Specialisation_P<double, 2>;
template struct LBM_Solver_Specialisation_P<double, 3>;

template struct LBM_Solver_P<float, 2>;
template struct LBM_Solver_P<float, 3>;
template struct LBM_Solver_P<double, 2>;
template struct LBM_Solver_P<double, 3>;

#pragma region solver_spec_2D
//specialised constructors
template<typename T>
LBM_Solver_Specialisation_P<T, 2>::LBM_Solver_Specialisation_P(SimDomain<T, 2> simDom, std::unique_ptr<IBMethod_P<T, 2>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 2>> writer)
	:sd(simDom),ibm(std::move(immersedBoundary)), writer(std::move(writer)), st(SimState<T, 2>{simDom, false}), st_P(SimState_P<T, 2>{simDom, immersedBoundary != nullptr}) {}	

template<typename T>
void LBM_Solver_Specialisation_P<T, 2>::streaming(const T& C_u, const T& C_p)
{
	//this->kernelSize = 1;
	dim3 GRID_DIM{ (this->sd.getGridDim_L(0) + prop.warpSize - 1) / prop.warpSize,  (this->sd.getGridDim_L(1) + warpCount - 1) / warpCount };
	dim3 BLOCK_DIM{ static_cast<unsigned int>(prop.warpSize), warpCount };

	streaming2D_kernel<T> << < GRID_DIM, BLOCK_DIM >> >
		(this->st_P.f.get(), this->st_P.f_star.get(), this->st_P.u_L.get(), this->st_P.rho_L.get(), C_u, C_p);

	//streaming2D_kernel_L<T> <<<GRID_DIM, BLOCK_DIM >>>
	//	(this->st_P.f.get(), this->st_P.f_star.get(), this->st_P.u_L.get(), this->st_P.rho_L.get());

	//streaming2D_kernel_R<T> <<<GRID_DIM, BLOCK_DIM>>>
	//	(this->st_P.f.get(), this->st_P.f_star.get(), this->st_P.u_L.get(), this->st_P.rho_L.get());

	//streaming2D_kernel_T<T> <<<GRID_DIM, BLOCK_DIM>>>
	//	(this->st_P.f.get(), this->st_P.f_star.get(), this->st_P.u_L.get(), this->st_P.rho_L.get());

	//streaming2D_kernel_B<T> <<<GRID_DIM, BLOCK_DIM>>>
	//	(this->st_P.f.get(), this->st_P.f_star.get(), this->st_P.u_L.get(), this->st_P.rho_L.get());
}

template<typename T>
void LBM_Solver_Specialisation_P<T, 2>::collision_CM(T r_vis, T zerothMomMean, T firstMomMean, T secondMomMean)
{
	//PARALLEL CODE
	dim3 GRID_DIM{ (this->sd.getGridDim_L(0) + prop.warpSize - 1) / prop.warpSize,  (this->sd.getGridDim_L(1) + warpCount - 1) / warpCount };
	dim3 BLOCK_DIM{ static_cast<unsigned int>(prop.warpSize), warpCount };

	if (sd.isLocalRelaxationEnabled())
		collision2D_kernel_CM<T> << < GRID_DIM, BLOCK_DIM >> >
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.F_ext_L.get(), this->st_P.f.get(), this->st_P.f_star.get(), r_vis, zerothMomMean, firstMomMean, secondMomMean);
	else
		collision2D_kernel_CM<T> << < GRID_DIM, BLOCK_DIM >> >
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.F_ext_L.get(), this->st_P.f.get(), this->st_P.f_star.get(), r_vis);


}
#pragma endregion
//3D
//-----------------------
template<typename T>
LBM_Solver_Specialisation_P<T, 3>::LBM_Solver_Specialisation_P(SimDomain<T, 3> simDom, std::unique_ptr<IBMethod_P<T, 3>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 3>> writer)
	: sd(simDom), ibm(std::move(immersedBoundary)), writer(std::move(writer)), st(SimState<T, 3>{simDom, false}), st_P(SimState_P<T, 3>{simDom, immersedBoundary != nullptr}) {}		

//base constructors
template<typename T, size_t D>
LBM_Solver_P<T,D>::LBM_Solver_P(SimDomain<T, D> simDom, std::unique_ptr<IBMethod_P<T, D>> immersedBoundary, std::unique_ptr<LBM_Writer<T, D>> writer, T maxSimulationTime, SimInitialiser_P<T, D> simInit, bool unitTime)
	:LBM_Solver_Specialisation_P<T, D>(simDom, std::move(immersedBoundary), std::move(writer)), unitTime(unitTime)
{
	if (unitTime)
		this->maxSimulationTime = maxSimulationTime * this->sd.getTimeStep();
	else
		this->maxSimulationTime = maxSimulationTime;
	simulationStep = 0;

	memset(&prop, 0, sizeof(cudaDeviceProp));

	int count, dev;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; ++i)
	{
		if (prop.major > 3)
			dev = i;
	}

	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaSetDevice(dev));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
 
	//printGPUInfo(dev);

	SimDomain_dev<T, D, SimDomain<T, D>::Q> sd_temp;
	this->sd.copyDataToStruct(sd_temp);
	//sd_temp.alpha = this->sd.getMaxNodeCount();
	sd_temp.alpha = 32;


	if (!normilized_parameter)
	{
		simInit.u /= simDom.getC_u();
		simInit.rho /= simDom.getRho();
		simInit.F_ext /= simDom.getC_F();
	}

	HANDLE_ERROR
	(
		cudaMemcpyToSymbol(sd_dev<T, D, SimDomain<T, D>::Q>, &sd_temp, sizeof(SimDomain_dev<T,D, SimDomain<T, D>::Q>))
	);

	if (this->ibm)
	{
		this->ibm->setCudaProp(prop);
		this->ibm->setWarpCount(warpCount);
		this->ibm->copyConstantSimDom(sd_temp);
	}

	initializeField_kernel<T, D, SimDomain<T, D>::Q> <<<(this->sd.getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize * warpCount >>>
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.F_ext_L.get(), this->st_P.f_star.get(), simInit);
}

template<typename T, size_t D>
LBM_Solver_P<T, D>::LBM_Solver_P(SimDomain<T, D> simDom, std::unique_ptr<LBM_Writer<T, D>> writer, T maxSimulationTime, SimInitialiser_P<T, D> simInit, bool unitTime)
:LBM_Solver_P(simDom, nullptr, std::move(writer), maxSimulationTime, simInit, unitTime) {}

template<typename T, size_t D>
LBM_Solver_P<T, D>::LBM_Solver_P(SimDomain<T, D> simDom, T maxSimulationTime, SimInitialiser_P<T, D> simInit, bool unitTime)
	:LBM_Solver_P(simDom, nullptr, maxSimulationTime, simInit, unitTime) {}


//specialised methods


template<typename T>
void LBM_Solver_Specialisation_P<T, 3>::streaming(const T& C_u, const T& C_p)
{
	dim3 GRID_DIM{ (this->sd.getGridDim_L(0) + prop.warpSize - 1) / prop.warpSize,  (this->sd.getGridDim_L(1) + warpCount/2 - 1) / (warpCount/2), (this->sd.getGridDim_L(2) + 2 - 1) / 2 };
	dim3 BLOCK_DIM{ static_cast<unsigned int>(prop.warpSize), warpCount/2, 2 };

	streaming3D_kernel<T> <<< GRID_DIM, BLOCK_DIM >>>
		(this->st_P.f.get(), this->st_P.f_star.get(), this->st_P.u_L.get(), this->st_P.rho_L.get(), C_u, C_p);
}


template<typename T>
void LBM_Solver_Specialisation_P<T, 3>::collision_CM(T r_vis, T zerothMomMean, T firstMomMean, T secondMomMean)
{
	//PARALLEL CODE
	dim3 GRID_DIM{ (this->sd.getGridDim_L(0) + prop.warpSize - 1) / prop.warpSize,  (this->sd.getGridDim_L(1) + warpCount/2 - 1) / (warpCount/2), (this->sd.getGridDim_L(2) + 2 - 1) / 2 };
	dim3 BLOCK_DIM{ static_cast<unsigned int>(prop.warpSize), warpCount/2, 2 };

	if (sd.isLocalRelaxationEnabled())
		collision3D_kernel_CM<T> <<< GRID_DIM, BLOCK_DIM >>>
			(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.secondMom_L.get(), this->st_P.F_ext_L.get(), this->st_P.f.get(), this->st_P.f_star.get(), r_vis, zerothMomMean, firstMomMean, secondMomMean);
	else
		collision3D_kernel_CM<T> << < GRID_DIM, BLOCK_DIM >> >
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.secondMom_L.get(), this->st_P.F_ext_L.get(), this->st_P.f.get(), this->st_P.f_star.get(), r_vis);

}

//---------------------------------------------
//base methods
template<typename T, size_t D>
void LBM_Solver_P<T, D>::collision_BGK(T r_vis)
{
	//PARALLEL CODE

	collision_kernel_BGK<T, D, SimDomain<T, D>::Q> <<<(this->sd.getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize * warpCount >>>
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.F_ext_L.get(), this->st_P.f.get(), this->st_P.f_star.get(), r_vis);
}

template<typename T, size_t D>
void LBM_Solver_P<T, D>::updateMoments()
{
	//PARALLEL CODE
	if (ibm)
	{
			updateMoments_kernel<T, D, SimDomain<T, D>::Q> << <(this->sd.getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
			(this->st_P.rho_L.get(), this->st_P.u_unc_L.get(), this->st_P.u_L.get(), this->st_P.secondMom_L.get(), this->st_P.uMag.get(), this->st_P.zerothMomSum.get(), this->st_P.firstMomSum.get(), this->st_P.secondMomSum.get(), this->st_P.F_ext_L.get(), this->st_P.f.get());
	}
	else
	{
			updateMoments_kernel<T, D, SimDomain<T, D>::Q> << <(this->sd.getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
				(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.secondMom_L.get(), this->st_P.uMag.get(), this->st_P.zerothMomSum.get(), this->st_P.firstMomSum.get(), this->st_P.secondMomSum.get(), this->st_P.F_ext_L.get(), this->st_P.f.get());
	}

	if (ibm)
	{
		vec<T, D> Fsg_temp{};
		HANDLE_ERROR(cudaMemcpy(this->st_P.Fsg.get(), &Fsg_temp, sizeof(vec<T,D>), cudaMemcpyHostToDevice));
		ibm->calcFs(this->st_P.u_unc_L.get(), this->st_P.F_ext_L.get(), this->st_P.rho_L.get(), this->st_P.Fsg.get());
		HANDLE_ERROR(cudaMemcpy(&Fsg_temp, this->st_P.Fsg.get(), sizeof(T), cudaMemcpyDeviceToHost));
		T b = sd.get_uRef();
		T c = sd.getC_u();
		T a = ((sd.get_uRef() / sd.getC_u()) * (sd.get_uRef() / sd.getC_u()) * this->A);
		Cd = -(T)2.0 * (T)0.36 * Fsg_temp[0] / a; //Fsgx
		Cl = -(T)2.0 * (T)0.36 * Fsg_temp[1] / a;
	}

}

template<typename T, size_t D>
void LBM_Solver_P<T, D>::setParticleGenerator(std::unique_ptr<ParticleGenerator_P<T, D>> particleGenerator, bool showParticleDensity)
{
	this->particleGenerator = std::move(particleGenerator);
	this->showParticleDensity = showParticleDensity;

	if (showParticleDensity)
	{
		this->st_P.allocateParticleDensity();
		this->st.allocateParticleDensity();
	}

	this->particleGenerator->setWarpCount(warpCount);
	this->particleGenerator->setCudaProp(prop);
}

template<typename T, size_t D>
const std::string LBM_Solver_P<T, D>::getClockOutput() const
{
	return oStringStr.str();
}

template<typename T, size_t D>
void LBM_Solver_P<T, D>::enableBGK()
{
	this->BGKused = true;
}

template<typename T, size_t D>
void LBM_Solver_P<T, D>::disableAdaptiveTimeStep() { this->adaptiveTimeStep = false; }

template<typename T, size_t D>
void LBM_Solver_P<T, D>::rescaleMoments_BGK(const T& scale_u, const T& scale_F)
{
	rescaleMoments_BGK_kernel<T, D, SimDomain<T, D>::Q> << <(this->sd.getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.u_unc_L.get(), this->st_P.f.get(), this->st_P.F_ext_L.get(),scale_u, scale_F);
}

template<typename T, size_t D>
void LBM_Solver_P<T, D>::rescaleMoments_CM(const T& scale_u, const T& scale_F)
{
	rescaleMoments_CM_kernel<T, D, SimDomain<T, D>::Q> << <(this->sd.getMaxNodeCount() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
		(this->st_P.rho_L.get(), this->st_P.u_L.get(), this->st_P.u_unc_L.get(), this->st_P.f.get(), this->st_P.F_ext_L.get(),scale_u, scale_F);
}

template<typename T, size_t D>
void LBM_Solver_P<T, D>::solve( T A=1)
{
	this->A = A/(this->sd.getGridSize()* this->sd.getGridSize());
	
	//PARALLEL & SEQUENTIELL CODE
	bool visualData = false;
	int percent = 0;
	std::string loadingPoints = "";
	T uMax_mag = 0, scale_u = 1, scale_F = 1;
	T currentSimulationTime = 0;
	T zerothMomMean = 0, firstMomMean = 0, secondMomMean = 0;
	if (this->sd.isLocalRelaxationEnabled())
		this->st_P.allocateSecondMom();

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	oStringStr << "\n\nOhne GPU-CPU-COPY und auf die Festplatte schreiben\n-----------------\n";
	try
	{
		if (this->ibm)
			this->ibm->reset(this->sd);
		
		HANDLE_ERROR(cudaEventRecord(start, 0));

		std::cout << "Calculating: \n" << loadingPoints << " 0%";
		while (currentSimulationTime <= maxSimulationTime)
		{
			int maxSimulationTimeSteps = (int)(maxSimulationTime / this->sd.getTimeStep());
			
			int percentTemp = (int)((currentSimulationTime / maxSimulationTime) * 100.0);
			if (percent != percentTemp)
			{
				percent = percentTemp;
				loadingPoints += ".";
				std::cout << "\r";
				std::cout << loadingPoints << " " << percent << "%";
			}

			if (this->sd.isLocalRelaxationEnabled())
			{
				zerothMomMean = 0; firstMomMean = 0; secondMomMean = 0;
				HANDLE_ERROR(cudaMemcpy(this->st_P.zerothMomSum.get(), &zerothMomMean, sizeof(T), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(this->st_P.firstMomSum.get(), &firstMomMean, sizeof(T), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(this->st_P.secondMomSum.get(), &secondMomMean, sizeof(T), cudaMemcpyHostToDevice));
			}
			
			{
				//for verification
				std::ofstream  drag;
				std::ofstream  lift;
				std::ofstream  timeData;
				drag.open("Drag_Data.txt", std::ios::out | std::ios::app);
				lift.open("Lift_Data.txt", std::ios::out | std::ios::app);
				timeData.open("Time_Data.txt", std::ios::out | std::ios::app);
				drag << Cd << "\n";
				lift << Cl << "\n";
				timeData << currentSimulationTime << "\n";
				drag.close();
				lift.close();
				timeData.close();
			}

			this->streaming(this->sd.getC_u(), this->sd.getC_p());
			updateMoments();
			this->st_P.memCpyMoments(this->st, DevToHost);

			if (this->particleGenerator)
			{
				if (this->particleGenerator->isSimultan())
					this->particleGenerator->activateParticlesSimultaneously(this->sd, this->st_P.u_L.get(), currentSimulationTime);

				
				if (this->showParticleDensity)
					this->particleGenerator->calcParticleDensity(simulationStep, this->st_P.particleDensity.get(), this->sd);
			}

			if (this->writer && simulationStep % this->writer->getStepsPerFrame() == 0)
			{
				if (this->writer->getDestination() == VTK_FILE)
					visualData = this->writer->writeMomentsToVTKFile(this->sd, this->st, currentSimulationTime, simulationStep);
				else
					visualData = this->writer->writeMomentsToTXTFile(this->sd, this->st, currentSimulationTime, simulationStep);

				if (this->ibm)
					this->writer->writeImBody(this->sd, *this->ibm, simulationStep);

				if (this->particleGenerator)
					this->writer->writeParticles(this->sd, this->st, *this->particleGenerator, currentSimulationTime, simulationStep, this->showParticleDensity);
			}


			if (this->sd.isLocalRelaxationEnabled())
			{
				HANDLE_ERROR(cudaMemcpy(&zerothMomMean, this->st_P.zerothMomSum.get(), sizeof(T), cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemcpy(&firstMomMean, this->st_P.firstMomSum.get(), sizeof(T), cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemcpy(&secondMomMean, this->st_P.secondMomSum.get(), sizeof(T), cudaMemcpyDeviceToHost));

				zerothMomMean /= this->sd.getMaxNodeCount();
				firstMomMean /= this->sd.getMaxNodeCount();
				secondMomMean /= this->sd.getMaxNodeCount();
			}

			if (BGKused)
			{
				collision_BGK(this->sd.getRelaxationConstant());
			}
			else
			{
				this->collision_CM(this->sd.getRelaxationConstant(), zerothMomMean, firstMomMean, secondMomMean);
			}
			
			if(this->particleGenerator)
				this->particleGenerator->updateParticles(simulationStep, this->st_P.u_L.get(), this->st_P.rho_L.get(), this->sd);

			if (ibm)
				ibm->calcInlet(this->st_P.u_L.get(), this->st_P.rho_L.get(), this->st_P.f_star.get(), this->st_P.f.get(), BGKused);


			++simulationStep;
			currentSimulationTime += this->sd.getTimeStep();

			T inlet_Mag = 0;
			if (ibm && ibm->getTag() == IBM_DYNAMIC)
			{
				inlet_Mag = ibm->update(currentSimulationTime, this->sd, particleGenerator.get());
			}

			if (this->adaptiveTimeStep)
			{
				//Problem mit überskalierten Zeitschritt
				uMax_mag = 0;
				HANDLE_ERROR(cudaMemcpy(this->st_P.uMag.get(), &uMax_mag, sizeof(T), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(&uMax_mag, this->st_P.uMag.get(), sizeof(T), cudaMemcpyDeviceToHost));

				if (uMax_mag < inlet_Mag)
					uMax_mag = inlet_Mag;

				if (uMax_mag > 0 && (uMax_mag < this->sd.get_uRef_L() - 0.02 || uMax_mag > this->sd.get_uRef_L() + 0.02))
				{
					this->sd.rescaleConFactors(uMax_mag, scale_u, scale_F);

					if (BGKused)
						this->rescaleMoments_BGK(scale_u, scale_F);
					else
						this->rescaleMoments_CM(scale_u, scale_F);

					if (this->ibm)
						this->ibm->rescaleVelocities(scale_u);

					if (this->particleGenerator)
						this->particleGenerator->rescaleParticleData(scale_u);
				}

			}
			cudaDeviceSynchronize();
		}
		loadingPoints += ".";
		std::cout << "\r";
		std::cout << loadingPoints << " " << 100 << "%";
		HANDLE_ERROR(cudaEventRecord(stop, 0));
		HANDLE_ERROR(cudaEventSynchronize(stop));

		if (visualData && this->ibm)
			this->writer->writePVDFile(*this->ibm, this->showParticleDensity);
		else if (visualData)
			this->writer->writePVDFile(this->showParticleDensity);
		
		
	}
	catch (const std::exception e)
	{
		std::cerr << e.what() << " Timestep: " << simulationStep << std::endl;
	}

	oStringStr << "Maximal-Simulation-Steps: " << simulationStep << "\n";

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	oStringStr << "Duration of the whole calculation: " << elapsedTime << "ms\n";

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
}
