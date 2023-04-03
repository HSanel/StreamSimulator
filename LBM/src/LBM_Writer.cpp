#include "LBM_Writer.h"

template class LBM_Writer<float,2>;
template class LBM_Writer<double, 2>;
template class LBM_Writer<float,3>;
template class LBM_Writer<double, 3>;

#pragma region writer_spec_2D
template<typename T>
LBM_Writer_Specialised<T, 2>::LBM_Writer_Specialised(int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG)
	:LBM_Writer_Base(stepsPerFrame, path, im_used, particles_used, L_UNIT_TAG) {}

template<typename T>
void LBM_Writer_Specialised<T, 2>::setPoints(const SimDomain<T,2> &sd)
{
	for (int y = 0; y < sd.getGridDim_L(1); ++y)
		for (int x = 0; x < sd.getGridDim_L(0); ++x)
		{
			if (L_UNIT_TAG == 2 || L_UNIT_TAG == 3)
				point_Data->SetPoint(y * sd.getGridDim_L(0) + x, x, y, 0);
			else
				point_Data->SetPoint(y * sd.getGridDim_L(0) + x, static_cast<float>(x) * sd.getGridSize(), static_cast<float>(y) * sd.getGridSize(), 0);
		}
}
#pragma endregion

LBM_Writer_Base::LBM_Writer_Base(int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG)
	:stepsPerFrame(stepsPerFrame), path(path), im_used(im_used), particles_used(particles_used), L_UNIT_TAG(L_UNIT_TAG) {}

template<typename T>
LBM_Writer_Specialised<T,3>::LBM_Writer_Specialised(int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG)
	:LBM_Writer_Base(stepsPerFrame, path, im_used, particles_used, L_UNIT_TAG) {}

template<typename T, size_t D>
LBM_Writer<T,D>::LBM_Writer(const SimDomain<T, D>& sd, const int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG)
	: LBM_Writer_Specialised<T,D>(stepsPerFrame, path, im_used, particles_used, L_UNIT_TAG)
{
	if (boost::filesystem::exists(path))
		boost::filesystem::remove_all(path);

	boost::filesystem::create_directories(path);

	structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
	point_Data = vtkSmartPointer<vtkPoints>::New();
	density_Data = vtkSmartPointer<vtkFloatArray>::New();
	velocity_Data = vtkSmartPointer<vtkFloatArray>::New();

	structuredGrid->SetDimensions(sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2));
	velocity_Data->SetNumberOfComponents(3);
	velocity_Data->SetNumberOfTuples(sd.getMaxNodeCount());
	density_Data->SetNumberOfValues(sd.getMaxNodeCount());
	point_Data->SetNumberOfPoints(sd.getMaxNodeCount());

	velocity_Data->SetName("u");
	density_Data->SetName("Density");

	this->setPoints(sd); //specialisation for 2D or 3D	
	structuredGrid->SetPoints(point_Data);
			
	if (im_used)
	{
		//writeImBody will throw an exception if im_used=false!
		im_polyData = vtkSmartPointer< vtkPolyData>::New();
		im_point_Data = vtkSmartPointer< vtkPoints>::New();
		im_vertices = vtkSmartPointer<vtkCellArray>::New();
	}

	if (particles_used)
	{
		//writeImBody will throw an exception if im_used=false!
		particle_structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
		particle_structuredGrid->SetDimensions(sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2));
		particle_structuredGrid->SetPoints(point_Data);

		particleDensity_Data = vtkSmartPointer<vtkFloatArray>::New();
		particleDensity_Data->SetNumberOfValues(sd.getMaxNodeCount());
		particleDensity_Data->SetName("Particle Density");
		particles_polyData = vtkSmartPointer< vtkPolyData>::New();
		particles_point_Data = vtkSmartPointer< vtkPoints>::New();
		particles_vertices = vtkSmartPointer<vtkCellArray>::New();
		particles_diameter = vtkSmartPointer<vtkFloatArray>::New();

		particles_diameter->SetName("Particle-Diameter");
	}
}

template<typename T>
void LBM_Writer_Specialised<T,3>::setPoints(const SimDomain<T, 3>& sd)
{
	for (int z = 0; z < sd.getGridDim_L(2); ++z)
		for (int y = 0; y < sd.getGridDim_L(1); ++y)
			for (int x = 0; x < sd.getGridDim_L(0); ++x)
			{
				if (L_UNIT_TAG == 2 || L_UNIT_TAG == 3)
					point_Data->SetPoint((z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x, x, y, z);
				else
					point_Data->SetPoint((z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x, static_cast<float>(x) * sd.getGridSize(), static_cast<float>(y) * sd.getGridSize(), static_cast<float>(z) * sd.getGridSize());
			}
}

//----------------------------------------------------
//base methods
template<typename T, size_t D>
const int LBM_Writer<T,D>::getStepsPerFrame() const { return stepsPerFrame; }

template<typename T, size_t D>
void LBM_Writer<T,D>::setStepsPerFrame(int stepPerFrame) { this->stepsPerFrame = stepPerFrame; };

template<typename T, size_t D>
void LBM_Writer<T,D>::setDestination(writeDestination destination) { this->destination = destination; }

template<typename T, size_t D>
writeDestination LBM_Writer<T,D>::getDestination() const { return this->destination; }

template<typename T, size_t D>
void LBM_Writer<T,D>::writePVDFile(const bool& showParticleDensity)
{
	std::stringstream ss;
	ss << path << "/LBM_Output.pvd";

	std::ofstream myfile;
	myfile.open(ss.str().c_str(), std::ios::out);
	myfile << "<?xml version=\"1.0\"?>\n";
	myfile << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"  compressor=\"vtkZLibDataCompressor\">\n";
	myfile << "<Collection>\n";

	for (int i = 0; i < timePoints.size(); ++i)
	{
		myfile << " <DataSet timestep=\"" << this->timePoints[i] << "\" group=\"0\" part=\"0\" file=\"output" << D << "D" << i << ".vts\"/>\n";
	}

	myfile << "</Collection>\n";
	myfile << "</VTKFile>" << std::flush;
	myfile.close();

	if (particles_used)
	{
		std::stringstream ss_part;
		std::ofstream myfile_part;
		myfile_part.clear();
		ss_part << path << "/Particles_Output.pvd";
		myfile_part.open(ss_part.str().c_str(), std::ios::out);
		myfile_part << "<?xml version=\"1.0\"?>\n";
		myfile_part << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"  compressor=\"vtkZLibDataCompressor\">\n";

		myfile_part << "<Collection>\n";


		for (int i = 0; i < timePoints.size(); ++i)
		{
			myfile_part << " <DataSet timestep=\"" << this->timePoints[i] << "\" group=\"0\" part=\"0\" file=\"particles" << D << "D" << i << ".vtp\"/>\n";
		}


		myfile_part << "</Collection>\n";
		myfile_part << "</VTKFile>" << std::flush;
		myfile_part.close();

		//paricle density
		if (showParticleDensity)
		{
			std::stringstream ss_partDens;
			std::ofstream myfile_partDens;
			myfile_partDens.clear();
			ss_partDens << path << "/ParticleDensity_Output.pvd";
			myfile_partDens.open(ss_partDens.str().c_str(), std::ios::out);
			myfile_partDens << "<?xml version=\"1.0\"?>\n";
			myfile_partDens << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"  compressor=\"vtkZLibDataCompressor\">\n";

			myfile_partDens << "<Collection>\n";


			for (int i = 0; i < timePoints.size(); ++i)
			{
				myfile_partDens << " <DataSet timestep=\"" << this->timePoints[i] << "\" group=\"0\" part=\"0\" file=\"outputParticleDensity" << D << "D" << i  << ".vts\"/>\n";
			}


			myfile_partDens << "</Collection>\n";
			myfile_partDens << "</VTKFile>" << std::flush;
			myfile_partDens.close();
		}
	}
}

template<typename T, size_t D>
void LBM_Writer<T,D>::writePVDFile(const IBMethod<T,D> &ibm, const bool& showParticleDensity)
{
	writePVDFile(showParticleDensity);

	if (ibm.getTag() == IBM_DYNAMIC)
	{
		std::stringstream ss2;
		std::ofstream myfile2;
		myfile2.clear();
		ss2 << path << "/IBM_Bodies.pvd";
		myfile2.open(ss2.str().c_str(), std::ios::out);
		myfile2 << "<?xml version=\"1.0\"?>\n";
		myfile2 << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"  compressor=\"vtkZLibDataCompressor\">\n";

		myfile2 << "<Collection>\n";


		for (int i = 0; i < this->timePoints.size(); ++i)
		{
			myfile2 << " <DataSet timestep=\"" << this->timePoints[i] << "\" group=\"\" part=\"1\" file=\"outputPoly" << D << "D" << i  << ".vtp\"/>\n";
		}


		myfile2 << "</Collection>\n";
		myfile2 << "</VTKFile>" << std::flush;
		myfile2.close();
	}
}

template<typename T, size_t D>
void LBM_Writer<T, D>::writePVDFile(const IBMethod_P<T, D>& ibm, const bool& showParticleDensity)
{
	writePVDFile(showParticleDensity);

	if (ibm.getTag() == IBM_DYNAMIC)
	{
		std::stringstream ss2;
		std::ofstream myfile2;
		myfile2.clear();
		ss2 << path << "/IBM_Bodies.pvd";
		myfile2.open(ss2.str().c_str(), std::ios::out);
		myfile2 << "<?xml version=\"1.0\"?>\n";
		myfile2 << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"  compressor=\"vtkZLibDataCompressor\">\n";

		myfile2 << "<Collection>\n";


		for (int i = 0; i < this->timePoints.size(); ++i)
		{
			myfile2 << " <DataSet timestep=\"" << this->timePoints[i] << "\" group=\"\" part=\"1\" file=\"outputPoly" << D << "D" << i << ".vtp\"/>\n";
		}


		myfile2 << "</Collection>\n";
		myfile2 << "</VTKFile>" << std::flush;
		myfile2.close();
	}
}

template<typename T, size_t D>
bool LBM_Writer<T,D>::writeMomentsToVTKFile(const SimDomain<T,D> &sd, const SimState<T, D>& st, const T& currentTime, const int &timeStep)
{
	int fileIndex = timeStep / stepsPerFrame;
	this->timePoints.push_back(currentTime);

	T C_u = 1.0, C_rho = 1.0;

	if (!(L_UNIT_TAG == 1 || L_UNIT_TAG == 3))
	{
		C_u = sd.getC_u();
		C_rho = sd.getRho();
	}

	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
	{
		density_Data->SetValue(pos, st.rho_L[pos] * C_rho);

		auto velOut = vec<T, 3>{};
		auto vel = st.u_L[pos] * C_u;
		std::swap_ranges(vel.begin(), vel.end(), velOut.begin());

		
		velocity_Data->SetTuple(pos, velOut.data());
		bool isInfinityRho = std::isinf(st.rho_L[pos]);
		bool isInfinityVel = false;
		for (int i = 0; i < D; ++i)
			isInfinityVel = isinf(st.u_L[pos][i]);

		if (isInfinityRho || isInfinityVel)
			throw std::invalid_argument("ERROR::VALUE::Infinity Value of density or velocity::");
	}

	structuredGrid->GetPointData()->SetScalars(density_Data);
	structuredGrid->GetPointData()->SetVectors(velocity_Data);

	std::stringstream ss;
	ss << path << "/" << "output" << D << "D" << fileIndex << ".vts";
	vtkSmartPointer<vtkXMLStructuredGridWriter> writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(ss.str().c_str());
	writer->SetInputData(structuredGrid);
	writer->Write();


	return true;
}

template<typename T, size_t D>
void LBM_Writer<T, D>::writeImBody(const SimDomain<T, D>& sd, const IBMethod<T, D>& ibm, const int &timeStep)
{
	int fileIndex = timeStep / stepsPerFrame;

	if (!im_used)
		throw std::runtime_error("IM_BODY ERROR::Immersed Body can't be wrote, because im_used=false.");

	if (ibm.getTag() == IBM_DYNAMIC || (ibm.getTag() == IBM_STATIC && fileIndex == 0))
	{
		T C_l = 1.0;
		if (!(L_UNIT_TAG == 2 || L_UNIT_TAG == 3))
		{
			C_l = sd.getGridSize();
		}
		im_point_Data->Reset();
		im_vertices->Reset();
		im_point_Data->SetNumberOfPoints(ibm.getMaxPointCount());

		int pointCount = 0;

		for (int n = 0; n < ibm.getBodyCount(); n++)
		{
			const IM_BODY<T, D>* body = ibm.getBodyAt(n);
			for (int i = 0; i < body->samples.size(); ++i)
			{

				vtkIdType pid[1];

				auto posOut = vec<T, 3>{};
				auto oVec = body->samples.at(i) * C_l;
				std::swap_ranges(oVec.begin(), oVec.end(), posOut.begin());

				pid[0] = pointCount;
				im_point_Data->SetPoint(pointCount, posOut.data());

				vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
				vertex->GetPointIds()->SetId(0, pointCount);

				im_vertices->InsertNextCell(vertex);
				++pointCount;
			}

			for (int i = 0; i < body->inletSamples.size(); ++i)
			{

				vtkIdType pid[1];

				auto posOut = vec<T, 3>{};
				auto oVec = body->inletSamples.at(i) * C_l;
				std::swap_ranges(oVec.begin(), oVec.end(), posOut.begin());

				pid[0] = pointCount;
				im_point_Data->SetPoint(pointCount, posOut.data());

				vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
				vertex->GetPointIds()->SetId(0, pointCount);

				im_vertices->InsertNextCell(vertex);
				++pointCount;
			}
		}
	
		im_polyData->SetPoints(im_point_Data);
		im_polyData->SetVerts(im_vertices);

		std::stringstream ss;
		ss << path << "/" << "outputPoly" << D << "D" << fileIndex << ".vtp";
		vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer->SetFileName(ss.str().c_str());
		writer->SetInputData(im_polyData);
		writer->Write();
	}
}

template<typename T, size_t D>
void LBM_Writer<T, D>::writeImBody(const SimDomain<T, D>& sd, const IBMethod_P<T, D>& ibm, const int &timeStep)
{
	int fileIndex = timeStep / stepsPerFrame;

	if (!im_used)
		throw std::runtime_error("IM_BODY ERROR::Immersed Body can't be wrote, because im_used=false.");

	if (ibm.getTag() == IBM_DYNAMIC || (ibm.getTag() == IBM_STATIC && fileIndex == 0))
	{
		T C_l = 1.0;
		if (!(L_UNIT_TAG == 2 || L_UNIT_TAG == 3))
		{
			C_l = sd.getGridSize();
		}
		im_point_Data->Reset();
		im_vertices->Reset();
		im_point_Data->SetNumberOfPoints(ibm.getMaxPointCount());

		int pointCount = 0;

		for (int n = 0; n < ibm.getBodyCount(); n++)
		{
			const IM_BODY<T, D>* body = ibm.getBodyAt(n);
			for (int i = 0; i < body->samples.size(); ++i)
			{

				vtkIdType pid[1];

				auto posOut = vec<T, 3>{};
				auto oVec = body->samples.at(i) * C_l;
				std::swap_ranges(oVec.begin(), oVec.end(), posOut.begin());

				pid[0] = pointCount;
				im_point_Data->SetPoint(pointCount, posOut.data());

				vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
				vertex->GetPointIds()->SetId(0, pointCount);

				im_vertices->InsertNextCell(vertex);
				++pointCount;
			}

			for (int i = 0; i < body->inletSamples.size(); ++i)
			{

				vtkIdType pid[1];

				auto posOut = vec<T, 3>{};
				auto oVec = body->inletSamples.at(i) * C_l;
				std::swap_ranges(oVec.begin(), oVec.end(), posOut.begin());

				pid[0] = pointCount;
				im_point_Data->SetPoint(pointCount, posOut.data());

				vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
				vertex->GetPointIds()->SetId(0, pointCount);

				im_vertices->InsertNextCell(vertex);
				++pointCount;
			}
		}

		im_polyData->SetPoints(im_point_Data);
		im_polyData->SetVerts(im_vertices);

		std::stringstream ss;
		ss << path << "/" << "outputPoly" << D << "D" << fileIndex << ".vtp";
		vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer->SetFileName(ss.str().c_str());
		writer->SetInputData(im_polyData);
		writer->Write();
	}
}

template<typename T, size_t D>
bool LBM_Writer<T,D>::writeMomentsToTXTFile(const SimDomain<T, D>& sd, const SimState<T, D>& st, const T &currentSimulationTime, const int &timeStep)
{
	T C_l = 1.0;
	T C_u = 1.0;
	T C_rho = 1.0;

	if (!(L_UNIT_TAG == 2 || L_UNIT_TAG == 3))
	{
		C_l = sd.getGridSize();
		C_u = sd.getC_u();
		C_rho = sd.getRho();
	}

	std::stringstream ss;
	ss << path << "/Output"<< D <<"D_" << timeStep << ".txt";
	std::ofstream myfile;
	myfile.open(ss.str().c_str(), std::ios::out);

	myfile << "Time: " << currentSimulationTime << "\n";
	myfile << "-----------------------------------------\n";

	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
	{
		if (L_UNIT_TAG == 1 || L_UNIT_TAG == 3)
			myfile <<pos << ": rho=" << st.rho_L[pos] << ", u=(" << st.u_L[pos] << ")\n";
		else
			myfile << pos << ": rho=" << st.rho_L[pos] * C_rho << ", u=(" << st.u_L[pos] * C_u << ")\n";

		myfile << "\n";
	}
	myfile << std::flush;
	myfile.close();
	return false;
}

template<typename T, size_t D>
void LBM_Writer<T, D>::writeParticles(const SimDomain<T, D>& sd, const SimState<T, D>& st, const ParticleGenerator<T, D>& particleGenerator,const T &currentSimulationTime, const int &timeStep, const bool& showParticleDensity)
{
	int fileIndex = timeStep / stepsPerFrame;

	if (!particles_used)
		throw std::runtime_error("PARTICLE_GENERATOR ERROR::Particles can't be wrote, because particle_used=false.");

	T C_l = 1.0;
	if (!(L_UNIT_TAG == 2 || L_UNIT_TAG == 3))
	{
		C_l = sd.getGridSize();
	}
	const std::vector<ParticleData<T, D>>& particles = particleGenerator.getParticles();

	particles_point_Data->Reset();
	particles_vertices->Reset();
	particles_diameter->Reset();
	particles_point_Data->SetNumberOfPoints(particleGenerator.getActiveParticleCount());
	particles_diameter->SetNumberOfValues(particleGenerator.getActiveParticleCount());
	particles_vertices->SetNumberOfCells(0);
	
	bool particlesSet = false;

	for (int i = 0; i < particleGenerator.getActiveParticleCount(); ++i)
	{
		if (!(particles[i].is_active) || currentSimulationTime < particles[i].time)
			continue;
		
		vtkIdType pid[1];

		auto posOut = vec<T, 3>{};
		auto oVec = particles[i].position * C_l;
		std::swap_ranges(oVec.begin(), oVec.end(), posOut.begin());

		particles_diameter->SetValue(i, particles[i].diameter * C_l);
		pid[0] = i;
		particles_point_Data->SetPoint(i, posOut.data());

		vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
		vertex->GetPointIds()->SetId(0, i);

		particles_vertices->InsertNextCell(vertex);
		particlesSet = true;
	}

	if (particlesSet)
	{
		particles_polyData->SetPoints(particles_point_Data);
		particles_polyData->SetVerts(particles_vertices);
		particles_polyData->GetPointData()->SetScalars(particles_diameter);
	}

	std::stringstream ss;
	ss << path << "/" << "particles" << D << "D" << fileIndex << ".vtp";
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer->SetFileName(ss.str().c_str());
	writer->SetInputData(particles_polyData);
	writer->Write();

	//particle density
	if (showParticleDensity)
	{
		for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
		{
			particleDensity_Data->SetValue(pos, st.particleDensity[pos]);


			bool isInfinityDensity = std::isinf(st.particleDensity[pos]);

			if (isInfinityDensity)
				throw std::invalid_argument("ERROR::VALUE::Infinity Value of particle density::");
		}

		particle_structuredGrid->GetPointData()->SetScalars(particleDensity_Data);

		std::stringstream ssDens;
		ssDens << path << "/" << "outputParticleDensity" << D << "D" << fileIndex << ".vts";
		vtkSmartPointer<vtkXMLStructuredGridWriter> writerDens = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
		writerDens->SetFileName(ssDens.str().c_str());
		writerDens->SetInputData(particle_structuredGrid);
		writerDens->Write();
	}
}


template<typename T, size_t D>
void LBM_Writer<T, D>::writeParticles(const SimDomain<T, D>& sd, const SimState<T, D>& st, const ParticleGenerator_P<T, D>& particleGenerator, const T& currentSimulationTime, const int &timeStep, const bool & showParticleDensity)
{
	int fileIndex = timeStep / stepsPerFrame;

	if (!particles_used)
		throw std::runtime_error("PARTICLE_GENERATOR ERROR::Particles can't be wrote, because particle_used=false.");

	T C_l = 1.0;
	if (!(L_UNIT_TAG == 2 || L_UNIT_TAG == 3))
	{
		C_l = sd.getGridSize();
	}
	const std::vector<ParticleData<T, D>>& particles = particleGenerator.getParticles();
	int activeParticleCount = particleGenerator.getActiveParticleCount();
	particles_point_Data->Reset();
	particles_vertices->Reset();
	particles_diameter->Reset();

	particles_point_Data->SetNumberOfPoints(activeParticleCount);
	particles_diameter->SetNumberOfValues(activeParticleCount);

	bool particlesSet = false;

	int indActive = 0;
	for (int i = 0; i < particles.size(); ++i)
	{
		if (!(particles[i].is_active) || currentSimulationTime < particles[i].time)
			continue;

		vtkIdType pid[1];

		auto posOut = vec<T, 3>{};
		auto oVec = particles[i].position * C_l;
		
		std::swap_ranges(oVec.begin(), oVec.end(), posOut.begin());

		particles_diameter->SetValue(indActive, particles[i].diameter * C_l);
		pid[0] = indActive;
		particles_point_Data->SetPoint(indActive, posOut.data());

		vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
		vertex->GetPointIds()->SetId(0, indActive);

		particles_vertices->InsertNextCell(vertex);
		particlesSet = true;
		indActive++;
	}

	if (particlesSet)
	{
		particles_polyData->SetPoints(particles_point_Data);
		particles_polyData->SetVerts(particles_vertices);
		particles_polyData->GetPointData()->SetScalars(particles_diameter);
	}

	std::stringstream ss;
	ss << path << "/" << "particles" << D << "D" << fileIndex << ".vtp";
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer->SetFileName(ss.str().c_str());
	writer->SetInputData(particles_polyData);
	writer->Write();

	//particle density
	if (showParticleDensity)
	{
		for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
		{
			particleDensity_Data->SetValue(pos, st.particleDensity[pos]);


			bool isInfinityDensity = std::isinf(st.particleDensity[pos]);

			if (isInfinityDensity)
				throw std::invalid_argument("ERROR::VALUE::Infinity Value of particle density::");
		}

		particle_structuredGrid->GetPointData()->SetScalars(particleDensity_Data);

		std::stringstream ssDens;
		ssDens << path << "/" << "outputParticleDensity" << D << "D" << fileIndex << ".vts";
		vtkSmartPointer<vtkXMLStructuredGridWriter> writerDens = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
		writerDens->SetFileName(ssDens.str().c_str());
		writerDens->SetInputData(particle_structuredGrid);
		writerDens->Write();
	}
}


