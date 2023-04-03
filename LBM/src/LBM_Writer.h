#ifndef LBM_WRITER
#define LBM_WRITER

#include <string>
#include <vtkVersion.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkVertex.h>
#include <boost/filesystem.hpp>
#include "LBM_Types.h"
#include "ImmersedBoundaryMethod.h"
#include "ImmersedBoundaryMethod_P.h"
#include "ImmersedBody.h"
#include "SimDomain.h"
#include "SimState.h"
#include "ParticleGenerator.h"
#include "ParticleGenerator_P.h"



//using T = globalType;

enum writeDestination { TXT_FILE, VTK_FILE };

class LBM_Writer_Base
{
	//LBM_Writer_Base(const LBM_Writer_Base&) = delete;
	//LBM_Writer_Base& operator=(const LBM_Writer_Base&) = delete;
	//LBM_Writer_Base(LBM_Writer_Base&&) = delete;
	//LBM_Writer_Base& operator=(LBM_Writer_Base&&) = delete;
protected:
	int  stepsPerFrame;
	const bool im_used;
	const bool particles_used;
	const std::string path;
	const unsigned char L_UNIT_TAG;
	writeDestination destination = VTK_FILE;
	vtkSmartPointer<vtkStructuredGrid> structuredGrid;
	vtkSmartPointer<vtkPoints> point_Data;
	vtkSmartPointer<vtkFloatArray> density_Data;
	vtkSmartPointer<vtkFloatArray> velocity_Data;

	vtkSmartPointer<vtkPolyData> im_polyData;
	vtkSmartPointer<vtkPoints> im_point_Data;
	vtkSmartPointer<vtkCellArray> im_vertices;

	vtkSmartPointer<vtkStructuredGrid> particle_structuredGrid;
	vtkSmartPointer<vtkFloatArray> particleDensity_Data;
	vtkSmartPointer<vtkPolyData> particles_polyData;
	vtkSmartPointer<vtkPoints> particles_point_Data;
	vtkSmartPointer<vtkCellArray> particles_vertices;
	vtkSmartPointer<vtkFloatArray> particles_diameter;
	LBM_Writer_Base(int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG);
};

template<typename T, size_t D> class LBM_Writer_Specialised;

template<typename T>
class LBM_Writer_Specialised<T,2>:public LBM_Writer_Base
{
protected:
	std::vector<T> timePoints;
protected:
	void setPoints(const SimDomain<T, 2>& sd);
	LBM_Writer_Specialised(int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG);

};

template<typename T>
class LBM_Writer_Specialised<T,3> :public LBM_Writer_Base
{
protected:
	std::vector<T> timePoints;
protected:
	void setPoints(const SimDomain<T, 3>& sd);
	LBM_Writer_Specialised(int stepsPerFrame, const std::string path, const bool im_used, const bool particles_used, const unsigned char L_UNIT_TAG);
	
};

template<typename T, size_t D>
class LBM_Writer:LBM_Writer_Specialised<T,D>
{
private:
	LBM_Writer_Specialised<T,D>::stepsPerFrame;
	LBM_Writer_Specialised<T,D>::im_used;
	LBM_Writer_Specialised<T,D>::path;
	LBM_Writer_Specialised<T,D>::L_UNIT_TAG;
	LBM_Writer_Specialised<T,D>::destination;

	LBM_Writer_Specialised<T,D>::structuredGrid;
	LBM_Writer_Specialised<T,D>::point_Data;
	LBM_Writer_Specialised<T,D>::density_Data;
	LBM_Writer_Specialised<T,D>::velocity_Data;

	LBM_Writer_Specialised<T,D>::im_polyData;
	LBM_Writer_Specialised<T,D>::im_point_Data;
	LBM_Writer_Specialised<T,D>::im_vertices;

	LBM_Writer_Specialised<T,D>::setPoints;
public:
	LBM_Writer(const SimDomain<T, D>& sd, int stepsPerFrame, const std::string path,  const bool im_used = false, const bool particles_used = false, const unsigned char L_UNIT_TAG = 0);

	void setDestination(writeDestination destination);
	writeDestination getDestination() const;
	const int getStepsPerFrame() const;
	void setStepsPerFrame(int stepPerFrame);

	void writePVDFile(const bool& showParticleDensity);
	void writePVDFile(const IBMethod<T, D>&ibm, const bool& showParticleDensity);
	void writePVDFile(const IBMethod_P<T, D>& ibm, const bool& showParticleDensity);

	bool writeMomentsToVTKFile(const SimDomain<T,D> &sd, const SimState<T, D>& st, const T& currentTime, const int &timeStep);
	bool writeMomentsToTXTFile(const SimDomain<T,D> &sd, const SimState<T, D>& st, const T &currentSimulationTime, const int &timeStep);

	void writeImBody(const SimDomain<T,D> &sd, const IBMethod<T,D>& ibm, const int &timeStep);
	void writeImBody(const SimDomain<T,D> &sd, const IBMethod_P<T, D>& ibm, const int &timeStep);

	void writeParticles(const SimDomain<T,D> &sd, const SimState<T, D>& st, const ParticleGenerator<T, D>& particleGenerator,const T &currentSimulationTime, const int &timeStep, const bool& showParticleDensity);
	void writeParticles(const SimDomain<T,D> &sd, const SimState<T, D>& st, const ParticleGenerator_P<T, D>& particleGenerator, const T &currentSimulationTime, const int &timeStep,const bool &showParticleDensity);
};
#endif