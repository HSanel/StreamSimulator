#ifndef SIMREND
#define SIMREND

#include <array>
#include <vector>
#include <assert.h> 

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCylinderSource.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <vtkStructuredGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkClipDataSet.h>
#include <vtkMath.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <QVTKInteractor.h>
#include <QVTKRenderWidget.h>
#include <DataStructureAlg.h>

struct SimRenderDimension
{
	int xDim;
	int yDim;
	int zDim;
	int maxNodeCount;
	float GridSize = 1;

	SimRenderDimension(int xDim, int yDim, int zDim, float GridSize);
	SimRenderDimension(std::array<unsigned int,3> dim, float GridSize);
};

//TODO:: Transformation und Rotation fixieren und über Eingabe steuern

class SimRenderer
{
private:
	const QVTKRenderWidget* _vtkWidget = nullptr;
	const SimRenderDimension _simDim;

	vtkSmartPointer<vtkStructuredGrid> _structuredGrid;
	vtkSmartPointer<vtkDataSetMapper> _dataMapper;

	vtkSmartPointer<vtkPoints> _pointData;
	vtkSmartPointer<vtkFloatArray> _densityData;
	vtkSmartPointer<vtkFloatArray> _velocityData;

	vtkSmartPointer<vtkPlane> _clippingPlane;
	vtkSmartPointer<vtkCutter> _cutter;
	vtkSmartPointer<vtkClipDataSet> _clipper;

	vtkSmartPointer<vtkActor> _actor;
	vtkSmartPointer<vtkRenderer> _renderer;

public:
	SimRenderer(const QVTKRenderWidget* vtkWidget, const SimRenderDimension);

	void setCutClipPlane(std::array<double, 3> origin, std::array<double, 3> normal);
	void activateCliping();
	void activateCutting();
	void deactivateClipAndCut();
	
	void showDensity();
	void showVelocity(int axis=-1);

	void updateData(const std::vector<float> *density, const std::vector<std::array<float,3>> *velocity);
	void updateData(const float* density, const float* velocity);
	void render();
};

#endif // !SIMREND