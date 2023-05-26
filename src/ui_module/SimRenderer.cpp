#include "SimRenderer.h"

SimRenderDimension::SimRenderDimension(int xDim, int yDim, int zDim, float GridSize) : xDim(xDim), yDim(yDim), zDim(zDim), maxNodeCount(xDim * yDim* zDim), GridSize(GridSize) {}
SimRenderDimension::SimRenderDimension(std::array<unsigned int, 3> dim, float GridSize) : xDim(dim[0]), yDim(dim[1]), zDim(dim[2]), maxNodeCount(dim[0] * dim[1] * dim[2]), GridSize(GridSize) {}

SimRenderer::SimRenderer(const QVTKRenderWidget* vtkWidget, const SimRenderDimension simDim): _vtkWidget(vtkWidget), _simDim(simDim)
{
	_structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
	_dataMapper = vtkSmartPointer<vtkDataSetMapper>::New();
	_actor = vtkSmartPointer<vtkActor>::New();
	_renderer = vtkSmartPointer<vtkRenderer>::New();
	_clippingPlane = vtkSmartPointer<vtkPlane>::New();
	_cutter = vtkSmartPointer<vtkCutter>::New();
	_clipper = vtkSmartPointer<vtkClipDataSet>::New();
	
	_pointData = vtkSmartPointer<vtkPoints>::New();
	_densityData = vtkSmartPointer<vtkFloatArray>::New();
	_velocityData = vtkSmartPointer<vtkFloatArray>::New();

	_structuredGrid->SetDimensions(simDim.xDim, simDim.yDim, simDim.zDim);
	_velocityData->SetNumberOfComponents(3);
	_velocityData->SetNumberOfTuples(simDim.maxNodeCount);
	_densityData->SetNumberOfValues(simDim.maxNodeCount);
	_pointData->SetNumberOfPoints(simDim.maxNodeCount);

	_velocityData->SetName("u");
	_densityData->SetName("Density");

	for (int z = 0; z < simDim.zDim; ++z)
		for (int y = 0; y < simDim.yDim; ++y)
			for (int x = 0; x < simDim.xDim; ++x)
			{
				_pointData->SetPoint((z * simDim.yDim + y) * simDim.xDim + x, x * simDim.GridSize, y * simDim.GridSize, z * simDim.GridSize);
			}

	for (int pos = 0; pos < _simDim.maxNodeCount; ++pos)
	{
		_densityData->SetValue(pos, 0.f);
		_velocityData->SetTuple(pos, std::array<float, 3>{0.f, 0.f, 0.f}.data());
	}

	_structuredGrid->GetPointData()->SetScalars(_densityData);
	_structuredGrid->GetPointData()->SetVectors(_velocityData);

	_structuredGrid->SetPoints(_pointData);
	_dataMapper->SetInputData(_structuredGrid);

	_dataMapper->SetScalarModeToUsePointData();
	_dataMapper->SelectColorArray("Density");

	_actor->SetMapper(_dataMapper);
	_renderer->AddActor(_actor);
	_renderer->ResetCamera();
	_renderer->GetActiveCamera()->Zoom(1.5);
	_vtkWidget->renderWindow()->AddRenderer(_renderer);
	_vtkWidget->renderWindow()->Render();
}

void SimRenderer::setCutClipPlane(std::array<double, 3> origin, std::array<double, 3> normal)
{
	vtkMath::Normalize(normal.data());
	_clippingPlane->SetOrigin(origin.data());
	_clippingPlane->SetNormal(normal.data());
}

void SimRenderer::activateCliping()
{
	_clipper->SetInputData(_structuredGrid);
	_clipper->SetClipFunction(_clippingPlane);
	_dataMapper->SetInputConnection(_clipper->GetOutputPort());
	_dataMapper->Update();
}

void SimRenderer::activateCutting()
{
	_cutter->SetInputData(_structuredGrid);
	_cutter->SetCutFunction(_clippingPlane);
	_dataMapper->SetInputConnection(_cutter->GetOutputPort());
	_dataMapper->Update();
}

//TODO: Deactivate Filters
void SimRenderer::deactivateClipAndCut()
{
	//_dataMapper->RemoveAllClippingPlanes();
	//_dataMapper->SetInputConnection(nullptr);
	//_dataMapper->Update();
}

void SimRenderer::showDensity()
{
	_dataMapper->SelectColorArray("Density");
	_dataMapper->Update();
}

//TODO: Shows only components but not magnitude
void SimRenderer::showVelocity(int axis)
{
	_dataMapper->SetScalarModeToUsePointFieldData();
	_dataMapper->SelectColorArray("u");
	if(axis >= 0 && axis < 3)
		_dataMapper->SetArrayComponent(axis);

	_dataMapper->Update();
}

void SimRenderer::render()
{
	_vtkWidget->renderWindow()->Render();
}

void SimRenderer::updateData(const std::vector<float> *density, const std::vector<std::array<float, 3>> *velocity)
{
	assert(density.size() == _simDim.maxNodeCount || "ERROR:: Data dimension doesn't correspond to the Domain Dimension");
	assert(velocity.size() == _simDim.maxNodeCount || "ERROR:: Velocity dimension doesn't correspond to the Domain Dimension");

	for (int pos = 0; pos < _simDim.maxNodeCount ; ++pos)
	{
		_densityData->SetValue(pos, (*density)[pos]);
		_velocityData->SetTuple(pos, (*velocity)[pos].data());
	}
	_structuredGrid->GetPointData()->SetScalars(_densityData);
	_structuredGrid->GetPointData()->SetVectors(_velocityData);
	_dataMapper->Update();
}

void SimRenderer::updateData(const float* density, const float* velocity)
{
	assert(density.size() == _simDim.maxNodeCount || "ERROR:: Data dimension doesn't correspond to the Domain Dimension");
	assert(velocity.size() == _simDim.maxNodeCount || "ERROR:: Velocity dimension doesn't correspond to the Domain Dimension");

	for (int pos = 0; pos < _simDim.maxNodeCount; ++pos)
	{
		std::array<float, 3> temp_array{ velocity[arrayLayout(pos,0)], velocity[arrayLayout(pos,1)] ,velocity[arrayLayout(pos,2)] };
		_densityData->SetValue(pos, density[pos]);
		_velocityData->SetTuple(pos, temp_array.data());
	}
	_structuredGrid->GetPointData()->SetScalars(_densityData);
	_structuredGrid->GetPointData()->SetVectors(_velocityData);
	_dataMapper->Update();
}