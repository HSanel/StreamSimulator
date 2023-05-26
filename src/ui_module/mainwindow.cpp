#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(const SimRenderDimension &simDim, QWidget* parent) :
    QMainWindow(parent), _simRenderDim(simDim), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    try 
    {
        QVTKRenderWidget* _qvtkWidget = new QVTKRenderWidget(this);
        _simRenderer = std::make_unique<SimRenderer>(_qvtkWidget, _simRenderDim);
        ui->verticalLayout_vtk->addWidget(_qvtkWidget);
        ui->verticalLayout_vtk->update();
    }
    catch (std::exception e)
    {
        qDebug() << "EXCEPTION::" << e.what() << "\n";
    }
}

void MainWindow::updateData(const float *rho, const float *u)
{
    _simRenderer->updateData(rho, u);
    _simRenderer->showVelocity();
    _simRenderer->showDensity();

    _simRenderer->setCutClipPlane({ 8.0,0.0,0.0 }, { 1.0,0.0,0.0 });
    _simRenderer->activateCutting();
    _simRenderer->deactivateClipAndCut();
    _simRenderer->render();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showEvent(QShowEvent* event)
{
    _simRenderer->render();
}