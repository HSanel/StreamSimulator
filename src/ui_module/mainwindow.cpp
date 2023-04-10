#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    try 
    {
        auto simDim = SimRenderDimension{ 10, 10, 10, 1.f };
        QVTKRenderWidget* _qvtkWidget = new QVTKRenderWidget(this);
        simRenderer = std::make_unique<SimRenderer>(_qvtkWidget, simDim);
        ui->verticalLayout_vtk->addWidget(_qvtkWidget);
        ui->verticalLayout_vtk->update();

        std::vector<float> density(1000);
        std::vector<std::array<float, 3>> velocity(1000);

        for (int i = 0; i < 1000; ++i)
        {
            if (i > 500)
            {
                density[i] = rand() / static_cast<float>(RAND_MAX);
                
            }
            else
            {
                density[i] = 0.5;
            }
            velocity[i] = std::array<float, 3>{rand() / static_cast<float>(RAND_MAX), 0.5f, 0.5f};
        }

        simRenderer->updateData(density, velocity);
        simRenderer->showVelocity();
        simRenderer->showDensity();

        simRenderer->setCutClipPlane({ 8.0,0.0,0.0 }, { 1.0,0.0,0.0 });
        simRenderer->activateCutting();
        simRenderer->deactivateClipAndCut();
    }
    catch (std::exception e)
    {
        qDebug() << "EXCEPTION::" << e.what() << "\n";
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showEvent(QShowEvent* event)
{
    simRenderer->render();
}