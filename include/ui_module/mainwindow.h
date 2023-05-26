#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <QMainWindow>
#include <qdebug.h>
#include <QVTKRenderWidget.h>
#include "SimRenderer.h"
#include <iostream>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const SimRenderDimension &simDim, QWidget *parent = nullptr);
    ~MainWindow();

    void updateData(const float* rho, const float* u);
private:
    SimRenderDimension _simRenderDim;
    std::unique_ptr<SimRenderer> _simRenderer;

    Ui::MainWindow *ui;
    virtual void showEvent(QShowEvent* event) override;


};

#endif // MAINWINDOW_H
