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
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    std::unique_ptr<SimRenderer> simRenderer;

    virtual void showEvent(QShowEvent* event) override;

};

#endif // MAINWINDOW_H
