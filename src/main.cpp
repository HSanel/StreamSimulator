#include <QApplication>
#include "ui_module/mainwindow.h"


int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();



	return EXIT_SUCCESS;
}