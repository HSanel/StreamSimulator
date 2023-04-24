#include <QApplication>
#include <ui_module/Mainwindow.h>
#include <lbm_module/LBMSolver.h>
#include <lbm_module/SimDomain.h>




int main(int argc, char* argv[])
{
	SimDomain sd({}, 1, 1, 1);
	LBMSolver solver(sd);
	solver.solve();

	//QApplication a(argc, argv);
	//MainWindow w;
	//w.show();
	//return a.exec();

	return EXIT_SUCCESS;
}