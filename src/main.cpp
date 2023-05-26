#include <QApplication>
#include <ui_module/MainWindow.h>
#include <lbm_module/LBMSolver.h>
#include <lbm_module/SimDomain.h>
#include <lbm_module/SimState.h>
#include <iostream>


int main(int argc, char* argv[])
{
	SimDomain sd({10,10,10}, 1, 1, 1);
	LBMSolver solver(sd);
	SimRenderDimension sim_dim{ sd.getGridDim_L(), 1 };

	QApplication a(argc, argv);
	MainWindow w(sim_dim);
	w.show();

	solver.solve();
	auto st = solver.getSimState();
	w.updateData(st->rho_L.data(), st->u_L.data());

	return a.exec();

	return EXIT_SUCCESS;
}