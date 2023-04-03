#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <SimDomain.h>
#include <ImmersedBody.h>
#include <ImmersedBoundaryMethod.h>
#include <LBM_Writer.h>
#include <LBM_Solver.h>
#include <LBM_Solver_P.h>
#include <ParticleGenerator.h>
#include <ParticleSource.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <Trajectory.h>
#include <functional>

using T = float;

//physic-data: https://stoffdaten-online.de/fluide/luft/
T dia = 0.06;
int lengthX_L_3D = 150;
const T lengthX = dia*49;
const T lengthY = dia*34;
const T lengthZ = dia*34;
const T rho = 1.189;
const T kinVis = 15.32e-6;
const T UMax = 5.f;
const T URef = 0.00766;

const T UMax_L = 0.2;


const int stepsPerFrame = 1;
const double simulationTime = 100.0;
//---------------------------------------------

//const T tau_L = 0.52;
const T tau_L = 0.8;
/*
#pragma region examples
const T lengthX_3D = 3;
const T lengthX_2D = 0.03;
const int lengthX_L_2D = 2000;
void calc_2D()
{
	const T lengthX = lengthX_2D;
	const int lengthX_L = lengthX_L_2D;

	SimDomain<T, 2> simDom{ { lengthX, lengthY }, kinVis, rho, UMax };

	if (!simDom.calcConvFactors(lengthX_L))
		std::cout << "Method 3: Unstable or inaccurate\n";
	else
		std::cout << "Method 3: dt=" << simDom.getTimeStep() << "; dh=" << simDom.getGridSize() << "; GridDimX_L=" << simDom.getGridDim_L(0) << " tau_l=" << simDom.getTau_L() << "\n\n";

	int gridDimX_L = simDom.getGridDim_L(0);
	int gridDimY_L = simDom.getGridDim_L(1);

	T RADIUS = gridDimY_L / 7.0;
	double alpha = static_cast<double>((T)(20/180.0) * PI<T>);
	vec<T, 4> R{ (T)cos(alpha),(T)-sin(alpha),(T)sin(alpha),(T)cos(alpha) };
	//std::unique_ptr<SPHERE2D<T>> body = std::make_unique<SPHERE2D<T>>((T)0.5, vec<T, 2>{(T)(2.0 * RADIUS), (T)(gridDimY_L / 2.0)}, RADIUS, 0.01);
	//std::unique_ptr<CUBOID2D<T>> body = std::make_unique<CUBOID2D<T>>((T)0.5, vec<T, 2>{(T)(gridDimX_L / 2.0), (T)(gridDimY_L / 2.0)}, (T)(gridDimY_L / 8), (T)(gridDimY_L / 8), 0 , R);
	//std::unique_ptr<CAPSULE2D<T>> body = std::make_unique<CAPSULE2D<T>>((T)0.5, vec<T, 2>{(T)(2.0 * RADIUS), (T)(gridDimY_L / 2.0)}, RADIUS, RADIUS);


	std::list<vec<int, 2>> indices{ {0,1},{1,2},{2,3},{3,0}};
	vec<T, 2> pos_0{ RADIUS/2,-RADIUS/2 };
	vec<T, 2> pos_1{ RADIUS/2,RADIUS/2  };
	vec<T, 2> pos_2{ -RADIUS/2, RADIUS/2  };
	vec<T, 2> pos_3{ -RADIUS/2,-RADIUS/2  };

	vec<T, 2> norm_0{ 1, 0};
	vec<T, 2> norm_1{ 1, 0};
	vec<T, 2> norm_2{ -1, 0};
	vec<T, 2> norm_3{ -1, 0};

	std::vector< vec<T, 2>> samples{ pos_0,pos_1,pos_2,pos_3 };
	std::vector< vec<T, 2>> normals{ norm_0,norm_1,norm_2,norm_3 };
	std::vector<char> tags{ 1,1,0,0 };
	std::unique_ptr<MESH<T, 2>> body = std::make_unique<MESH<T, 2>>( MESH<T,2> {0.5, vec<T, 2>{(T)(gridDimX_L / 2.0), (T)(gridDimY_L / 2.0)},indices,samples,normals,tags} );
	

	std::unique_ptr<IBMethod<T, 2>> ibm = std::make_unique<IBMethod<T, 2>>(IBMethod<T, 2>{IBM_STATIC});
	std::unique_ptr<LBM_Writer<T, 2>> writer = std::make_unique<LBM_Writer<T, 2>>(LBM_Writer<T, 2>{simDom, stepsPerFrame, "LBM_OUTPUT", L_UNIT_POS | L_UNIT_OUTPUT, true, true});

	body->setInletVelocity({ UMax_L,0.0 });
	//body->initializeKinematic({ 0.0,0.0 }, { 0.0,0.0,0.001 });
	//body->appendMoment(0, 10.0, vec<T, 3>{0.0, 0.0, 0.01});

	//ibm->addBody(std::move(box));
	ibm->addBody(std::move(body));
	//	ibm->addBody(std::make_unique<IM_BODY<T,2>>(cylinder));
	//	ibm.addBody(std::make_unique<SPHERE2D<T>>(cylinder));

	T uw_L = UMax_L;
	T dpw_L = 0.0 / simDom.getC_p();

	SimInitialiser<T, 2> simInit{};
	simInit.u = vec<T, 2>{ uw_L, 0.0};

	VELOCITY_BOUND<T, 2> leftInlet{ left, gridDimY_L / 2, gridDimY_L, uw_L };
	PRESSURE_BOUND<T, 2> rightOutlet{ right, gridDimY_L / 2, gridDimY_L , dpw_L };

	simDom.addVelocityBound(leftInlet);
	simDom.addPressureBound(rightOutlet);

	//ParticleGenerator
	ParticleSource<T, 2> particleSource{ vec<T,2>{(T)gridDimX_L, (T)gridDimY_L / 2}, vec<T,4>{init_R_2D}, vec<T,2>{(T)gridDimY_L / 10,  (T)gridDimY_L / 10}, (T)gridDimY_L / 5000, (T)gridDimY_L / 5300,(T)3,(T)997.0/simDom.getRho(),40000 };
	std::unique_ptr<ParticleGenerator <T, 2>> particleGenerator = std::make_unique<ParticleGenerator <T, 2>>(ParticleGenerator <T, 2>{});
	particleGenerator->appendSource(particleSource);
	particleGenerator->createParticles(simDom);


	//LBM_Solver<T, 2> solver{ simDom,std::move(writer), simulationTime, true, simInit};
	LBM_Solver<T, 2> solver{ simDom, std::move(ibm), std::move(writer),simulationTime, true, simInit };
	
	//solver.setParticleGenerator(std::move(particleGenerator));
	//solver.enableBGK();
	
	solver.solve();
	std::cout << solver.getClockOutput();
}

void calc_3D()
{
	const T lengthX = lengthX_3D;
	const int lengthX_L = lengthX_L_3D;

	SimDomain<T, 3> simDom{ { lengthX, lengthY, lengthZ }, kinVis, rho, UMax };

	if (!simDom.calcConvFactors(tau_L, lengthX_L))
		std::cout << "Method 3: Unstable or inaccurate\n";
	else
		std::cout << "Method 3: dt=" << simDom.getTimeStep() << "; dh=" << simDom.getGridSize() << "; GridDimX_L=" << simDom.getGridDim_L(0) << " tau_l=" << simDom.getTau_L() << "\n\n";

//if (!simDom->calcConvFactors(UMax_L, length_L))
//{
//	std::cout << "Method 1: Unstable or inaccurate" << std::endl;

//	if (!simDom->calcConvFactors(UMax_L, tau_L))
//		std::cout << "Method 2: Unstable or inaccurate" << std::endl;
//	else
//		std::cout << "Method 2: dt=" << simDom->getTimeStep() << "; dh=" << simDom->getGridSize() <<"; GridSizeX_L=" << simDom->getGridSizeX_L() << " tau_l=" << simDom->getTau_L() <<  std::endl;
//}
//else
//{
//	std::cout << "Method 1: dt=" << simDom->getTimeStep() << "; dh=" << simDom->getGridSize() << "; GridSizeX_L=" << simDom->getGridSizeX_L() << " tau_l=" << simDom->getTau_L() << std::endl;
//}

	double alpha = static_cast<double>((T)(20 / 180.0) * PI<T>);
	vec<T, 9> R{ (T)cos(alpha),(T)-sin(alpha),0,(T)sin(alpha),(T)cos(alpha),0,0,0,1 };

	int gridDimX_L = simDom.getGridDim_L(0);
	int gridDimY_L = simDom.getGridDim_L(1);
	int gridDimZ_L = simDom.getGridDim_L(2);


	T RADIUS = gridDimY_L / 5.0;
	T a =  RADIUS / 2;
	vec<T, 3> P0{ 1 * a,-1 * a, 1 * a};
	vec<T, 3> P1{ 1 * a, 1 * a, 1 * a};
	vec<T, 3> P2{ -1 * a, 1 * a, 1 * a};
	vec<T, 3> P3{ -1 * a,-1 * a, 1 * a};
	vec<T, 3> P4{ 1 * a,-1 * a, -1 * a };
	vec<T, 3> P5{ 1 * a, 1 * a, -1 * a };
	vec<T, 3> P6{ -1 * a, 1 * a, -1 * a };
	vec<T, 3> P7{ -1 * a,-1 * a, -1 * a };

	vec<T, 3> n0{ 1,  0, 0};
	vec<T, 3> n1{ 1,  0, 0};
	vec<T, 3> n2{-1,  0, 0};
	vec<T, 3> n3{ -1, 0, 0};
	vec<T, 3> n4{ 1,  0, 0};
	vec<T, 3> n5{ 1,  0, 0};
	vec<T, 3> n6{-1,  0, 0};
	vec<T, 3> n7{-1,  0, 0};
	std::list<vec<int, 3>> indices{ {0,5,1},{0,4,5},{3,1,2},{3,0,1},{2,5,6},{2,1,5},{2,7,3},{2,6,7},{6,4,7},{6,5,4},{3,7,4},{3,4,0} };
	std::vector<vec<T, 3>> samples{ P0, P1, P2, P3, P4, P5, P6, P7 };
	std::vector<vec<T, 3>> normals{ n0, n1, n2, n3, n4, n5, n6, n7 };
	std::vector<char> tags{ 1, 1, 0, 0, 1, 1, 0, 0 };

	//std::unique_ptr<SPHERE3D<T>> body = std::make_unique<SPHERE3D<T>>( (T)0.5, vec<T,3>{(T)(2.0 *RADIUS), (T)(gridDimY_L / 2.0), (T)(gridDimZ_L / 2.0)}, RADIUS, 0);
	//std::unique_ptr<CYLINDER<T>> body = std::make_unique<CYLINDER3D<T>>((T)0.5, vec<T, D>{(T)(2.0 * RADIUS), (T)(gridDimY_L / 2.0), (T)(gridDimZ_L / 2.0)}, RADIUS, RADIUS);
	//std::unique_ptr<CUBOID3D<T>> body = std::make_unique<CUBOID3D<T>>((T)0.5, vec<T, 3>{(T)(2.0 * RADIUS), (T)(gridDimY_L / 2.0), (T)(gridDimZ_L / 2.0)}, RADIUS, RADIUS, RADIUS,0,R);
	//std::unique_ptr<CAPSULE3D<T>> body = std::make_unique<CAPSULE3D<T>>((T)0.5, vec<T, D>{(T)(2.0 * RADIUS), (T)(gridDimY_L / 2.0), (T)(gridDimZ_L / 2.0)}, RADIUS, RADIUS);
	std::unique_ptr<MESH<T, 3>> body = std::make_unique<MESH<T, 3>>(MESH<T, 3>{(T)0.5, vec<T, 3>{(T)(gridDimX_L / 2.0), (T)(gridDimY_L / 2.0), (T)(gridDimZ_L / 2.0)},indices,samples,normals,tags});

	body->setInletVelocity({ UMax_L,0,0 });
	//body->initializeKinematic({ 0.0, 0.0 }, { 0.0,0.0,0.1 });
	//body->appendMoment(0, 10.0, vec<T, 3>{0.0, 0.0, 0.001});

	//body->transformFunc = transFunc;
	//body->dynamicFunc = dynFunc;

	std::unique_ptr<IBMethod<T, 3>> ibm = std::make_unique<IBMethod<T, 3>>(IBMethod<T, 3>{ IBM_STATIC});
	std::unique_ptr<LBM_Writer<T, 3>> writer = std::make_unique<LBM_Writer<T, 3>>(LBM_Writer<T, 3>{simDom, stepsPerFrame, "LBM_OUTPUT", L_UNIT_POS | L_UNIT_OUTPUT, true, true});
	//ibm.setSharpness(SHARP);


	//ibm->addBody(std::move(body));

	T uw_L = UMax_L;
	T dpw_L = 0.0 / simDom.getC_p();

	SimInitialiser<T, 3> simInit{};
	//simInit.u = vec<T, 3>{ uw_L, 0.0, 0.0};

	VELOCITY_BOUND<T, 3> leftInlet{ left, {gridDimY_L / 2, gridDimZ_L / 2}, gridDimZ_L, gridDimY_L, uw_L };
	PRESSURE_BOUND<T, 3> rightOutlet{ right, {gridDimY_L / 2, gridDimZ_L / 2}, gridDimZ_L, gridDimY_L, dpw_L };


	//simDom.addVelocityBound(leftInlet);
	//simDom.addPressureBound(rightOutlet);

	ParticleSource<T, 3> particleSource{ vec<T,3>{(T)gridDimX_L, (T)gridDimY_L / 2, (T)gridDimZ_L / 2}, vec<T,9>{init_R_3D}, vec<T,3>{(T)gridDimY_L / 10,  (T)gridDimY_L / 10, (T)gridDimY_L / 10}, (T)gridDimY_L / 5000, (T)gridDimY_L / 5300,  (T)3,(T)997.0/simDom.getRho(),1000 };
	std::unique_ptr<ParticleGenerator <T, 3>> particleGenerator = std::make_unique< ParticleGenerator <T, 3>>(ParticleGenerator <T, 3>{});
	particleGenerator->appendSource(particleSource);
	particleGenerator->createParticles(simDom);

	LBM_Solver<T, 3> solver{ simDom, std::move(ibm), std::move(writer),simulationTime, true, simInit };
	//LBM_Solver<T, 3> solver{ simDom,std::move(writer), simulationTime, true, simInit};

	solver.setParticleGenerator(std::move(particleGenerator));
	solver.solve();

	std::cout << solver.getClockOutput();
}

void calc_2D_P()
{
	const T lengthX = lengthX_2D;
	const int lengthX_L = lengthX_L_2D;


	SimDomain<T, 2> simDom{ { lengthX, lengthY }, kinVis, rho, UMax };

	if (!simDom.calcConvFactors(tau_L, lengthX_L))
		std::cout << "Method 3: Unstable or inaccurate\n";
	else
		std::cout << "Method 3: dt=" << simDom.getTimeStep() << "; dh=" << simDom.getGridSize() << "; GridDimX_L=" << simDom.getGridDim_L(0) << " tau_l=" << simDom.getTau_L() << "\n\n";

	std::unique_ptr<IBMethod_P<T, 2>> ibm = std::make_unique<IBMethod_P<T, 2>>(IBMethod_P<T, 2>{ IBM_STATIC});
	std::unique_ptr<LBM_Writer<T, 2>> writer = std::make_unique<LBM_Writer<T, 2>>(LBM_Writer<T, 2>{simDom, stepsPerFrame, "LBM_OUTPUT", L_UNIT_POS | L_UNIT_OUTPUT, true, true});


	int gridDimX_L = simDom.getGridDim_L(0);
	int gridDimY_L = simDom.getGridDim_L(1);

	T RADIUS =  gridDimY_L / 7.0;
	double alpha = static_cast<double>((T)(20 / 180.0) * PI<T>);
	vec<T, 4> R{ (T)cos(alpha),(T)-sin(alpha),(T)sin(alpha),(T)cos(alpha) };
	
	std::list<vec<int, 2>> indices{ {0,1},{1,2},{2,3},{3,0} };
	vec<T, 2> pos_0{ RADIUS / 2,-RADIUS / 2 };
	vec<T, 2> pos_1{ RADIUS / 2,RADIUS / 2 };
	vec<T, 2> pos_2{ -RADIUS / 2, RADIUS / 2 };
	vec<T, 2> pos_3{ -RADIUS / 2,-RADIUS / 2 };

	vec<T, 2> norm_0{ 1, 0 };
	vec<T, 2> norm_1{ 1, 0 };
	vec<T, 2> norm_2{ -1, 0 };
	vec<T, 2> norm_3{ -1, 0 };

	std::vector< vec<T, 2>> samples{ pos_0,pos_1,pos_2,pos_3 };
	std::vector< vec<T, 2>> normals{ norm_0,norm_1,norm_2,norm_3 };
	std::vector<char> tags{ 1,1,0,0 };
	std::unique_ptr<MESH<T, 2>> body = std::make_unique<MESH<T, 2>>(MESH<T, 2> {0.5, vec<T, 2>{(T)(gridDimX_L / 2.0), (T)(gridDimY_L / 2.0)}, indices, samples, normals, tags});


	//std::unique_ptr<SPHERE2D<T>> body = std::make_unique<SPHERE2D<T>>(SPHERE2D<T>{(T)0.5, vec<T, 2>{(T)(3.0 * RADIUS), (T)(gridDimY_L / 2.0)}, RADIUS, 0.01});
	//std::unique_ptr<MESH<T,2>> body = std::make_unique<MESH<T,2>>(MESH<T,2>{(T)0.5, indices, samples, 0.01});
	//std::unique_ptr<CUBOID2D<T>> body = std::make_unique<CUBOID2D<T>>((T)0.5, vec<T, 2>{(T)(2.0 * RADIUS), (T)(gridDimY_L / 2.0)}, RADIUS, RADIUS,0.01);

	//body->initializeKinematic({ 0.0, 0.0 } , { 0.0,0.0,0.001 });
	//body->appendMoment(0, 10.0, vec<T, 3>{0.0, 0.0,0.001});
	//body->appendForce(0, 10.0, vec<T, 2>{0.00001, 0.0});

	body->setInletVelocity({ UMax_L/2,0 });

	ibm->addBody(std::move(body));
	T uw_L = UMax_L;
	T dpw_L = 0.0 / simDom.getC_p();

	SimInitialiser_P<T, 2> simInit{};
	simInit.u = vec<T, 2>{ -uw_L/2, 0.0 };

	VELOCITY_BOUND<T, 2> leftInlet{ right, gridDimY_L / 2, gridDimY_L, uw_L/2 };
	PRESSURE_BOUND<T, 2> rightOutlet{ left, gridDimY_L / 2, gridDimY_L , dpw_L };

	simDom.addVelocityBound(leftInlet);
	simDom.addPressureBound(rightOutlet);

	ParticleSource<T, 2> particleSource{ vec<T,2>{(T)gridDimX_L, (T)gridDimY_L / 2}, vec<T,4>{init_R_2D}, vec<T,2>{(T)gridDimY_L / 10,  (T)gridDimY_L / 10}, (T)gridDimY_L / 5000, (T)gridDimY_L / 5300,(T)3,(T)997.0 / simDom.getRho(),40000 };
	std::unique_ptr<ParticleGenerator_P<T, 2>> particleGenerator = std::make_unique<ParticleGenerator_P<T, 2>>(ParticleGenerator_P<T, 2>{});
	particleGenerator->appendSource(particleSource);
	particleGenerator->createParticles(simDom);

	LBM_Solver_P<T, 2> solver{ simDom,std::move(ibm),std::move(writer) ,simulationTime, true, simInit };
	//LBM_Solver_P<T, 2> solver{ simDom,std::move(writer) ,simulationTime, true, simInit };
	solver.setParticleGenerator(std::move(particleGenerator),false);

	solver.solve();
	std::cout << solver.getClockOutput();

}
#pragma endregion
*/

T tStart = 0.1;
T tEnd = 0.4;
T tVar = (tEnd - tStart) / (T)2.0;

T scaler(T t)
{
	

	T r = 0;
	if (t > tStart && t < tEnd)
		r = 1.0 - std::abs((tVar + tStart - t) / tVar);
	else 
		r = std::sin(PI<T>*t/0.2)*(1.0/(UMax/0.139));	//Amplitude=0.139

	return r;
}

T rate(T t, T timeStep)
{
	T r = (T)5.0;  //particle/s
	if (t > tStart && t < tEnd)
		r = (T)100000.0;
	return r * timeStep;  //particle within a timeStep

}

void calc_example()
{
	const int lengthX_L = lengthX_L_3D;
	SimDomain<T, 3> simDom{ { lengthX, lengthY, lengthZ }, kinVis, rho, URef };

	simDom.calcConvFactors(lengthX_L);
	//if (!simDom.calcConvFactors(tau_L,lengthX_L))
	//	std::cout << "Method 3: Unstable or inaccurate\n";
	//else
	//	std::cout << "Method 3: dt=" << simDom.getTimeStep() << "; dh=" << simDom.getGridSize() << "; GridDimX_L=" << simDom.getGridDim_L(0) << " tau_l=" << (T)1.0/simDom.getRelaxationConstant() << "\n\n";

	simDom.setZerothRelaxationTime(0); //only 0
	simDom.setLowRelaxationTimes(1); //or 1



	std::unique_ptr<IBMethod_P<T, 3>> ibm = std::make_unique<IBMethod_P<T, 3>>(IBMethod_P<T, 3>{ IBM_STATIC});
	std::unique_ptr<LBM_Writer<T, 3>> writer = std::make_unique<LBM_Writer<T, 3>>(LBM_Writer<T, 3>{simDom, stepsPerFrame, "LBM_OUTPUT", true,false});
	ibm->setSharpness(SMOOTH);

	double alpha = static_cast<double>( PI<T>/2);
	vec<T, 9> R{ 1,0,0,0,(T)cos(alpha),(T)-sin(alpha),0,(T)sin(alpha),(T)cos(alpha) };

	T RADIUS = 0.6;
	
	auto position = vec<T, 3>{ (T)2.0, (T)1.5, (T)1.5 };
	
	T a = RADIUS / 2;
	vec<T, 3> P0{ 1 * a,-1 * a, 1 * a };
	vec<T, 3> P1{ 1 * a, 1 * a, 1 * a };
	vec<T, 3> P2{ -1 * a, 1 * a, 1 * a };
	vec<T, 3> P3{ -1 * a,-1 * a, 1 * a };
	vec<T, 3> P4{ 1 * a,-1 * a, -1 * a };
	vec<T, 3> P5{ 1 * a, 1 * a, -1 * a };
	vec<T, 3> P6{ -1 * a, 1 * a, -1 * a };
	vec<T, 3> P7{ -1 * a,-1 * a, -1 * a };

	vec<T, 3> n0{ 1,  0, 0 };
	vec<T, 3> n1{ 1,  0, 0 };
	vec<T, 3> n2{ -1,  0, 0 };
	vec<T, 3> n3{ -1, 0, 0 };
	vec<T, 3> n4{ 1,  0, 0 };
	vec<T, 3> n5{ 1,  0, 0 };
	vec<T, 3> n6{ -1,  0, 0 };
	vec<T, 3> n7{ -1,  0, 0 };
	std::list<vec<int, 3>> indices{ {0,5,1},{0,4,5},{3,1,2},{3,0,1},{2,5,6},{2,1,5},{2,7,3},{2,6,7},{6,4,7},{6,5,4},{3,7,4},{3,4,0} };
	std::vector<vec<T, 3>> samples{ P0, P1, P2, P3, P4, P5, P6, P7 };
	std::vector<vec<T, 3>> normals{ n0, n1, n2, n3, n4, n5, n6, n7 };
	std::vector<char> tags{ 0,0,1, 1, 0, 0, 1, 1 };

	ParticleSource<T, 3> particleSource{ position, vec<T,9>{init_R_3D}, vec<T,3>{ RADIUS,   RADIUS ,  RADIUS}, (T)3.0 / 5000, (T)3.0 / 5300,  (T)0.0085714 ,(T)997.0, 60000,{0.7,0.0,0.0 }, rate, 3 };

	//std::unique_ptr<MESH<T, 3>> head = std::make_unique<MESH<T, 3>>(MESH<T, 3>{(T)0.2, position, indices, samples, normals, tags});
	//std::unique_ptr<CUBOID3D<T>> head = std::make_unique<CUBOID3D<T>>(CUBOID3D<T>{0.6,position, RADIUS, RADIUS, RADIUS, vec<T, 9>{init_R_3D}, true});
	//PositionRead<T> posRead = readFile<T>("exp.csv");
	//head->setPositionReader(posRead,1.5);

	/*head->inletVelocityScaler = scaler;
	head->inlet_velocity = { UMax,0,0 };
	head->sourceID = particleSource.getID();*/


	position = vec<T, 3>{ 2*9* dia, (T)lengthY/2, (T)lengthZ/2 };

	//T A = dia * lengthZ;
	//std::unique_ptr <CYLINDER3D<T>> head = std::make_unique<CYLINDER3D<T>>(CYLINDER3D<T>{0.6, position, dia / 2, lengthZ* (T)1.1,R});

	T A = dia * dia * PI<T> /4;
	std::unique_ptr <SPHERE3D<T>> head = std::make_unique<SPHERE3D<T>>(SPHERE3D<T>{0.6, position, dia / 2});
	//std::unique_ptr<ParticleGenerator_P<T, 3>> particleGenerator = std::make_unique< ParticleGenerator_P<T, 3>>(ParticleGenerator_P<T, 3>{SMOOTH});
	//particleGenerator->appendSource(particleSource);
	//particleGenerator->createParticles(simDom, true);
	//particleGenerator->enableGravity();
	
	ibm->addBody(std::move(head));
	
	
	
	T dp_w = 0; //[Pa]

	SimInitialiser_P<T, 3> simInit{ rho, vec<T, 3>{URef, 0.0, 0.0 }, vec<T, 3>{}};
	

	VELOCITY_BOUND<T, 3> leftInlet{ left, URef };
	PRESSURE_BOUND<T, 3> rightOutlet{ right, dp_w };
	PRESSURE_BOUND<T, 3> leftOutlet{ left, dp_w };
	PRESSURE_BOUND<T, 3> topOutlet{ top, dp_w };
	PRESSURE_BOUND<T, 3> bottomOutlet{ bottom, dp_w };
	PRESSURE_BOUND<T, 3> frontOutlet{ front, dp_w };
	PRESSURE_BOUND<T, 3> backOutlet{ back, dp_w };

	simDom.addVelocityBound(leftInlet);
	simDom.addPressureBound(rightOutlet);
	//simDom.addPressureBound(leftOutlet);
	//simDom.addPressureBound(topOutlet);
	//simDom.addPressureBound(bottomOutlet);
	/*simDom.addPressureBound(frontOutlet);
	simDom.addPressureBound(backOutlet);*/


	LBM_Solver_P<T, 3> solver{ simDom, std::move(ibm), std::move(writer), simulationTime,simInit, false};
	//LBM_Solver_P<T, 3> solver{ simDom,std::move(writer), simulationTime, true, simInit };
	//solver.setParticleGenerator(std::move(particleGenerator));
	//solver.disableAdaptiveTimeStep();

	solver.solve(A);
	std::cout << solver.getClockOutput();
}

int main(int argc, char* argv[])
{
	if(argc > 1)
		lengthX_L_3D = std::stoi(argv[1]);
	std::cout << lengthX_L_3D << std::endl;
	calc_example();
	return 0;
}