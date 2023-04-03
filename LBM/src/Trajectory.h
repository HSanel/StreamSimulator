/*File to decline Trajektorian eqaution

Trajektorien sollen abgebildet werden und ausgewertet werden kˆnnen
- Transformations- oder Rotationsmatrix mit Translationsvktor zu geg.Zeitpunkt t und weiteren Inpulsen

-erstmal Trajektorien mit nur Translationsanteil

Ziel: Klasse soll f¸r Bewegung genutzt werden (solver muss bescheid gegeben werden)

Fragen:
-Trajktorie soll f¸r jeden Agenten berechnet werden?
- Wir befinden uns im 2D Raum? -> x und z interessant, da nur vorw‰rtsbegen und nicht hoch, oder? heiﬂt
y ist irrelevant
*/

#pragma once
#include <vector>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

//#include <Eigen/Dense>
//#include <Eigen/Geometry>
//using namespace Eigen;

#include <fstream>
#include <istream>
#include <cmath>
#include <cfloat>
#include <math.h>
#include "LBM_Types.h"


template<typename Type>
struct PositionRead
{
	std::vector<vec<Type, 2>> positions;
	std::vector<Type> angles;
	std::vector<Type> timestep;
};

template<typename Type>
PositionRead<Type> readFile(std::string filename);


template<typename Type>
class Trajectory
{
private:
	std::vector<vec<Type,2>> positions;
	std::vector<Type> angles;
	std::vector<Type> timestep;
	int index;
	vec<Type,4> position;
	int step;
	bool clockwiseTurn; // if true --> turn in clock direction
	Type startAngle;
	Type endAngle;
	Type diffenceAngle;

	vec<Type,16> Trans; 	// Transformationsmatrix
	vec<Type, 16> R;		// Rotationsmatrix
	vec<Type,4> T;	// Translationsvector


public:
	Trajectory();


	// Give the variables the values
	void setValues(PositionRead<Type>& positionInfo);

	// get transformtionmtrix at specific timestep
	void getTransformationMatrix(bool interpolate, Type time);

	// get specific position at a timestep
	vec<Type,3> getPosition(Type time);

	//std::vector<Position> Get index and true if in timesteps included
	bool getIndex(Type time);

	// declare which turn direction
	void getTurn();

	// Calculate rotation depending on clockwise turn
	Type calculateRotation(Type alpha);

	int getIdx() { return index; };

	vec<Type, 9> getRotation();

	void reset();

};
