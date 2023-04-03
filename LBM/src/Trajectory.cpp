#include "Trajectory.h"

template class Trajectory<float>;
template class Trajectory<double>;
template class PositionRead<float>;
template class PositionRead<double>;
template PositionRead<float> readFile(std::string filename);
template PositionRead<double> readFile(std::string filename);

template<typename Type>
PositionRead<Type> readFile(std::string filename)
{
	std::fstream file;
	file.open(filename);

	if (!file.is_open()) {
		std::cout << "Not able to open File" << std::endl;
	}
	std::string line, word;
	PositionRead<Type> info;
	std::vector<std::string> row;

	std::cout << filename << std::endl;

	// Go through each line
	while (getline(file, line))
	{
		row.clear();
		std::stringstream s(line);
		while (std::getline(s, word, ',')) {
			row.push_back(word);
		}

		info.timestep.push_back(std::stof(row[1]));
		vec<Type,2> pos{ (Type)std::stof(row[2]), (Type)std::stof(row[3]) };
		info.positions.push_back(pos);
		info.angles.push_back((Type)std::stof(row[4]));
	}

	return info;
}

template<typename Type>
Trajectory<Type>::Trajectory()
{
	index = 0;
}

template<typename Type>
void Trajectory<Type>::setValues(PositionRead<Type> &positionInfo) {
	// positions = positionInfo.positions;
	for (int i = 0; i < positionInfo.positions.size(); i++) {
		//Eigen::Vector2f test = Vector2f(positionInfo.positions[i][0], positionInfo.positions[i][1]);
		positions.push_back(positionInfo.positions[i]);
	}
	angles = positionInfo.angles;
	timestep = positionInfo.timestep;
	// TODO Adapt so it takes from original first position
	position = { positions[index][0], 0.0, positions[index][1], 1 };
	Type angle = angles.at(index);
	Type cos = std::cos(angle * PI<Type> / 180.0);
	Type sin = std::sin(angle * PI<Type> / 180.0);
	R = { cos, 0, sin, 0,
		0, 1, 0, 0,
		-sin, 0, cos, 0,
		0, 0, 0, 1 };
	step = 0;
	getTurn();
}

// Test if angle same
// if angle > 360
// if -50
// if 180 --> 90
template<typename Type>
void Trajectory<Type>::getTurn() {
	startAngle = angles[index];
	if (startAngle > 360 || startAngle < - 360) {
		int test = std::round(startAngle);
		Type a = test % 360;
		startAngle = a;
	}

	endAngle = angles[index + 1];
	if (endAngle > 360 || endAngle < - 360) {
		int test = std::round(endAngle);
		Type b = test % 360;
		endAngle = b;
	}

	if (startAngle < endAngle) {
		clockwiseTurn = false;
	}
	else
	{
		clockwiseTurn = true;
	}
	diffenceAngle = std::abs(endAngle - startAngle);
}

template<typename Type>
Type Trajectory<Type>::calculateRotation(Type alpha)
{
	Type angle;
	if (clockwiseTurn) {
		angle = startAngle - alpha * diffenceAngle;
	}
	else
	{
		angle = startAngle + alpha * diffenceAngle;
	}
	return angle;
}

template<typename Type>
void Trajectory<Type>::getTransformationMatrix(bool interpolate, Type time)
{
	Trans = vec<Type,16>{ 1,0,0,0,
						  0,1,0,0,
						  0,0,1,0,
						  0,0,0,1 };

	Type x, y, z, angle;

	Type alpha;
	alpha = (time - timestep[index]) / (timestep[index + 1] - timestep[index]);
		
	// Get new position
	x = alpha * (positions[index + 1][0] - positions[index][0]);
	y = 0.0;
	z = alpha * (positions[index + 1][1] - positions[index][1]);
	// TODO with Rotation and Translation correctly
	Trans[3] = x;
	Trans[7] = y;
	Trans[11] = z;
		
	// Get new rotation
	// angle = alpha * (angles.at(index + 1) - angles.at(index));
	angle = calculateRotation(alpha);
	Type cos = std::cos(angle * PI<Type> / 180.0);
	Type sin = std::sin(angle * PI<Type> / 180.0);
	R = { cos, 0, sin, 0,
		0, 1, 0, 0,
		-sin, 0, cos, 0,
		0, 0, 0, 1 };

	// TODO: Compose correct transfomation matrix
	//T = matMult<T,4>(R, Trans);

}

template<typename Type>
bool Trajectory<Type>::getIndex(Type time) {

	bool interpolate;
	int test = timestep.size();

	for (int idx = index; idx <= timestep.size(); idx++) {
		if (std::abs(timestep[idx] - time) < 0.0001 || idx == test) {
			interpolate = false;
			index = idx;
			break;
		}
		if (time < timestep[idx]) {
			interpolate = true;
			index = idx - 1;
			if (index == 0) { step++; }
			break;
		}
	}
	//std::cout << "Index \t" << index << std::endl;
	return interpolate;
}

template<typename Type>
vec<Type, 9> Trajectory<Type>::getRotation()
{
	vec<Type, 9> R_out{};
	R_out[0] = R[0];
	R_out[1] = R[1];
	R_out[2] = R[2];
	R_out[3] = R[4];
	R_out[4] = R[5];
	R_out[5] = R[6];
	R_out[6] = R[8];
	R_out[7] = R[9];
	R_out[8] = R[10];


	return R_out;

}

template<typename Type>
vec<Type,3> Trajectory<Type>::getPosition(Type time) {

	vec<Type,3> cart_pos;

	if (time == 0.0)
	{
		cart_pos = vec<Type, 3>{ position[0] / position[3], position[1] / position[3], position[2] / position[3] };

		return cart_pos;
	}

	// test if timestep in vector -> if yes: return trans
	bool interpolate = Trajectory::getIndex(time);

	if (!interpolate)
	{
		std::cout << "Index: " << index << " Size: " << timestep.size() << std::endl;
		if (index + 1 != timestep.size()) {
			getTurn();
		}
		position = vec<Type, 4>{ positions[index][0], 0.0, positions[index][1], 1 };
		cart_pos = vec<Type, 3>{ position[0] / position[3], position[1] / position[3], position[2] / position[3] };
		Type angle = angles[index];
		Type cos = std::cos(angle * PI<Type> / 180.0);
		Type sin = std::sin(angle * PI<Type> / 180.0);
		R = vec<Type, 16>{ cos, 0, sin, 0,
			0, 1, 0, 0,
			-sin, 0, cos, 0,
			0, 0, 0, 1 };
		return cart_pos;
	}

	// Set up matrix
	Trajectory<Type>::getTransformationMatrix(interpolate, time);

	vec<Type,4> newPosition = Trans * position;

	cart_pos = vec<Type, 3>{ newPosition[0] / newPosition[3], newPosition[1] / newPosition[3], newPosition[2] / newPosition[3] };

	return cart_pos;

}

template<typename Type>
void Trajectory<Type>::reset() {
	index = 0;
	position = { positions[index][0], 0.0, positions[index][1], 1 };
	Type angle = angles[index];
	Type cos = std::cos(angle * PI<Type> / 180.0);
	Type sin = std::sin(angle * PI<Type> / 180.0);
	R = { cos, 0, sin, 0,
		0, 1, 0, 0,
		-sin, 0, cos, 0,
		0, 0, 0, 1 };
	step = 0;
}