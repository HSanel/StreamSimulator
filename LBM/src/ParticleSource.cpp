#include "ParticleSource.h"
template class ParticleSource_Specialisation<float, 2>;
template class ParticleSource_Specialisation<double, 2>;
template class ParticleSource_Specialisation<float, 3>;
template class ParticleSource_Specialisation<double, 3>;

template class ParticleSource<float, 2>;
template class ParticleSource<double, 2>;
template class ParticleSource<float, 3>;
template class ParticleSource<double, 3>;

#pragma region part_source_spec_2D
template<typename T>
ParticleSource_Specialisation<T, 2>::ParticleSource_Specialisation(vec<T, 2> position, vec<T, 4> rotMat, vec<T, 2> sourceDim, T min_dia, T max_dia, T time, T rho_p, int particleCountPerEvent, vec<T, 2> shift)
	: position(position), R(rotMat), uDim(sourceDim[0]), vDim(sourceDim[1]), min_dia(min_dia), max_dia(max_dia), time(time), rho_p(rho_p), count(particleCountPerEvent), shift(shift) {}

template<typename T>
void ParticleSource_Specialisation<T, 2>::create(std::vector<ParticleData<T, 2>>& particleData, const T& dh, const T& dt, const T& rho, vec<grid_size_t, 3>& gridDim_L, int& activeParticleCount, bool simultan)
{
	T C_l = 1.0, C_rho = 1.0, C_t = 1.0;

	if (!normilized_parameter)
	{
		C_l = dh;
		C_rho = rho;
	}
	else
	{
		C_t = dt;
	}

	position /= C_l;
	shift /= C_l;

	for (int i = 0; i < count; ++i)
	{
		T a = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * uDim - uDim / 2;
		T b = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * vDim - vDim / 2;
		T dia = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * (max_dia - min_dia) + min_dia) / C_l;
		vec<T, 2> pos = position + R * (vec<T, 2>{a, b} / C_l + shift);

		if (pos[0] >= 0 && pos[0] <= gridDim_L[0] - 1
			&& pos[1] >= 0 && pos[1] <= gridDim_L[1] - 1)
		{
			particleData.push_back(ParticleData<T, 2>{pos, vec<T, 2>{}, dia, rho_p / C_rho, this->time* C_t, true});
			activeParticleCount++;
		}
	}
}

template<typename T>
T ParticleSource_Specialisation<T, 2>::getDim(int axis) const
{
	if (axis == 0)
		return uDim;
	else if (axis == 1)
		return vDim;
	else
		throw std::runtime_error("ERROR:: ParticleSource:: axis does not exist");
}
#pragma endregion
//3D
//-----------------
template<typename T>
ParticleSource_Specialisation<T, 3>::ParticleSource_Specialisation(vec<T, 3> position, vec<T, 9> rotMat, vec<T, 3> sourceDim, T min_dia, T max_dia, T time, T rho_p, int particleCountPerEvent, vec<T, 3> shift)
	:position(position), R(rotMat), uDim(sourceDim[0]), vDim(sourceDim[1]), wDim(sourceDim[2]), min_dia(min_dia), max_dia(max_dia), time(time), rho_p(rho_p), count(particleCountPerEvent), shift(shift){}

template<typename T, size_t D>
ParticleSource<T, D>::ParticleSource(vec<T, D> position, vec<T, D*D> rotMat, vec<T, D> sourceDim, T min_dia, T max_dia, T time, T rho_p, int particleCountPerEvent, vec<T, D> shift, T(*rateFunction)(T, T), int timeStepCountPerEvent)
	: ParticleSource_Specialisation(position, rotMat, sourceDim, min_dia, max_dia, time, rho_p, particleCountPerEvent, shift),rate(rateFunction), timeStepCountPerEvent(timeStepCountPerEvent)
{
	this->ID = global_ID;
	global_ID++;
}


template<typename T, size_t D>
int ParticleSource<T, D>::global_ID = 0;

template<typename T, size_t D>
int ParticleSource<T, D>::getID() const
{
	return this->ID;
}

template<typename T, size_t D>
int ParticleSource<T, D>::getMaxTimeStepPerEvent() const
{
	return timeStepCountPerEvent;
}

template<typename T>
void ParticleSource_Specialisation<T, 3>::create(std::vector<ParticleData<T, 3>>& particleData, const T& dh, const T& dt, const T& rho, vec<grid_size_t, 3>& gridDim_L, int& activeParticleCount, bool simultan)
{
	T C_l = 1.0, C_rho = 1.0, C_t = 1.0;

	if (!normilized_parameter)
	{
		C_l = dh;
		C_rho = rho;
	}
	else
	{
		C_t = dt;
	}

	position /= C_l;
	shift /= C_l;

	for (int i = 0; i < count; ++i)
	{
		T a = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * uDim - uDim / 2;
		T b = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * vDim - vDim / 2;
		T c = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * wDim - wDim / 2;
		T dia = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * (max_dia - min_dia) + min_dia) / C_l;
		

		if (simultan)
		{
			vec<T, 3> pos = position + R * (vec<T, 3>{0, b, c}/C_l + shift);

			if (pos[0] >= 0 && pos[0] <= gridDim_L[0] - 1
				&& pos[1] >= 0 && pos[1] <= gridDim_L[1] - 1
				&& pos[2] >= 0 && pos[2] <= gridDim_L[2] - 1)
			{
				particleData.push_back(ParticleData<T, 3>{pos, vec<T, 3>{}, dia, rho_p / C_rho, 0, false});
			}
		}
		else
		{
			vec<T, 3> pos = position + R * (vec<T, 3>{a, b, c}/C_l + shift);

			if (pos[0] >= 0 && pos[0] <= gridDim_L[0] - 1
				&& pos[1] >= 0 && pos[1] <= gridDim_L[1] - 1
				&& pos[2] >= 0 && pos[2] <= gridDim_L[2] - 1)
			{
				particleData.push_back(ParticleData<T, 3>{pos, vec<T, 3>{}, dia, rho_p / C_rho, this->time* C_t, true});
				activeParticleCount++;
			}
		}
	}
}


template<typename T>
T ParticleSource_Specialisation<T, 3>::getDim(int axis) const
{
	if (axis == 0)
		return uDim;
	else if (axis == 1)
		return vDim;
	else if (axis == 2)
		return wDim;
	else
		throw std::runtime_error("ERROR:: ParticleSource:: axis does not exist");
}

template<typename T, size_t D>
T ParticleSource<T, D>::getTime() const
{
	return time;
}

template<typename T, size_t D>
vec<T, D> ParticleSource<T, D>::getPosition() const
{
	return position;
}

template<typename T, size_t D>
vec<T, D*D> ParticleSource<T, D>::getRotMat() const
{
	return R;
}

template<typename T, size_t D>
int ParticleSource<T, D>::getCount() const
{
	return this->count;
}

template<typename T, size_t D>
void ParticleSource<T, D>::setPosition(vec<T, D> position)
{
	this->position = position;
}

template<typename T, size_t D>
void ParticleSource<T, D>::setRotation(vec<T, D*D> R)
{
	this->R = R;
}

