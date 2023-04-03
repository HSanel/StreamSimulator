#ifndef PARTICLE_SOURCE
#define PARTICLE_SOURCE
#include "LBM_Types.h"
#include <functional>

template<typename T, size_t D>
struct ParticleData
{
	vec<T, D> position;
	vec<T, D> velocity;
	T diameter;
	T rho_p;
	T time;
	bool is_active;
};


template<typename T, size_t D> class ParticleSource_Specialisation;

#pragma region part_source_spec_2D
template<typename T>
class ParticleSource_Specialisation<T, 2>
{
protected:
	int count;
	vec<T, 2> position;
	vec<T, 2> shift;
	vec<T, 4> R;
	T min_dia, max_dia;
	T uDim,vDim;
	T time;
	T rho_p;
	int activateParticleInSource;
public:
	ParticleSource_Specialisation(vec<T, 2> position, vec<T, 4> rotMat, vec<T,2> sourceDim, T min_dia, T max_dia, T time, T rho_p, int particleCountPerEvent, vec<T, 2> shift);
	void create(std::vector<ParticleData<T, 2>>& particleData, const T& dh, const T& dt, const T& rho, vec<grid_size_t, 3>& gridDim_L, int& activeParticleCount, bool simultan);
	T getDim(int axis) const;
};
#pragma endregion

template<typename T>
class ParticleSource_Specialisation<T, 3>
{
protected:
	int count;
	vec<T, 3> position;
	vec<T, 3> shift;
	vec<T, 9> R;
	T min_dia, max_dia;
	T uDim,vDim,wDim;
	T time;
	T rho_p;
	int activateParticleInSource;
public:
	ParticleSource_Specialisation(vec<T, 3> position, vec<T, 9> rotMat, vec<T, 3> sourceDim, T min_dia, T max_dia, T time, T rho_p, int particleCountPerEvent, vec<T, 3> shift);
	void create(std::vector<ParticleData<T, 3>>& particleData, const T& dh, const T& dt, const T& rho, vec<grid_size_t, 3>& gridDim_L, int& activeParticleCount, bool simultan);
	T getDim(int axis) const;
};

template<typename T, size_t D>
class ParticleSource :public ParticleSource_Specialisation<T, D>
{
	static int global_ID;
	int ID;
	int timeStepCountPerEvent;
public:
	ParticleSource(vec<T, D> position, vec<T, D* D> rotMat, vec<T, D> sourceDim, T min_dia, T max_dia, T time, T rho_p, int particleCountPerEvent, vec<T, D> shift = {}, T(*rateFunction)(T, T) = nullptr, int timeStepCountPerEvent=1);
	T getTime() const;
	int getCount() const;
	vec<T, D> getPosition() const;
	vec<T, D* D> getRotMat() const;
	void setPosition(vec<T, D> position);
	void setRotation(vec<T, D* D> R);
	int memLocation = 0;
	int notActiveMemLocation = 0;
	int getID() const;
	int getMaxTimeStepPerEvent() const;
	std::function<T(T currentTime, T timeStep)> rate;
	int currentTempTimeStep = 0;
	T integratedParticleCount = (T)0.0;
};

#endif // !PARTICLE_SOURCE
