#ifndef IMB
#define IMB
#include "LBM_Types.h"

#include <list>
#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <functional>
#include "cyCodeBase/cyPoint.h"
#include "cyCodeBase/cySampleElim.h"
#include "cudaErrorHandle.h"
#include "Trajectory.h"

template<typename T, size_t D>
struct InletGhostCells
{
	int pos_bound;
	int pos_fluid;
	T q;
	vec<T, D> velocity;
	bool isInlet = false;
};

template<typename T>
T defaultScaler(T currentTime) { return (T)1.0; }

struct IDataGenerator
{
//protected:
public:
	IDataGenerator() = default;

	virtual void generateSamples(double dh, double con_u, double rho);
	virtual void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize);

	void executeGenerator(IDataGenerator& generator, double dh, double C_u, double rho);
	void executeGenerator(IDataGenerator& generator, double dh, double C_u, double rho, vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize);
};

template<typename T, size_t D> struct IM_BODY;
template<typename T, size_t D> struct MESH;


template<typename T, size_t D>
struct IM_BODY_P 
{
	vec_field<location_gpu, T, D> samples = nullptr;
	vec_field<location_gpu, T, D> velocities = nullptr;
	vec_field<location_gpu, T, D> normals = nullptr;

	vec_field<location_gpu, T, D> inletSamples = nullptr;
	vec_field<location_gpu, T, D> inletVelocities = nullptr;
	vec_field<location_gpu, T, D> inletNormals = nullptr;
	//test
	scal_field<location_gpu, char> is_inlet = nullptr;
	vec<T, D> initialPosition;

	IM_BODY_P(IM_BODY<T, D>& body);
	void copySamplesToHost(IM_BODY<T, D>& body);
};

template<typename T, size_t D>
struct IM_BODY : public IDataGenerator
{
public:
	vec<T, D> position;
	vec<T, D*D> R;

	std::vector<vec<T, D>> normals;
	std::vector<vec<T, D>> samples;
	std::vector<vec<T, D>> velocities;
	std::vector<vec<T, D>> inletNormals;
	std::vector<vec<T, D>> inletSamples;
	std::vector<vec<T, D>> inletVelocities;

	std::vector<char> is_inlet;
	int sourceID = -1;

	vec<T, D> velocity_center{};
	vec<T, D> inlet_velocity{};

	std::vector<T> parameter;
	T sampleDistance = 0.5;

	//Function-Format:: SCALE inletVelocitySampler(CURRENT_TIME_IN_s)
	std::function < T(T currentTime) > inletVelocityScaler = defaultScaler<T>;

	
	IM_BODY(T sampleDistance, vec<T, D> position, const std::vector<vec<T, D>> &samples,const std::vector<vec<T, D>> &normals,const std::vector<char> &inletTags, vec<T, D* D> initialRot);
};

template<typename T>
struct MESH<T, 3> : public IM_BODY<T, 3>
{
protected:
	Trajectory<T> trajectory;
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
	bool trajectoryDefined = false;
public:
	std::list<vec<int, 3>> primitives;
	void setPositionReader(PositionRead<T> positionReader, T height);
	Trajectory<T>& getTrajectory();

	MESH(T sampleDistance, vec<T, 3> position, vec<T, 9> initialRot = { init_R_3D });
	MESH(T sampleDistance, vec<T, 3> position, const std::list<vec<int, 3>>& primitives, const std::vector<vec<T, 3>>& roughSamples, vec<T, 9> initialRot = { init_R_3D });
	MESH(T sampleDistance, vec<T, 3> position, const std::list<vec<int, 3>>& primitives, const std::vector<vec<T, 3>>& roughSamples, const std::vector<vec<T, 3>>& normals, const std::vector<char>& inletTags, vec<T, 9> initialRot = { init_R_3D });

	bool isTrajectoryDefined();
};

//3D:
//------
template<typename T>
struct SPHERE3D : public MESH<T, 3>
{
	T radius;
	SPHERE3D(T sampleDistance, vec<T, 3> position, T radius, vec<T, 9> initialRot = { init_R_3D });

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};

template<typename T>
struct CYLINDER3D : public MESH<T, 3>
{
	T radius;
	T length;
	CYLINDER3D(T sampleDistance, vec<T, 3> position, T radius, T length, vec<T, 9> initialRot = { init_R_3D });

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};

template<typename T>
struct CUBOID3D : public MESH<T, 3>
{
	bool withInlet = false;
	T xDim;
	T yDim;
	T zDim;
	CUBOID3D(T sampleDistance, vec<T, 3> position, T xDim, T yDim, T zDim, vec<T, 9> initialRot = { init_R_3D }, bool withInlet = false);

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};

template<typename T>
struct CAPSULE3D : public MESH<T, 3>
{
	bool withInlet = false;
	T radius;
	T length;
	CAPSULE3D(T sampleDistance, vec<T, 3> position, T radius, T length, vec<T, 9> initialRot = { init_R_3D }, bool withInlet = false);

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};

#pragma region 2D-Body
//2D
//-----------------------
template<typename T>
struct MESH<T, 2> : public IM_BODY<T, 2>
{
protected:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
public:
	std::list<vec<int, 2>> primitives;
	MESH(T sampleDistance, vec<T, 2> position, vec<T, 4> initialRot = { init_R_2D });
	MESH(T sampleDistance, vec<T, 2> position, const std::list<vec<int, 2>>& primitives, const std::vector<vec<T, 2>>& roughSamples, vec<T, 4> initialRot = { init_R_2D });
	MESH(T sampleDistance, vec<T, 2> position, const std::list<vec<int, 2>>& primitives, const std::vector<vec<T, 2>>& roughSamples, const std::vector<vec<T, 2>>& normals, const std::vector<char>& inletTags, vec<T, 4> initialRot = { init_R_2D });


};

template<typename T>
struct SPHERE2D : public MESH<T, 2>
{
	T radius;
	SPHERE2D(T sampleDistance, vec<T, 2> position, T radius, vec<T,4> initialRot = { init_R_2D });

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};



template<typename T>
struct CUBOID2D : public MESH<T, 2>
{
	T xDim;
	T yDim;
	CUBOID2D(T sampleDistance, vec<T, 2> position, T xDim, T yDim, vec<T, 4> initialRot = {init_R_2D });

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};



template<typename T>
struct CAPSULE2D : public MESH<T, 2>
{
	T radius;
	T length;
	CAPSULE2D(T sampleDistance, vec<T, 2> position, T radius, T length, vec<T, 4> initialRot = { init_R_2D });

private:
	void generateSamples(double dh, double C_u, double rho) override;
	void mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize) override;
};


#pragma endregion
#endif // !IM_BODY