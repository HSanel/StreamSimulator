#ifndef IMB_METHOD_P
#define IMB_METHOD_P
#include <math.h>
#include <memory>
#include <map>

#include "ImmersedBody.h"
#include "ImmersedBoundaryMethod.h"
#include "LBM_Types.h"
#include "SimDomain.h"
#include "cudaErrorHandle.h"
#include "Trajectory.h"
#include "ParticleGenerator_P.h"



#define IBM_STATIC 1
#define IBM_DYNAMIC 2

template<typename T, size_t D> class IBMethod_Specialisation_P;

#pragma region IBMethod_Spec_2D
template<typename T>
class IBMethod_Specialisation_P<T, 2> :public IBMethod_Base
{
	//IBMethod_Specialisation(const IBMethod_Specialisation<T, 2>&) = delete;
	//IBMethod_Specialisation<T, 2>& operator=(const IBMethod_Specialisation<T, 2>&) = delete;
	//IBMethod_Specialisation(IBMethod_Specialisation<T, 2>&&) = delete;
	//IBMethod_Specialisation<T, 2>& operator=(IBMethod_Specialisation<T, 2>&&) = delete;
protected:
	std::vector<std::unique_ptr<MESH<T, 2>>> im_list;
	std::vector<std::unique_ptr<IM_BODY_P<T, 2>>> im_list_P;
	IBMethod_Specialisation_P(const unsigned char ibm_tag);
	vec<T, 4> getCrossProductMat(vec<T, 3> v);
public:
	T update(T currentTime, const SimDomain<T, 2>& sd, ParticleGenerator_P<T, 2> *partGenerator = nullptr);
	void calcFs(vec<T,2> * u_unc_L, vec<T,2> * F_ext_L, T* rho, vec<T, 2>* Fsg);
	void calcInlet(vec<T, 2>* u_L, T* rho_L, T* f_star, T *f, bool BGK_used);

};
#pragma endregion 

template<typename T>
class IBMethod_Specialisation_P<T, 3> :public IBMethod_Base
{
	//IBMethod_Specialisation<T, 3>& operator=(const IBMethod_Specialisation<T, 3>&) = delete;
	//IBMethod_Specialisation<T, 3>& operator=(IBMethod_Specialisation<T, 3>&&) = delete;
	// 
	//TEST
	vec<T, 3> direction{ 1,0,0 };
protected:
	std::vector<std::unique_ptr<MESH<T, 3>>> im_list;
	std::vector<std::unique_ptr<IM_BODY_P<T, 3>>> im_list_P;
	IBMethod_Specialisation_P(const unsigned char ibm_tag);
	vec<T, 9> getCrossProductMat(vec<T, 3> v);
public:
	T update(T currentTime, const SimDomain<T, 3>& sd, ParticleGenerator_P<T,3> *partGenerator = nullptr);
	void calcFs(vec<T,3>* u_unc_L, vec<T,3>* F_ext_L, T* rho, vec<T, 3>* Fsg);
	void calcInlet(vec<T, 3>* u_L, T* rho_L, T* f_star, T* f, bool BGK_used);
};

template<typename T, size_t D>
class IBMethod_P : public IDataGenerator, public IBMethod_Specialisation_P<T, D>
{
private:
	IBMethod_Specialisation_P<T, D>::sharpness;
	IBMethod_Specialisation_P<T, D>::ibm_tag;
	IBMethod_Specialisation_P<T, D>::maxPointCount;
	IBMethod_Specialisation_P<T, D>::im_list;
	int bucketSize;
public:
	IBMethod_Specialisation_P<T, D>::calcFs;
	
	IBMethod_P(const unsigned char ibm_tag, int bucketSize = 0);
	void addBody(std::unique_ptr<MESH<T, D>> im_body);
	int getBodyCount() const;
	const MESH<T, D>* getBodyAt(int idx) const;
	void reset(const SimDomain<T,D> &sd);
	
	int getMaxPointCount() const;
	void setSharpness(unsigned char sharpness);
	const unsigned char getTag() const;
	void setCudaProp(cudaDeviceProp prop);
	void setWarpCount(unsigned int warpCount);
	void copyConstantSimDom(SimDomain_dev<T, D, SimDomain<T, D>::Q> &sd_temp);
	void rescaleVelocities(const T& scale_u);
};
#endif