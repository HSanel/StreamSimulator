#ifndef IMB_METHOD
#define IMB_METHOD
#include <math.h>
#include <memory>
#include <vector>

#include "ImmersedBody.h"
#include "LBM_Types.h"
#include "SimDomain.h"

#define IBM_STATIC 1
#define IBM_DYNAMIC 2

#define SHARP 0
#define	MEDIUM_SMOOTH 1
#define SMOOTH 2

template <typename T, size_t D>
struct Fs_DATA {
	int pos_L;
	vec<T, D> Fs;
};

class IBMethod_Base
{
	//IBMethod_Base(const IBMethod_Base&) = delete;
	//IBMethod_Base& operator=(const IBMethod_Base&) = delete;
	//IBMethod_Base(IBMethod_Base&&) = delete;
	//IBMethod_Base& operator=(IBMethod_Base&&) = delete;
protected:
	cudaDeviceProp prop;
	unsigned int warpCount;
	int kernelSize;
	unsigned char sharpness = MEDIUM_SMOOTH;
	const unsigned char ibm_tag;
	int maxPointCount = 0;
	IBMethod_Base(const unsigned char ibm_tag);
};

template<typename T, size_t D> class IBMethod_Specialisation;

template<typename T>
class IBMethod_Specialisation<T, 2>:public IBMethod_Base
{
protected:
	std::vector<std::unique_ptr<IM_BODY<T, 2>>> im_list;
	IBMethod_Specialisation(const unsigned char ibm_tag);
	vec<T, 4> getCrossProductMat(vec<T, 3> v);

public:
	void calcFs(SimDomain<T, 2> &sd, vec<T,2> * u, T* rho, std::vector<Fs_DATA<T, 2>>& Fs_dataList);
	void calcInlet(SimDomain<T, 2>& sd, vec<InletGhostCells<T, 2>, 9>* sourceNodes);
};

template<typename T>
class IBMethod_Specialisation<T, 3>:public IBMethod_Base
{
protected:
	std::vector<std::unique_ptr<IM_BODY<T, 3>>> im_list;
	IBMethod_Specialisation(const unsigned char ibm_tag);
	vec<T, 9> getCrossProductMat(vec<T, 3> v);
public:
	void calcFs(SimDomain<T,3> &sd, vec<T,3>* u, T* rho, std::vector<Fs_DATA<T, 3>>& Fs_dataList);
	void calcInlet(SimDomain<T, 3>& sd, vec<InletGhostCells<T, 3>, 27>* sourceNodes);
};

template<typename T, size_t D>
class IBMethod : public IDataGenerator, public IBMethod_Specialisation<T,D>
{
private:
	IBMethod_Specialisation<T, D>::sharpness;
	IBMethod_Specialisation<T, D>::ibm_tag;
	IBMethod_Specialisation<T, D>::maxPointCount;
	IBMethod_Specialisation<T, D>::im_list;
public:
	IBMethod_Specialisation<T, D>::calcFs;

	IBMethod(const unsigned char ibm_tag);
	void addBody(std::unique_ptr<IM_BODY<T, D>> im_body);
	int getBodyCount() const;
	const IM_BODY<T, D>* getBodyAt(int idx) const;
	void reset(T dh, T C_u, T rho);
	void update(T timeStep);
	int getMaxPointCount() const;
	void setSharpness(unsigned char sharpness);
	const unsigned char getTag() const;
	void rescaleVelocities(const T& scale_u);
	
};
#endif