#include "ImmersedBoundaryMethod_P.h"
#include "ImmersedBoundaryMethod_Kernels.h"

template struct IBMethod_Specialisation_P<float, 2>;
template struct IBMethod_Specialisation_P<double, 2>;
template struct IBMethod_Specialisation_P<float, 3>;
template struct IBMethod_Specialisation_P<double, 3>;

template struct IBMethod_P<float, 2>;
template struct IBMethod_P<float, 3>;
template struct IBMethod_P<double, 2>;
template struct IBMethod_P<double, 3>;

//Specialisation
#pragma region IBM_Spec_2D
template <typename T>
IBMethod_Specialisation_P<T, 2>::IBMethod_Specialisation_P(const unsigned char ibm_tag) : IBMethod_Base(ibm_tag) {}

template <typename T>
void IBMethod_Specialisation_P<T, 2>::calcFs(vec<T, 2>* u_unc_L, vec<T, 2>* F_ext_L, T* rho_L, vec<T,2>* Fsg)
{
	//PARALLEL CODE
	if (sharpness == SHARP)
		this->kernelSize = 3;
	else if (sharpness == MEDIUM_SMOOTH || sharpness == SMOOTH)
		this->kernelSize = 5;

	for (int i = 0; i < this->im_list.size(); ++i)
	{
		calcFs2D_kernel<T> << < (im_list[i]->samples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
			(u_unc_L, F_ext_L, rho_L, im_list_P[i]->samples.get(), im_list_P[i]->velocities.get(), this->kernelSize, im_list[i]->samples.size(), this->sharpness);
	}
}

template <typename T>
void IBMethod_Specialisation_P<T, 2>::calcInlet(vec<T, 2>* u_L, T* rho_L, T* f_star, T* f, bool BGK_used)
{
	for (int i = 0; i < this->im_list.size(); ++i)
	{
		if (!im_list[i]->inletSamples.empty())
			calcInlet2D_kernel<T> << < (im_list[i]->inletSamples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
			(rho_L, u_L, f, f_star, im_list_P[i]->inletSamples.get(), im_list_P[i]->inletVelocities.get(), im_list_P[i]->inletNormals.get(), im_list[i]->inletSamples.size(), BGK_used);
	}
}

template <typename T>
vec<T, 4> IBMethod_Specialisation_P<T, 2>::getCrossProductMat(vec<T, 3> v)
{
	return { 0.0, -v[2],
			v[2],  0.0 };
}


template <typename T>
T IBMethod_Specialisation_P<T, 2>::update(T currentTime, const SimDomain<T, 2>& sd, ParticleGenerator_P<T, 2> * partGenerator)
{
	for (int i = 0; i < im_list.size(); ++i)
	{
		auto& body = im_list[i];

		/*updatePosAndVel_kernel<T, D> <<< (body->samples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize * warpCount >>>
			(im_list_P[i]->velocities.get(), im_list_P[i]->samples.get(), omega_star, body->velocity_center, body->R, oldPos, body->position, body->samples.size());

		im_list_P[i]->copySamplesToHost(*body);*/
	}
	return 0;
}

#pragma endregion

template <typename T>
IBMethod_Specialisation_P<T, 3>::IBMethod_Specialisation_P(const unsigned char ibm_tag) : IBMethod_Base(ibm_tag) {}

template <typename T>
void IBMethod_Specialisation_P<T, 3>::calcFs(vec<T,3>* u_unc_L, vec<T,3>* F_ext_L, T* rho_L, vec<T, 3>* Fsg)
{
	//PARALLEL CODE
	if (sharpness == SHARP)
		this->kernelSize = 3;
	else if (sharpness == MEDIUM_SMOOTH || sharpness == SMOOTH)
		this->kernelSize = 5;

	for (int i = 0; i < this->im_list.size(); ++i)
	{
		calcFs3D_kernel<T> <<< (im_list[i]->samples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >>>
			(u_unc_L, F_ext_L, rho_L, im_list_P[i]->samples.get(), im_list_P[i]->velocities.get(), Fsg, this->kernelSize, im_list[i]->samples.size(), this->sharpness);
	}
}

template <typename T>
void IBMethod_Specialisation_P<T, 3>::calcInlet(vec<T, 3>* u_L, T* rho_L, T* f_star, T* f, bool BGK_used)
{
	for (int i = 0; i < this->im_list.size(); ++i)
	{
		if (!im_list[i]->inletSamples.empty())
			calcInlet3D_kernel<T> <<< (im_list[i]->inletSamples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >>>
				(rho_L, u_L, f, f_star, im_list_P[i]->inletSamples.get(), im_list_P[i]->inletVelocities.get(), im_list_P[i]->inletNormals.get(), im_list[i]->inletSamples.size(), BGK_used);
	}
}

template <typename T>
vec<T, 9> IBMethod_Specialisation_P<T, 3>::getCrossProductMat(vec<T, 3> v)
{
	return { 0.0, -v[2],  v[1],
			v[2],  0.0,  -v[0],
			-v[1], v[0],  0.0};
}

//TEST
double alpha = 0;

template <typename T>
T IBMethod_Specialisation_P<T, 3>::update(T currentTime, const SimDomain<T, 3>& sd, ParticleGenerator_P<T, 3> * partGenerator)
{
	CustomVecLength<T,3> vecLength;
	T uMax_mag = 0;
	for (int i = 0; i < im_list.size(); ++i)
	{
		auto& body = im_list[i];
		Trajectory<T> trajectory = body->getTrajectory();

		vec<T, 3> oldPos = body->position;

		if (body->isTrajectoryDefined())
		{
			vec<T,3> pos = trajectory.getPosition(currentTime) / sd.getGridSize();

			//TEST
			alpha += 0.01 * sd.getTimeStep();
			body->R = { (T)cos(alpha),0,(T)sin(alpha),0,1,0,(T)-sin(alpha),0,(T)cos(alpha) };


			//body->R = trajectory.getRotation();
		/*	body->position[0] = pos[0];
			body->position[2] = pos[2];*/

			body->position += this->direction * 0.1 * sd.getTimeStep();
		}
		int sourceID = body->sourceID;
		if (partGenerator != nullptr && sourceID != -1)
			partGenerator->updateSourcePosition(body->position, body->R, currentTime, sourceID);

		body->inlet_velocity = body->R * body->inlet_velocity;
		vec<T, 3> inletVelocity = body->inlet_velocity * body->inletVelocityScaler(currentTime);
		
		T inletMag = vecLength.length(inletVelocity);
		if (uMax_mag < inletMag)
			uMax_mag = inletMag;

		//updatePosAndVel_kernel<T, 3> <<< (body->samples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize * warpCount >>>
		//	(im_list_P[i]->velocities.get(), im_list_P[i]->samples.get(), im_list_P[i]->normals.get(), body->R, oldPos, body->position, body->samples.size());

		//updatePosAndVel_kernel<T, 3> << < (body->inletSamples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
		//	(im_list_P[i]->inletVelocities.get(), im_list_P[i]->inletSamples.get(), im_list_P[i]->inletNormals.get(), inletVelocity, body->R, oldPos, body->position, body->inletSamples.size());

		updatePosAndVel_kernel<T, 3> << < (body->samples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
			(im_list_P[i]->velocities.get(), im_list_P[i]->samples.get(), im_list_P[i]->normals.get(), im_list_P[i]->is_inlet.get(), inletVelocity, body->R, oldPos, body->position, body->samples.size());

		im_list_P[i]->copySamplesToHost(*body);
	}

	return uMax_mag;
}

//---------------------------------
template<typename T, size_t D>
IBMethod_P<T, D>::IBMethod_P(const unsigned char ibm_tag, int bucketSize) : IBMethod_Specialisation_P<T, D>(ibm_tag), bucketSize(bucketSize) {}

template<typename T, size_t D>
void IBMethod_P<T, D>::copyConstantSimDom(SimDomain_dev<T, D, SimDomain<T, D>::Q> &sd_temp)
{
	HANDLE_ERROR
	(
		cudaMemcpyToSymbol(sd_dev<T, D, SimDomain<T, D>::Q>, &sd_temp, sizeof(SimDomain_dev<T, D, SimDomain<T, D>::Q>))
	);
}

////Base Methods:
template <typename T, size_t D>
void IBMethod_P<T, D>::addBody(std::unique_ptr<MESH<T, D>> im_body) {im_list.push_back(std::move(im_body));}

template <typename T, size_t D>
int IBMethod_P<T, D>::getBodyCount() const { return im_list.size(); }

template <typename T, size_t D>
const MESH<T, D>* IBMethod_P<T, D>::getBodyAt(int idx) const { return im_list[idx].get(); }

//Base Methods:
template <typename T, size_t D>
const unsigned char IBMethod_P<T, D>::getTag() const
{
	return ibm_tag;
}

template <typename T, size_t D>
void IBMethod_P<T, D>::setCudaProp(cudaDeviceProp prop) { this->prop = prop; }

template <typename T, size_t D>
void IBMethod_P<T, D>::setWarpCount(unsigned int warpCount) { this->warpCount = warpCount; }

template <typename T, size_t D>
void IBMethod_P<T, D>::reset(const SimDomain<T, D>& sd)
{
	maxPointCount = 0;
	im_list_P.clear();
	vec<grid_size_t, 3> gridDim_L{ sd.getGridDim_L(0), sd.getGridDim_L(1), sd.getGridDim_L(2) };
	for (auto& body : im_list)
	{
		try
		{
			if (this->bucketSize > 0)
				executeGenerator(*body, (double) sd.getGridSize(), (double) sd.getC_u(), (double) sd.getRho(), gridDim_L, this->bucketSize);
			else
				executeGenerator(*body, (double) sd.getGridSize(), (double) sd.getC_u(), (double) sd.getRho());

			maxPointCount += body->samples.size() + body->inletSamples.size();
			im_list_P.push_back(std::make_unique<IM_BODY_P<T, D>>(IM_BODY_P<T, D>{*body}));
		}
		catch (std::exception e)
		{
			std::cerr << "ERROR::RESET IMMERSED BOUNDARY:: " << e.what() << std::endl;
		}
	}
}

template <typename T, size_t D>
void IBMethod_P<T, D>::rescaleVelocities(const T& scale_u)
{
	for (int i = 0; i < im_list.size(); ++i)
	{
		im_list[i]->inlet_velocity *= scale_u;
		im_list[i]->velocity_center *= scale_u;
		rescaleVelocities_kernel<T,D> <<< (im_list[i]->samples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize * warpCount >>>
			(im_list_P[i]->velocities.get(), scale_u, im_list[i]->samples.size());

		if(!im_list[i]->inletSamples.empty())
			rescaleVelocities_kernel<T, D> << < (im_list[i]->inletSamples.size() + prop.warpSize * warpCount - 1) / (prop.warpSize * warpCount), prop.warpSize* warpCount >> >
			(im_list_P[i]->inletVelocities.get(), scale_u, im_list[i]->inletSamples.size());
	}
}

template <typename T, size_t D>
int IBMethod_P<T, D>::getMaxPointCount() const { return maxPointCount; }

template <typename T, size_t D>
void IBMethod_P<T, D>::setSharpness(unsigned char sharpness) { this->sharpness = sharpness; }

