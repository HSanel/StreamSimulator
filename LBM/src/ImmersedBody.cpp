#include "ImmersedBody.h"

template struct IM_BODY<float, 2>;
template struct IM_BODY<float, 3>;
template struct IM_BODY<double, 2>;
template struct IM_BODY<double, 3>;

template struct IM_BODY_P<float, 2>;
template struct IM_BODY_P<float, 3>;
template struct IM_BODY_P<double, 2>;
template struct IM_BODY_P<double, 3>;

template struct SPHERE3D<float>;
template struct SPHERE3D<double>;

template struct SPHERE2D<float>;
template struct SPHERE2D<double>;
template struct CYLINDER3D<float>;
template struct CYLINDER3D<double>;

template struct CUBOID2D<float>;
template struct CUBOID2D<double>;
template struct CUBOID3D<float>;
template struct CUBOID3D<double>;

template struct CAPSULE2D<float>;
template struct CAPSULE2D<double>;
template struct CAPSULE3D<float>;
template struct CAPSULE3D<double>;

template struct MESH<float, 2>;
template struct MESH<double, 2>;
template struct MESH<float, 3>;
template struct MESH<double, 3>;

namespace
{
	unsigned int expandBits(unsigned int v)
	{
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}

	template<typename T>
	unsigned int morton3D(T x, T y, T z)
	{
		x = std::min(std::max(x * (T)1024.0, (T)0.0), (T)1023.0);
		y = std::min(std::max(y * (T)1024.0, (T)0.0), (T)1023.0);
		z = std::min(std::max(z * (T)1024.0, (T)0.0), (T)1023.0);
		unsigned int xx = expandBits((unsigned int)x);
		unsigned int yy = expandBits((unsigned int)y);
		unsigned int zz = expandBits((unsigned int)z);
		return xx * 4 + yy * 2 + zz;
	}

	template <typename T>
	void mortonOrderSort3D(unsigned int bucketSize, vec<grid_size_t, 3> gridDim_L, std::vector<vec<T, 3>>& samples, std::vector<vec<T, 3>>& velocities)
	{
		int maxBucketCount_x = std::ceil(gridDim_L[0] / bucketSize);
		int maxBucketCount_y = std::ceil(gridDim_L[1] / bucketSize);
		int maxBucketCount_z = std::ceil(gridDim_L[2] / bucketSize);
		int maxBucketCount = maxBucketCount_x * maxBucketCount_y * maxBucketCount_z;

		std::vector<std::multimap<unsigned int, vec<T, 3>>> samples_bin(maxBucketCount);
		std::vector<std::multimap<unsigned int, vec<T, 3>>> velocities_bin(maxBucketCount);

		for (int i = 0; i < samples.size(); ++i)
		{
			int posX = (int)floor(samples[i][0]) - (int)floor(samples[i][0]) % bucketSize;
			int posY = (int)floor(samples[i][1]) - (int)floor(samples[i][1]) % bucketSize;
			int posZ = (int)floor(samples[i][2]) - (int)floor(samples[i][2]) % bucketSize;
			int pos = (posZ / bucketSize * maxBucketCount_y + posY / bucketSize) * maxBucketCount_x + posX / bucketSize;

			samples_bin[pos].emplace(morton3D<T>((samples[i][0] - posX) / bucketSize, (samples[i][1] - posY) / bucketSize, (samples[i][2] - posZ) / bucketSize), samples[i]);
			velocities_bin[pos].emplace(morton3D<T>((samples[i][0] - posX) / bucketSize, (samples[i][1] - posY) / bucketSize, (samples[i][2] - posZ) / bucketSize), velocities[i]);
		}

		samples.clear();
		velocities.clear();

		for (int pos = 0; pos < maxBucketCount; ++pos)
		{
			for (auto e = samples_bin[pos].begin(); e != samples_bin[pos].end(); ++e)
			{
				samples.push_back(e->second);
			}

			for (auto e = velocities_bin[pos].begin(); e != velocities_bin[pos].end(); ++e)
			{
				velocities.push_back(e->second);
			}
		}

	}


#pragma region 2D_HelperFunctions
	template <typename T>
	void mortonOrderSort2D(unsigned int bucketSize, vec<grid_size_t, 3> gridDim_L, std::vector<vec<T, 2>>& samples, std::vector<vec<T, 2>>& velocities)
	{
		int maxBucketCount_x = std::ceil(gridDim_L[0] / bucketSize);
		int maxBucketCount_y = std::ceil(gridDim_L[1] / bucketSize);
		int maxBucketCount = maxBucketCount_x * maxBucketCount_y;

		std::vector<std::multimap<unsigned int, vec<T, 2>>> samples_bin(maxBucketCount);
		std::vector<std::multimap<unsigned int, vec<T, 2>>> velocities_bin(maxBucketCount);

		for (int i = 0; i < samples.size(); ++i)
		{
			int posX = ((int)floor(samples[i][0]) - (int)floor(samples[i][0]) % bucketSize);
			int posY = ((int)floor(samples[i][1]) - (int)floor(samples[i][1]) % bucketSize);
			int pos = posY / bucketSize * maxBucketCount_x + posX / bucketSize;

			samples_bin[pos].emplace(morton3D<T>((samples[i][0] - posX) / bucketSize, (samples[i][1] - posY) / bucketSize, 0.0), samples[i]);
			velocities_bin[pos].emplace(morton3D<T>((samples[i][0] - posX) / bucketSize, (samples[i][1] - posY) / bucketSize, 0.0), velocities[i]);
		}

		samples.clear();
		velocities.clear();

		for (int pos = 0; pos < maxBucketCount; ++pos)
		{
			for (auto e = samples_bin[pos].begin(); e != samples_bin[pos].end(); ++e)
			{
				samples.push_back(e->second);
			}

			for (auto e = velocities_bin[pos].begin(); e != velocities_bin[pos].end(); ++e)
			{
				velocities.push_back(e->second);
			}
		}

	}

#pragma endregion
}

//INTERFACE
//-------------------------------------------
template<typename T, size_t D>
IM_BODY_P<T,D>::IM_BODY_P(IM_BODY<T, D>& body)
{
	samples = make_location_array<location_gpu, vec<T,D>>(body.samples.size());
	normals = make_location_array<location_gpu, vec<T, D>>(body.normals.size());
	velocities = make_location_array<location_gpu, vec<T, D>>(body.velocities.size());

	inletSamples = make_location_array<location_gpu, vec<T, D>>(body.inletSamples.size());
	inletNormals = make_location_array<location_gpu, vec<T, D>>(body.inletNormals.size());
	inletVelocities = make_location_array<location_gpu, vec<T, D>>(body.inletVelocities.size());
	//test
	is_inlet = make_location_array<location_gpu, char>(body.is_inlet.size());
	
	HANDLE_ERROR(cudaMemcpy(this->samples.get(), body.samples.data(), sizeof(vec<T, D>) * body.samples.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(this->normals.get(), body.normals.data(), sizeof(vec<T, D>) * body.normals.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(this->velocities.get(), body.velocities.data(), sizeof(vec<T, D>) * body.velocities.size(), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(this->inletSamples.get(), body.inletSamples.data(), sizeof(vec<T, D>) * body.inletSamples.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(this->inletNormals.get(), body.inletNormals.data(), sizeof(vec<T, D>) * body.inletNormals.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(this->inletVelocities.get(), body.inletVelocities.data(), sizeof(vec<T, D>) * body.inletVelocities.size(), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(this->is_inlet.get(), body.is_inlet.data(), sizeof(char) * body.is_inlet.size(), cudaMemcpyHostToDevice));
}

template<typename T, size_t D>
void IM_BODY_P<T, D>::copySamplesToHost(IM_BODY<T, D>& body)
{
	HANDLE_ERROR(cudaMemcpy(body.samples.data(), samples.get(), sizeof(vec<T, D>) * body.samples.size(), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(body.inletSamples.data(), inletSamples.get(), sizeof(vec<T, D>) * body.inletSamples.size(), cudaMemcpyDeviceToHost));
}

void IDataGenerator::generateSamples(double dh, double C_u, double rho)
{
	throw std::runtime_error("ERROR:: generateSample can not be called from IDataGenerator or IM_BODY");
}

void IDataGenerator::executeGenerator(IDataGenerator& generator, double dh, double C_u, double rho)
{
	generator.generateSamples(dh, C_u, rho);
}


void IDataGenerator::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	throw std::runtime_error("ERROR:: mortonOrderSort can not be called from IDataGenerator or IM_BODY");
}

void IDataGenerator::executeGenerator(IDataGenerator& generator, double dh, double C_u, double rho, vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	generator.generateSamples(dh, C_u, rho);
	generator.mortonOrderSort(gridDim_L, bucketSize);
}

//BASE CLASS
//--------------------------------------------

template<typename T, size_t D>
IM_BODY<T,D>::IM_BODY(T sampleDistance, vec<T, D> position, const std::vector<vec<T, D>> &samples,const std::vector<vec<T, D>> &normals,const std::vector<char> &inletTags, vec<T, D* D> initialRot)
	: sampleDistance(sampleDistance), position(position), velocity_center({}), R(initialRot), samples(samples), normals(normals), is_inlet(inletTags) {}

//MESH3D
//--------------------
template<typename T>
MESH<T, 3>::MESH(T sampleDistance, vec<T, 3> position, vec<T, 9> initialRot) :
	IM_BODY<T, 3>(sampleDistance, position, {}, {}, {}, initialRot) {}

template<typename T>
MESH<T, 3>::MESH(T sampleDistance, vec<T, 3> position, const std::list<vec<int, 3>>& primitives, const std::vector<vec<T, 3>>& roughSamples, const std::vector<vec<T, 3>>& normals,
	const std::vector<char>& inletTags, vec<T, 9> initialRot)
	: IM_BODY<T, 3>(sampleDistance, position, roughSamples, normals, inletTags, initialRot), primitives(primitives) {}

template<typename T>
MESH<T, 3>::MESH(T sampleDistance, vec<T, 3> position, const std::list<vec<int, 3>>& primitives, const std::vector<vec<T, 3>>& roughSamples, vec<T, 9> initialRot)
	: IM_BODY<T, 3>(sampleDistance, position, roughSamples, {}, {}, initialRot), primitives(primitives) {}




template<typename T>
void MESH<T, 3>::generateSamples(double dh, double con_u, double rho)
{
	CustomVecLength<T, 3> vecLength;
	T C_l = 1.0, C_u = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	if (trajectoryDefined && normilized_parameter)
	{
		this->position[0] /= dh;
		this->position[2] /= dh;
	}
	else if (trajectoryDefined)
		this->position /= dh;
	else
		this->position /= C_l;

	this->inlet_velocity /= C_u;

	bool sampleAppended = true;
	T area = (T)0.0;

	velocities.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();


	for (int i = 0; i < samples.size(); ++i)
	{
		samples[i] = this->R * (samples[i]) / C_l + this->position;

		if (!normals.empty())
		{
			normals[i] = R * normals[i];
		}
	}

	for (auto it = primitives.begin(); it != primitives.end(); ++it)
	{
		int firstIdx = (*it)[0];
		int secondIdx = (*it)[1];
		int thirdIdx = (*it)[2];
		area += (T)0.5 * vecLength.length(cross(samples[secondIdx] - samples[firstIdx], samples[thirdIdx] - samples[firstIdx]));
	}

	while (sampleAppended)
	{
		sampleAppended = false;

		for (auto it = primitives.begin(); it != primitives.end(); ++it)
		{
			int firstIdx = (*it)[0];
			int secondIdx = (*it)[1];
			int thirdIdx = (*it)[2];
			T dist_1_2 = vecLength.length(this->samples[secondIdx] - this->samples[firstIdx]);
			T dist_2_3 = vecLength.length(this->samples[thirdIdx] - this->samples[secondIdx]);
			T dist_3_1 = vecLength.length(this->samples[firstIdx] - this->samples[thirdIdx]);

			if (dist_1_2 > this->sampleDistance || dist_2_3 > this->sampleDistance || dist_3_1 > this->sampleDistance)
			{
				auto m1 = this->samples[firstIdx] + (T)0.5 * (this->samples[secondIdx] - this->samples[firstIdx]);
				auto m2 = this->samples[secondIdx] + (T)0.5 * (this->samples[thirdIdx] - this->samples[secondIdx]);
				auto m3 = this->samples[firstIdx] + (T)0.5 * (this->samples[thirdIdx] - this->samples[firstIdx]);

				samples.push_back(m1);
				samples.push_back(m2);
				samples.push_back(m3);

				if (!this->is_inlet.empty())
				{
					if (this->normals.empty())
						throw std::runtime_error("ERROR::Normals for inlet not defined");

					auto mn_1 = this->normals[firstIdx] + this->normals[secondIdx];
					auto mn_2 = this->normals[secondIdx] + this->normals[thirdIdx];
					auto mn_3 = this->normals[firstIdx] + this->normals[thirdIdx];

					mn_1 /= vecLength.length(mn_1);
					mn_2 /= vecLength.length(mn_2);
					mn_3 /= vecLength.length(mn_3);

					auto mT_1 = this->is_inlet[firstIdx] && this->is_inlet[secondIdx] ? true : false;
					auto mT_2 = this->is_inlet[secondIdx] && this->is_inlet[thirdIdx] ? true : false;
					auto mT_3 = this->is_inlet[firstIdx] && this->is_inlet[thirdIdx] ? true : false;



					this->normals.push_back(mn_1);
					this->normals.push_back(mn_2);
					this->normals.push_back(mn_3);

					this->is_inlet.push_back(mT_1);
					this->is_inlet.push_back(mT_2);
					this->is_inlet.push_back(mT_3);
				}

				auto m1_Idx = static_cast<int>(samples.size() - 3);
				auto m2_Idx = static_cast<int>(samples.size() - 2);
				auto m3_Idx = static_cast<int>(samples.size() - 1);

				vec<int, 3> firstPrimitive{ firstIdx, m1_Idx, m3_Idx };
				vec<int, 3> secondPrimitive{ m1_Idx, secondIdx, m2_Idx };
				vec<int, 3> thirdPrimitive{ m1_Idx, m2_Idx, m3_Idx };
				vec<int, 3> fourthPrimitive{ m3_Idx, m2_Idx, thirdIdx };

				primitives.insert(it, firstPrimitive);
				primitives.insert(it, secondPrimitive);
				primitives.insert(it, thirdPrimitive);
				primitives.insert(it, fourthPrimitive);

				primitives.erase(it--);

				sampleAppended = true;
			}
		}
	}


	if (this->is_inlet.empty())
	{
		std::vector<cy::Point3<T>> tightSamples;

		for (auto sample : this->samples)
			tightSamples.push_back(cy::Point3<T>{ sample[0], sample[1], sample[2]});

		cy::WeightedSampleElimination<cy::Point3<T>, T, 3, int > wse;
		int fiteredSamplesSize = wse.GetMinOutputSize(3, this->sampleDistance, area);
		std::vector<cy::Point3<T>> filteredSamples(fiteredSamplesSize);

		wse.Eliminate(tightSamples.data(), tightSamples.size(),
			filteredSamples.data(), filteredSamples.size(),
			true, this->sampleDistance * (T)2.0, 2);

		this->samples.clear();

		this->samples.resize(fiteredSamplesSize);
		this->velocities.resize(fiteredSamplesSize, vec<T, 3>{});

		for (int i = 0; i < filteredSamples.size(); ++i)
		{
			this->samples[i] = { filteredSamples[i].x, filteredSamples[i].y, filteredSamples[i].z };
		}
	}
	else
	{
		std::vector<cy::Point3<T>> tightSamples, tightNormals;

		for (int i = 0; i < this->samples.size(); ++i)
		{
			tightSamples.push_back(cy::Point3<T>{ this->samples[i][0], this->samples[i][1], this->samples[i][2]});
			tightNormals.push_back(cy::Point3<T>{ this->normals[i][0], this->normals[i][1], this->normals[i][2]});
		}

		cy::WeightedSampleElimination<cy::Point3<T>, T, 3, int > wse;
		int fiteredSamplesSize = wse.GetMinOutputSize(3, this->sampleDistance, area);

		std::vector<cy::Point3<T>> filteredSamples(fiteredSamplesSize);
		std::vector<cy::Point3<T>> filteredNormals(fiteredSamplesSize);
		std::vector<char> filteredTags(fiteredSamplesSize);

		wse.Eliminate(tightSamples.data(), tightNormals.data(), this->is_inlet.data(), tightSamples.size(),
			filteredSamples.data(), filteredNormals.data(), filteredTags.data(), filteredSamples.size(),
			true, this->sampleDistance * (T)2.0, 2);

		this->samples.clear();
		this->normals.clear();
		this->is_inlet.clear();


		for (int i = 0; i < fiteredSamplesSize; ++i)
		{
			if (filteredTags[i])
			{
				vec<T, 3> n{ filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z };
				this->inletSamples.push_back({ filteredSamples[i].x, filteredSamples[i].y, filteredSamples[i].z });
				this->inletVelocities.push_back({ });
				this->inletNormals.push_back(n);
			}
			else
			{
				this->samples.push_back({ filteredSamples[i].x, filteredSamples[i].y, filteredSamples[i].z });
				this->velocities.push_back({});
				this->normals.push_back({ filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z });
			}
		}
	}
}


template<typename T>
void MESH<T, 3>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort3D<T>(bucketSize, gridDim_L, samples, velocities);
}

template<typename T>
void MESH<T, 3>::setPositionReader(PositionRead<T> positionReader, T height)
{
	this->trajectory.setValues(positionReader);
	vec<T,3> pos = this->trajectory.getPosition(0);
	this->position[0] = pos[0];
	this->position[1] = height;
	this->position[2] = pos[2];
	this->R = this->trajectory.getRotation();
	trajectoryDefined = true;
}

template<typename T>
Trajectory<T>& MESH<T, 3>::getTrajectory()
{
	return this->trajectory;
}

template<typename T>
bool MESH<T, 3>::isTrajectoryDefined() { return trajectoryDefined; }

//3D::
//SPHERE3D
//--------------------------------------------



template<typename T>
SPHERE3D<T>::SPHERE3D(T sampleDistance, vec<T, 3> position, T radius, vec<T, 9> initialRot) :MESH<T, 3>(sampleDistance, position, initialRot), radius(radius) {}

template<typename T>
void SPHERE3D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_l = 1.0, C_u = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	if (trajectoryDefined && normilized_parameter)
	{
		this->position[0] /= dh;
		this->position[2] /= dh;
	}
	else if (trajectoryDefined)
		this->position /= dh;
	else
		this->position /= C_l;

	this->inlet_velocity /= C_u;
	this->radius /= C_l;

	samples.clear();
	velocities.clear();
	normals.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	vec<T,3> hVec;
	hVec[0] = 0.0;
	hVec[1] = radius;
	hVec[2] = 0.0;

	hVec = R * hVec;
	hVec += position;
	
	samples.push_back(hVec);

	hVec[0] = 0.0;
	hVec[1] = -radius;
	hVec[2] = 0.0;
	hVec = R * hVec;
	hVec += position;
	samples.push_back(hVec);
	
	T circumferenceLat = 2.0 * PI<T> * radius;
	int  latitudeSize = static_cast<int>(circumferenceLat / (sampleDistance * 2));
	T dPhi = 2.0 * PI<T> * sampleDistance / circumferenceLat;
	T phi, gamma;


	for (int i = 1; i <= latitudeSize; ++i)
	{
		phi = static_cast<T>(i) * dPhi - PI<T> / 2.f;
		T currentRadius = radius * std::cos(phi);
		T circumferenceLong = 2.0 * PI<T> * currentRadius;
		int longitudeSize = static_cast<int>(circumferenceLong / sampleDistance) + 1;
		T dGamma = 2.0 * PI<T> * sampleDistance / circumferenceLong;
		

		for (int k = 0; k <= longitudeSize; ++k)
		{
			gamma = static_cast<float>(k) * dGamma;
			hVec[0] = currentRadius * std::cos(gamma);
			hVec[1] = radius * std::sin(phi);
			hVec[2] = currentRadius * std::sin(gamma);

			hVec = R * hVec;
			hVec += position;
			samples.push_back(hVec);
		}
	}

	velocities.resize(samples.size(), vec<T,3>{});
}

template<typename T>
void SPHERE3D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort3D<T>(bucketSize, gridDim_L, samples, velocities);
}

//CYLINDER3D
//----------------------------------------------

template<typename T>
CYLINDER3D<T>::CYLINDER3D(T sampleDistance, vec<T, 3> position, T radius, T length, vec<T, 9> initialRot) :MESH<T, 3>(sampleDistance, position, initialRot), radius(radius), length(length) {}

template<typename T>
void CYLINDER3D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_u = 1.0, C_l = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	if (trajectoryDefined && normilized_parameter)
	{
		this->position[0] /= dh;
		this->position[2] /= dh;
	}
	else if (trajectoryDefined)
		this->position /= dh;
	else
		this->position /= C_l;

	this->inlet_velocity /= C_u;
	this->radius /= C_l;
	this->length /= C_l;

	samples.clear();
	velocities.clear();
	normals.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	vec<T, 3> hVec;

	hVec[0] = 0.0;
	hVec[1] = -length / 2.0;
	hVec[2] = 0.0;
	hVec = R * hVec;
	hVec += position;
	samples.push_back(hVec);

	hVec[0] = 0.0;
	hVec[1] = length / 2.0;
	hVec[2] = 0.0;
	hVec = R * hVec;
	hVec += position;
	samples.push_back(hVec);

	int sampleSizeRadius = radius / sampleDistance;
	T currentRadius = 0.0;
	T gamma = 0.0;

	for (int k = 1; k < sampleSizeRadius; ++k)
	{
		currentRadius = static_cast<T>(k) * sampleDistance;
		T circumference = 2.0 * PI<T> *currentRadius;
		int  samplesSize = static_cast<int>(circumference / sampleDistance) + 1;
		T dRad = 2.0 * PI<T> *sampleDistance / circumference;

		for (int i = 0; i <= samplesSize; ++i)
		{
			T x = currentRadius * std::cos(gamma);
			T z = currentRadius * std::sin(gamma);

			hVec[0] = x;
			hVec[1] = -length / 2.0;
			hVec[2] = z;
			hVec = R * hVec;
			hVec += position;
			samples.push_back(hVec);

			hVec[0] = x;
			hVec[1] = length / 2.0;
			hVec[2] = z;
			hVec = R * hVec;
			hVec += position;
			samples.push_back(hVec);
			gamma = static_cast<T>(i) * dRad;
		}
	}

	T circumference = 2.0 * PI<T> *radius;
	int  samplesSize = static_cast<int>(circumference / sampleDistance) + 1;
	int sampleSizeLength = static_cast<int>(length / sampleDistance);
	T dRad = 2.f * PI<T> *sampleDistance / circumference;

	for (int k = 0; k <= sampleSizeLength; ++k)
		for (int i = 0; i <= samplesSize; ++i)
		{
			gamma = static_cast<T>(i) * dRad;
			hVec[0] = radius * std::cos(gamma);
			hVec[2] = radius * std::sin(gamma);

			if (k == sampleSizeLength)
				hVec[1] = length / 2.f;
			else
				hVec[1] = static_cast<T>(k) * sampleDistance - length / 2.0;

			hVec = R * hVec;
			hVec += position;
			samples.push_back(hVec);
		}

	velocities.resize(samples.size(), vec<T, 3>{});
}

template<typename T>
void CYLINDER3D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort3D<T>(bucketSize, gridDim_L, samples, velocities);
}

//CUBOID3D
//----------------------------------------------

template<typename T>
CUBOID3D<T>::CUBOID3D(T sampleDistance, vec<T, 3> position, T xDim, T yDim, T zDim, vec<T, 9> initialRot, bool withInlet) 
	:MESH<T, 3>(sampleDistance, position, initialRot), xDim(xDim), yDim(yDim), zDim(zDim), withInlet(withInlet) {}

template<typename T>
void CUBOID3D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_u = 1.0, C_l = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	if (trajectoryDefined && normilized_parameter)
	{
		this->position[0] /= dh;
		this->position[2] /= dh;
	}
	else if (trajectoryDefined)
		this->position /= dh;
	else
		this->position /= C_l;

	this->inlet_velocity /= C_u;
	this->xDim /= C_l;
	this->yDim /= C_l;
	this->zDim /= C_l;


	samples.clear();
	velocities.clear();
	normals.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	vec<T, 3> leftCornerPos{};

	leftCornerPos[0] -= xDim / 2.0;
	leftCornerPos[1] -= yDim / 2.0;
	leftCornerPos[2] -= zDim / 2.0;

	int sampleCountX = static_cast<int>(xDim / sampleDistance);
	int sampleCountY = static_cast<int>(yDim / sampleDistance);
	int sampleCountZ = static_cast<int>(zDim / sampleDistance);

	for (int iy = 0; iy < sampleCountY; ++iy)
	{
		for (int ix = 0; ix < sampleCountX; ++ix)
		{
			vec<T, 3> front{ leftCornerPos[0] + static_cast<T>(ix) * sampleDistance, leftCornerPos[1] + static_cast<T>(iy) * sampleDistance, leftCornerPos[2] + zDim };
			vec<T, 3> back{ (leftCornerPos[0] + xDim) - static_cast<T>(ix) * sampleDistance, (leftCornerPos[1] + yDim) - static_cast<T>(iy) * sampleDistance, leftCornerPos[2] };
			front = R * front + position;
			back = R * back + position;

			samples.push_back(front);
			is_inlet.push_back(false);
			normals.push_back(R * vec<T, 3>{ 0, 0, 1 });

			samples.push_back(back);
			is_inlet.push_back(false);
			normals.push_back(R * vec<T, 3>{ 0, 0, -1 });

		}
	}

	for (int iy = 0; iy <= sampleCountY ; ++iy)
	{
		for (int iz = 0; iz < sampleCountZ ; ++iz)
		{
			vec<T, 3> right{ leftCornerPos[0] + xDim, leftCornerPos[1] + static_cast<T>(iy) * sampleDistance, (leftCornerPos[2] + zDim) - static_cast<T>(iz) * sampleDistance };
			vec<T, 3> left{ leftCornerPos[0], (leftCornerPos[1] + yDim) - static_cast<T>(iy) * sampleDistance, leftCornerPos[2] + static_cast<T>(iz) * sampleDistance };
			right = R * right + position;
			left = R * left + position;


			vec<T, 3> n = R * vec<T, 3>{ 1, 0, 0 };
			if (withInlet)
			{
				//is_inlet.push_back(true); //with Edges
				//if (iy > 2 && iy < (sampleCountY - 2)
				//	&& iz> 2 && iz < (sampleCountZ - 2))
				//{
				//	is_inlet.push_back(true);
				//	/*inletSamples.push_back(right);
				//	inletNormals.push_back(n);*/
				//	samples.push_back(right);
				//	normals.push_back(n);
				//}	
				//else
				{
					is_inlet.push_back(true);
					samples.push_back(right);
					normals.push_back(n);
				}
			}
			else
			{
				is_inlet.push_back(false);
				samples.push_back(right);
				normals.push_back(n);
			}


			samples.push_back(left);
			is_inlet.push_back(false);
			normals.push_back(R * vec<T, 3>{ -1, 0, 0 });
		}
	}

	for (int iz = 0; iz < sampleCountZ; ++iz)
	{
		for (int ix = 0; ix < sampleCountX; ++ix)
		{
			vec<T, 3> top{ leftCornerPos[0] + static_cast<T>(ix) * sampleDistance, leftCornerPos[1] + yDim, (leftCornerPos[2] + zDim) - static_cast<T>(iz) * sampleDistance };
			vec<T, 3> bottom{ (leftCornerPos[0] + xDim) - static_cast<T>(ix) * sampleDistance, leftCornerPos[1], leftCornerPos[2] + static_cast<T>(iz) * sampleDistance };
			top = R * top + position;
			bottom = R * bottom + position;

			samples.push_back(top);
			is_inlet.push_back(false);
			normals.push_back(R * vec<T, 3>{ 0, 1, 0 });

			samples.push_back(bottom);
			is_inlet.push_back(false);
			normals.push_back(R * vec<T, 3>{ 0, -1, 0 });

		}
	}

	velocities.resize(samples.size(), vec<T, 3>{});
	if(withInlet)
		inletVelocities.resize(inletSamples.size(), vec<T, 3>{});
}

template<typename T>
void CUBOID3D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort3D<T>(bucketSize, gridDim_L, samples, velocities);
}

//CAPSULE3D
//----------------------------------------------

template<typename T>
CAPSULE3D<T>::CAPSULE3D(T sampleDistance, vec<T, 3> position, T radius, T length, vec<T, 9> initialRot, bool withInlet) :MESH<T, 3>(sampleDistance, position, initialRot), radius(radius), length(length), withInlet(withInlet) {}

template<typename T>
void CAPSULE3D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_l = 1.0, C_u = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	if (trajectoryDefined && normilized_parameter)
	{
		this->position[0] /= dh;
		this->position[2] /= dh;
	}
	else if (trajectoryDefined)
		this->position /= dh;
	else
		this->position /= C_l;

	this->inlet_velocity /= C_u;
	this->length /= C_l;
	this->radius /= C_l;

	samples.clear();
	velocities.clear();
	normals.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	CustomVecLength<T, 3> vecLength;
	vec<T, 3> hVec;
	hVec[0] = 0.0;
	hVec[1] = radius + length / 2.0;
	hVec[2] = 0.0;
	hVec = R * hVec;
	hVec += position;
	samples.push_back(hVec);
	normals.push_back(vec<T, 3>{0, 1, 0});

	hVec[0] = 0.0;
	hVec[1] = - (radius + length / 2.0);
	hVec[2] = 0.0;
	hVec = R * hVec;
	hVec += position;
	samples.push_back(hVec);
	normals.push_back(vec<T, 3>{0, -1, 0});

	T circumferenceLat = 2.0 * PI<T> *radius;
	int  latitudeSize = static_cast<int>(circumferenceLat / (sampleDistance * 4.0));
	T dPhi = 2.0 * PI<T> *sampleDistance / circumferenceLat;
	T phi, gamma;

	for (int i = 1; i <= latitudeSize; ++i)
	{
		phi = static_cast<float>(i) * dPhi;
		T currentRadius = radius * std::cos(phi);
		T circumferenceLong = 2.0 * PI<T> *currentRadius;
		int longitudeSize = static_cast<int>(circumferenceLong / sampleDistance) + 1;
		T dGamma = 2.0 * PI<T> *sampleDistance / circumferenceLong;
		T x, y, z;

		for (int k = 0; k <= longitudeSize; ++k)
		{
			gamma = static_cast<T>(k) * dGamma;
			x = currentRadius * std::cos(gamma);
			y = radius * std::sin(phi);
			z = currentRadius * std::sin(gamma);

			hVec[0] = x;
			hVec[1] = y + length / 2.0;
			hVec[2] = z;
			hVec = R * hVec;
			hVec += position;
			samples.push_back(hVec);
			vec<T, 3> n{ x,y,z };
			n = n / vecLength.length(n);
			normals.push_back(n);


			y = -y;
			n = vec<T,3>{ x, y, z };
			n = n / vecLength.length(n);
			normals.push_back(n);
			hVec[0] = x;
			hVec[1] = y - length / 2.0;
			hVec[2] = z;
			hVec = R * hVec;
			hVec += position;
			samples.push_back(hVec);
		}
	}


	T circumference = 2.0 * PI<T> *radius;
	int  samplesSize = static_cast<int>(circumference / sampleDistance) + 1;
	int sampleSizeLength = static_cast<int>(length / sampleDistance);
	T dRad = 2.0 * PI<T> *sampleDistance / circumference;

	for (int k = 0; k <= sampleSizeLength; ++k)
		for (int i = 0; i <= samplesSize; ++i)
		{
			gamma = static_cast<T>(i) * dRad;
			hVec[0] = radius * std::cos(gamma);
			hVec[2] = radius * std::sin(gamma);
			vec<T, 3> n{ hVec[0],1,hVec[2] };
			n = n / vecLength.length(n);
			

			if (k == sampleSizeLength)
				hVec[1] = length / 2.0;
			else
				hVec[1] = static_cast<T>(k) * sampleDistance - length / 2.0;

			hVec = R * hVec;
			hVec += position;

			if (hVec[1] >= length / 6.0 && hVec[1] <= length / 3.0
				&& (gamma <= 0.5236 || gamma >= 5.7596) && withInlet)
			{
				inletSamples.push_back(hVec);
				inletNormals.push_back(n);
			}
			else
			{
				samples.push_back(hVec);
				normals.push_back(n);
			}

		}

	velocities.resize(samples.size(), vec<T, 3>{});
	if (withInlet)
		inletVelocities.resize(inletSamples.size(), vec<T, 3>{});
}

template<typename T>
void CAPSULE3D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort3D<T>(bucketSize, gridDim_L, samples, velocities);
}



#pragma region 2D-Bodies

//2D::
//----------------------------------------------

template<typename T>
MESH<T, 2>::MESH(T sampleDistance, vec<T, 2> position, const std::list<vec<int, 2>>& primitives, const std::vector<vec<T, 2>>& roughSamples, const std::vector<vec<T, 2>>& normals,
	const std::vector<char>& inletTags, vec<T, 4> initialRot)
	:IM_BODY<T, 2>(sampleDistance, position, roughSamples, normals, inletTags, initialRot), primitives(primitives) {}

template<typename T>
MESH<T, 2>::MESH(T sampleDistance, vec<T, 2> position, const std::list<vec<int, 2>>& primitives, const std::vector<vec<T, 2>>& roughSamples, vec<T, 4> initialRot)
	: IM_BODY<T, 2>(sampleDistance, position, roughSamples, {}, {}, initialRot), primitives(primitives) {}

template<typename T>
MESH<T, 2>::MESH(T sampleDistance, vec<T, 2> position, vec<T, 4> initialRot) :
	IM_BODY<T, 2>(sampleDistance, position, {}, {}, {}, initialRot) {}

template<typename T>
void MESH<T, 2>::generateSamples(double dh, double con_u, double rho)
{
	CustomVecLength<T, 2> vecLength;

	T C_l = 1.0, C_u = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	this->velocity_center /= C_u;
	this->inlet_velocity /= C_u;
	this->position /= C_l;

	bool sampleAppended = true;

	velocities.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	for (int i = 0; i < this->samples.size(); ++i)
	{
		this->samples[i] = this->R * (this->samples[i] / C_l) + this->position;

		if (!normals.empty())
			this->normals[i] = this->R * this->normals[i];
	}

	while (sampleAppended)
	{
		sampleAppended = false;

		for (auto it = primitives.begin(); it != primitives.end(); ++it)
		{
			int startIdx = (*it)[0];
			int endIdx = (*it)[1];
			T dist = vecLength.length(this->samples[endIdx] - this->samples[startIdx]) / C_l;

			if (dist > this->sampleDistance)
			{
				auto middlePoint = this->samples[startIdx] + (T)0.5 * (this->samples[endIdx] - this->samples[startIdx]);
				auto middleNormal = this->normals[startIdx] + this->normals[endIdx];
				this->samples.push_back(middlePoint);

				if (!this->is_inlet.empty())
				{
					if (this->normals.empty())
						throw std::runtime_error("ERROR::Normals for inlet not defined");

					middleNormal /= vecLength.length(middleNormal);
					middleNormal /= vecLength.length(middleNormal);
					this->normals.push_back(middleNormal);
					auto middleTag = this->is_inlet[startIdx] && this->is_inlet[endIdx] ? true : false;
					this->is_inlet.push_back(middleTag);
				}


				auto lastIdx = static_cast<int>(samples.size() - 1);
				vec<int, 2> firstPrimitive{ startIdx, lastIdx };
				vec<int, 2> secondPrimitive{ lastIdx, endIdx };
				primitives.insert(it, firstPrimitive);
				primitives.insert(it, secondPrimitive);

				primitives.erase(it--);

				sampleAppended = true;
			}
		}
	}

	std::vector<vec<T, 2>> tempSamp = this->samples;
	std::vector<vec<T, 2>> tempNorm = this->normals;

	this->samples.clear();
	this->normals.clear();


	for (int i = 0; i < tempSamp.size(); ++i)
	{
		if (!this->is_inlet.empty() && this->is_inlet[i])
		{
			this->inletSamples.push_back(tempSamp[i]);
			this->inletVelocities.push_back({ this->velocity_center + dot(this->inlet_velocity, tempNorm[i]) * tempNorm[i] });
			this->inletNormals.push_back(tempNorm[i]);
		}
		else
		{
			this->samples.push_back(tempSamp[i]);
			this->velocities.push_back(this->velocity_center);
			this->normals.push_back(tempNorm[i]);
		}
	}
}

template<typename T>
void MESH<T, 2>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort2D<T>(bucketSize, gridDim_L, samples, velocities);
}


template<typename T>
SPHERE2D<T>::SPHERE2D(T sampleDistance, vec<T, 2> position, T radius, vec<T, 4> initialRot) :MESH<T, 2>(sampleDistance, position, initialRot), radius(radius) {}

template<typename T>
void SPHERE2D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_l = 1.0, C_u = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	this->velocity_center /= C_u;
	this->inlet_velocity /= C_u;
	this->position /= C_l;
	this->radius /= C_l;

	samples.clear();
	velocities.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	T circumference = 2.0 * PI<T> * radius;
	int  samplesSize = static_cast<int>(circumference / sampleDistance);
	T dRad = 2.0 * PI<T> * sampleDistance / circumference;
	T gamma = 0.0;

	auto a = cs_sq<float>;

	for (int i = 0; i <= samplesSize; ++i)
	{
		vec<T, 2> hVec;
		hVec[0] = radius * std::cos(gamma);
		hVec[1] = radius * std::sin(gamma);
		hVec = R*hVec;
		hVec += position;
		samples.push_back(hVec);
		gamma += dRad;

		velocities.push_back(this->velocity_center);
	}

}

template<typename T>
void SPHERE2D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort2D<T>(bucketSize, gridDim_L, samples, velocities);
}


//CUBOID2D
//----------------------------------------------

template<typename T>
CUBOID2D<T>::CUBOID2D(T sampleDistance, vec<T, 2> position, T xDim, T yDim, vec<T, 4> initialRot) :MESH<T, 2>(sampleDistance, position, initialRot), xDim(xDim), yDim(yDim) {}

template<typename T>
void CUBOID2D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_l = 1.0, C_u = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}


	this->velocity_center /= C_u;
	this->inlet_velocity /= C_u;
	this->position /= C_l;
	this->xDim /= C_l;
	this->yDim /= C_l;

	this->inlet_velocity = R * this->inlet_velocity;

	samples.clear();
	velocities.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	vec<T, 2> leftCornerPos{};

	leftCornerPos[0] -= xDim / 2.0;
	leftCornerPos[1] -= yDim / 2.0;


	int sampleCountX = static_cast<int>(xDim / sampleDistance);
	int sampleCountY = static_cast<int>(yDim / sampleDistance);

	for (int ix = 0; ix < sampleCountX; ++ix)
	{
		vec<T, 2> bottom{ leftCornerPos[0] + static_cast<T>(ix) * sampleDistance, leftCornerPos[1] };
		vec<T, 2> top{ (leftCornerPos[0] + xDim) - static_cast<T>(ix) * sampleDistance, leftCornerPos[1] + yDim };
		bottom = R * bottom + position;
		top = R * top + position;

		is_inlet.push_back(false);
		samples.push_back(bottom);
		normals.push_back(R * vec<T,2>{ 0,-1 });

		is_inlet.push_back(false);
		samples.push_back(top);
		normals.push_back(R * vec<T, 2>{ 0,1 });

		velocities.push_back(this->velocity_center);
		velocities.push_back(this->velocity_center);
	}

	for (int iy = 1; iy < sampleCountY-1; ++iy)
	{
		vec<T, 2> left{ leftCornerPos[0], leftCornerPos[1] + yDim - static_cast<T>(iy) * sampleDistance };
		vec<T, 2> right{ leftCornerPos[0] + xDim, leftCornerPos[1] + static_cast<T>(iy) * sampleDistance };
		left = R * left + position;
		right = R * right + position;
		

		is_inlet.push_back(false);
		samples.push_back(left);
		vec<T, 2> n = R * vec<T, 2>{ -1, 0 };
		normals.push_back(n);

		is_inlet.push_back(true);
		inletSamples.push_back(right);
		inletNormals.push_back(R * vec<T, 2>{ 1,0 });

		velocities.push_back(this->velocity_center);
		inletVelocities.push_back(this->velocity_center + dot(n,this->inlet_velocity)*n);
	}
}

template<typename T>
void CUBOID2D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort2D<T>(bucketSize, gridDim_L, samples, velocities);
}

//CAPSULE2D
//----------------------------------------------

template<typename T>
CAPSULE2D<T>::CAPSULE2D(T sampleDistance, vec<T, 2> position, T radius, T length, vec<T, 4> initialRot) :MESH<T, 2>(sampleDistance, position, initialRot), radius(radius), length(length) {}

template<typename T>
void CAPSULE2D<T>::generateSamples(double dh, double con_u, double rho)
{
	T C_u = 1.0, C_l = 1.0;
	if (!normilized_parameter)
	{
		C_l = dh;
		C_u = con_u;
	}

	this->velocity_center /= C_u;
	this->inlet_velocity /= C_u;
	this->position /= C_l;
	this->radius /= C_l;
	this->length /= C_l;

	samples.clear();
	velocities.clear();
	inletSamples.clear();
	inletVelocities.clear();
	inletNormals.clear();

	int  circlSamplesSize = static_cast<int>(PI<T> * radius / sampleDistance) + 1;
	int cylinderSampleSize = static_cast<int>(length / sampleDistance);

	T circumference = 2.0 * PI<T> * radius;
	T dRad = 2.0 * PI<T> * sampleDistance / circumference;
	T gamma = 0.0;


	for (int i = 0; i <= circlSamplesSize; ++i)
	{
		gamma = static_cast<T>(i) * dRad;

		vec<T,2> hVec;
		T yCoord = length / 2.0;

		hVec[0] = radius * std::cos(gamma);
		hVec[1] = yCoord + radius * std::sin(gamma);
		hVec = R * hVec;
		hVec += position;

		samples.push_back(hVec / C_l);
		velocities.push_back(this->velocity_center);

		gamma = PI<T> + static_cast<T>(i) * dRad;

		if (gamma < 2.0 * PI<T> && i == circlSamplesSize)
			gamma = 2.0 * PI<T>;

		hVec[0] = radius * std::cos(gamma);
		hVec[1] = -yCoord + radius * std::sin(gamma);
		hVec = R * hVec;
		hVec += position;

		samples.push_back(hVec);
		velocities.push_back(this->velocity_center);
	}

	for (int i = 1; i <= cylinderSampleSize; ++i)
	{
		vec<T, 2> hVec{ radius, static_cast<T>(i) * sampleDistance - length / (T)2.0 };
		hVec = R * hVec;
		hVec += position;

		samples.push_back(hVec);
		velocities.push_back(this->velocity_center);

		hVec[0] = -radius;
		hVec[1] = static_cast<float>(i) * sampleDistance - length / 2.0;

		hVec = R*hVec;
		hVec += position;

		samples.push_back(hVec);
		velocities.push_back(this->velocity_center);
	}
}

template<typename T>
void CAPSULE2D<T>::mortonOrderSort(vec<grid_size_t, 3> gridDim_L, unsigned int bucketSize)
{
	mortonOrderSort2D<T>(bucketSize, gridDim_L, samples, velocities);
}

#pragma endregion