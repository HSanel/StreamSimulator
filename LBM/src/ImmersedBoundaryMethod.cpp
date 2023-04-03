#include "ImmersedBoundaryMethod.h"

template struct IBMethod_Specialisation<float, 2>;
template struct IBMethod_Specialisation<double, 2>;
template struct IBMethod_Specialisation<float, 3>;
template struct IBMethod_Specialisation<double, 3>;

template struct IBMethod<float,2>;
template struct IBMethod<float, 3>;
template struct IBMethod<double, 2>;
template struct IBMethod<double, 3>;

IBMethod_Base::IBMethod_Base(const unsigned char ibm_tag):ibm_tag(ibm_tag){}

//Specialisation
template <typename T>
IBMethod_Specialisation<T, 2>::IBMethod_Specialisation(const unsigned char ibm_tag):IBMethod_Base(ibm_tag){}

template <typename T>
IBMethod_Specialisation<T, 3>::IBMethod_Specialisation(const unsigned char ibm_tag) : IBMethod_Base(ibm_tag) {}


template <typename T>
void IBMethod_Specialisation<T,2>::calcFs(SimDomain<T,2> &sd, vec<T,2> * u_L, T* rho_L, std::vector<Fs_DATA<T, 2>> &Fs_dataList)
{
	std::vector<vec<T, 2>> normals;
	std::vector<vec<T, 2>> samples;
	std::vector<vec<T, 2>> vel;
	int kernelSize;

	if (sharpness == SHARP)
		kernelSize = 3;
	else if (sharpness == MEDIUM_SMOOTH || sharpness == SMOOTH)
		kernelSize = 5;

	auto calc_uf_lag = [&](vec<T, 2> sample)
	{
		int x{}, y{};
		vec<T, 2> uf_lag{};


		for (int yk = 0; yk < kernelSize; ++yk)
		{
			for (int xk = 0; xk < kernelSize; ++xk)
			{
				int r = kernelSize / 2;
				x = std::round(sample[0]) - r + xk;
				y = std::round(sample[1]) - r + yk;
				vec<T, 2> node{(T)x, (T)y};

				if (x >= 0 && x < sd.getGridDim_L(0) && y >= 0 && y < sd.getGridDim_L(1))
				{
					T delta = deltaFunc(vec<T, 2>{(T)x, (T)y}, sample, sharpness);
					uf_lag += u_L[y * sd.getGridDim_L(0) + x] * delta;
				}
			}
		}

		return uf_lag;
	};


	auto calc_rho_lag = [&](vec<T, 2> sample)
	{
		int x{}, y{};
		T rho_lag = 0.0;

		for (int yk = 0; yk < kernelSize; ++yk)
		{
			for (int xk = 0; xk < kernelSize; ++xk)
			{
				int r = kernelSize / 2;
				x = std::round(sample[0]) - r + xk;
				y = std::round(sample[1]) - r + yk;
				vec<T, 2> node{(T) x,(T) y };

				if (x >= 0 && x < sd.getGridDim_L(0) && y >= 0 && y < sd.getGridDim_L(1))
				{
					T delta = deltaFunc(vec<T, 2>{(T)x, (T)y}, sample, sharpness);
					rho_lag += rho_L[y * sd.getGridDim_L(0) + x] * delta;
					
				}
			}
		}

		return rho_lag;
	};

	for (int n = 0; n < im_list.size(); ++n)
	{
		const IM_BODY<T, 2>* body = im_list.at(n).get();
		samples = body->samples;
		vel = body->velocities;
		normals = body->normals;

		for (int i = 0; i < samples.size(); ++i)
		{
			int x = std::round(samples.at(i)[0]);
			int y = std::round(samples.at(i)[1]);
			int x_loc, y_loc;

			if (x < 0 || x >= sd.getGridDim_L(0) || y < 0 || y >= sd.getGridDim_L(1))
				continue;

			vec<T, 2> uf_lag = calc_uf_lag(samples[i]);
			T rho_lag = calc_rho_lag(samples[i]);

			vec<T, 2> Fs_lag = rho_lag * (vel[i] - uf_lag);

			for (int yk = 0; yk < kernelSize; ++yk)
			{
				for (int xk = 0; xk < kernelSize; ++xk)
				{
					int r = kernelSize / 2;
					x_loc = x - r + xk;
					y_loc = y - r + yk;

					if (x_loc >= 0 && x_loc < sd.getGridDim_L(0) && y_loc >= 0 && y_loc < sd.getGridDim_L(1))
					{

						int pos_L = y_loc * sd.getGridDim_L(0) + x_loc;

						T delta = deltaFunc(samples[i], { (T)x_loc, (T)y_loc }, sharpness);
						vec<T, 2> Fs_eul_loc = Fs_lag * delta;
						
						vec<T, 2> node{ (T)x_loc,(T)y_loc };

						if (sum(Fs_eul_loc * Fs_eul_loc) > 0)
							Fs_dataList.push_back(Fs_DATA<T, 2>{ pos_L, Fs_eul_loc });

					}
				}
			}
		}
	}
}

template <typename T>
void IBMethod_Specialisation<T,3>::calcFs(SimDomain<T, 3>& sd, vec<T,3>* u_L, T* rho_L, std::vector<Fs_DATA<T, 3>>& Fs_dataList)
{
	std::vector<vec<T, 3>> normals;
	std::vector<vec<T, 3>> samples;
	std::vector<vec<T, 3>> vel;
	int kernelSize;
	CustomVecLength<T, 3> vecLength;

	if (sharpness == SHARP)
		kernelSize = 3;
	else if (sharpness == MEDIUM_SMOOTH || sharpness == SMOOTH)
		kernelSize = 5;

	auto calc_uf_lag = [&](vec<T, 3> sample)
	{
		int x{}, y{}, z{};
		vec<T, 3> uf_lag{};


		for (int zk = 0; zk < kernelSize; ++zk)
		{
			for (int yk = 0; yk < kernelSize; ++yk)
			{
				for (int xk = 0; xk < kernelSize; ++xk)
				{
					int r = kernelSize / 2;
					x = std::round(sample[0]) - r + xk;
					y = std::round(sample[1]) - r + yk;
					z = std::round(sample[2]) - r + zk;

					if (x >= 0 && x < sd.getGridDim_L(0) && y >= 0 && y < sd.getGridDim_L(1) && z >= 0 && z < sd.getGridDim_L(2))
					{
						T delta = deltaFunc(vec<T, 3>{(T)x, (T)y, (T)z}, sample, sharpness);
						uf_lag += u_L[(z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x] * delta;
					}
				}
			}
		}

		return uf_lag;
	};

	auto calc_rho_lag = [&](vec<T, 3> sample)
	{
		int x{}, y{}, z{};
		T rho_lag = 0.0;

		for (int zk = 0; zk < kernelSize; ++zk)
		{
			for (int yk = 0; yk < kernelSize; ++yk)
			{
				for (int xk = 0; xk < kernelSize; ++xk)
				{
					int r = kernelSize / 2;
					x = std::round(sample[0]) - r + xk;
					y = std::round(sample[1]) - r + yk;
					z = std::round(sample[2]) - r + zk;

					if (x >= 0 && x < sd.getGridDim_L(0) && y >= 0 && y < sd.getGridDim_L(1) && z >= 0 && z < sd.getGridDim_L(2))
					{
						T delta = deltaFunc(vec<T, 3>{(T)x, (T)y, (T)z}, sample, sharpness);
						rho_lag += rho_L[(z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x] * delta;
					}
				}
			}
		}

		return rho_lag;
	};

	for (int n = 0; n < im_list.size(); ++n)
	{
		IM_BODY<T, 3>* body = im_list.at(n).get();
		samples = body->samples;
		vel = body->velocities;
		normals = body->normals;

		for (int i = 0; i < samples.size(); ++i)
		{
			int x = std::round(samples.at(i)[0]);
			int y = std::round(samples.at(i)[1]);
			int z = std::round(samples.at(i)[2]);
			int x_loc, y_loc, z_loc;

			if (x < 0 || x >= sd.getGridDim_L(0) || y < 0 || y >= sd.getGridDim_L(1) || z < 0 || z >= sd.getGridDim_L(2))
				continue;

			vec<T, 3> uf_lag = calc_uf_lag(samples.at(i));
			T rho_lag = calc_rho_lag(samples.at(i));
			vec<T, 3> Fs_lag = rho_lag * (vel.at(i) - uf_lag);

			for (int zk = 0; zk < kernelSize; ++zk)
			{
				for (int yk = 0; yk < kernelSize; ++yk)
				{
					for (int xk = 0; xk < kernelSize; ++xk)
					{
						int r = kernelSize / 2;
						x_loc = x - r + xk;
						y_loc = y - r + yk;
						z_loc = z - r + zk;

						if (x_loc >= 0 && x_loc < sd.getGridDim_L(0) && y_loc >= 0 && y_loc < sd.getGridDim_L(1) && z_loc >= 0 && z_loc < sd.getGridDim_L(2))
						{
							int pos_L = (z_loc * sd.getGridDim_L(1) + y_loc) * sd.getGridDim_L(0) + x_loc;
							T delta = deltaFunc(samples.at(i), { (T)x_loc, (T)y_loc, (T)z_loc }, sharpness);
							vec<T, 3> Fs_eul_loc = Fs_lag * delta;

							if (vecLength.length(Fs_eul_loc) > 0.f)
								Fs_dataList.push_back(Fs_DATA<T, 3>{ pos_L, Fs_eul_loc });
						}
					}
				}
			}
		}
	}
}


template <typename T>
void IBMethod_Specialisation<T, 2>::calcInlet(SimDomain<T, 2>& sd, vec<InletGhostCells<T, 2>, 9>* sourceNodes)
{
	std::vector<vec<T, 2>> normals;
	std::vector<vec<T, 2>> samples;
	std::vector<vec<T, 2>> vel;

	for (int n = 0; n < im_list.size(); ++n)
	{
		const IM_BODY<T, 2>* body = im_list.at(n).get();
		samples = body->inletSamples;
		vel = body->inletVelocities;
		normals = body->inletNormals;

		for (int i = 0; i < samples.size(); ++i)
		{
			int x = std::round(samples.at(i)[0]);
			int y = std::round(samples.at(i)[1]);
			int x_loc, y_loc;

			if (x < 0 || x >= sd.getGridDim_L(0) || y < 0 || y >= sd.getGridDim_L(1))
				continue;


			vec<T, 2> node_S{ (T)x,(T)y };

			if (dot(normals[i], (node_S - samples[i])) > 0)
				node_S -= normals[i];

			node_S[0] = round(node_S[0]);
			node_S[1] = round(node_S[1]);


			for (int k = 1; k < 9; ++k)
			{
				int x_B = round(node_S[0] + sd.c(k)[0]);
				int y_B = round(node_S[1] + sd.c(k)[1]);
				vec<T, 2> node_B{ (T)x_B, (T)y_B };

				if (dot(normals[i], (node_B - samples[i])) > 0)
				{
					if (dot(normals[i], (node_B - sd.c(k) - samples[i])) <= 0)
					{
						node_S[0] = round(node_B[0] - sd.c(k)[0]);
						node_S[1] = round(node_B[1] - sd.c(k)[1]);

						int node_F0 = round((T)x_B + sd.c(k)[0]);
						int node_F1 = round((T)y_B + sd.c(k)[1]);

						int pos_solid = node_S[1] * sd.getGridDim_L(0) + node_S[0];
						int pos_bound = y_B * sd.getGridDim_L(0) + x_B;
						int pos_fluid = node_F1 * sd.getGridDim_L(0) + node_F0;

						T denominator = dot(normals[i], sd.c(k));
						if (abs(denominator) > 0)
						{
							T q = abs(dot(normals[i], node_B - samples[i]) / denominator);
							sourceNodes[pos_solid][k].pos_bound = pos_bound;
							sourceNodes[pos_solid][k].pos_fluid = pos_fluid;
							sourceNodes[pos_solid][k].q = q;
							sourceNodes[pos_solid][k].velocity = vel[i];
							sourceNodes[pos_solid][k].isInlet = true;
						}
					}
				}


			}
		}
	}
}

template <typename T>
void IBMethod_Specialisation<T, 3>::calcInlet(SimDomain<T, 3>& sd, vec<InletGhostCells<T, 3>, 27>* sourceNodes)
{
	std::vector<vec<T, 3>> normals;
	std::vector<vec<T, 3>> samples;
	std::vector<vec<T, 3>> vel;

	for (int n = 0; n < im_list.size(); ++n)
	{
		IM_BODY<T, 3>* body = im_list.at(n).get();
		samples = body->inletSamples;
		vel = body->inletVelocities;
		normals = body->inletNormals;

		for (int i = 0; i < samples.size(); ++i)
		{
			int x = std::round(samples.at(i)[0]);
			int y = std::round(samples.at(i)[1]);
			int z = std::round(samples.at(i)[2]);
			int x_loc, y_loc, z_loc;

			if (x < 0 || x >= sd.getGridDim_L(0) || y < 0 || y >= sd.getGridDim_L(1) || z < 0 || z >= sd.getGridDim_L(2))
				continue;

			vec<T, 3> node_S{ (T)x,(T)y, (T)z };

			if (dot(normals[i], (node_S - samples[i])) > 0)
				node_S -= normals[i];

			node_S[0] = round(node_S[0]);
			node_S[1] = round(node_S[1]);
			node_S[2] = round(node_S[2]);


			for (int k = 1; k < 27; ++k)
			{
				int x_B = round(node_S[0] + sd.c(k)[0]);
				int y_B = round(node_S[1] + sd.c(k)[1]);
				int z_B = round(node_S[2] + sd.c(k)[2]);
				vec<T, 3> node_B{ (T)x_B, (T)y_B, (T)z_B };

				if (dot(normals[i], (node_B - samples[i])) > 0)
				{
					if (dot(normals[i], (node_B - sd.c(k) - samples[i])) <= 0)
					{
						node_S[0] = round(node_B[0] - sd.c(k)[0]);
						node_S[1] = round(node_B[1] - sd.c(k)[1]);
						node_S[2] = round(node_B[2] - sd.c(k)[2]);

						int node_F0 = round((T)x_B + sd.c(k)[0]);
						int node_F1 = round((T)y_B + sd.c(k)[1]);
						int node_F2 = round((T)z_B + sd.c(k)[2]);

						int pos_Solid = (node_S[2] * sd.getGridDim_L(1) + node_S[1]) * sd.getGridDim_L(0) + node_S[0];
						int pos_bound = (z_B * sd.getGridDim_L(1) + y_B) * sd.getGridDim_L(0) + x_B;
						int pos_fluid = (node_F2 * sd.getGridDim_L(1) + node_F1) * sd.getGridDim_L(0) + node_F0;

						T denominator = dot(normals[i], sd.c(k));
						if (abs(denominator) > 0)
						{
							T q = abs(dot(normals[i], node_B - samples[i]) / denominator);
							sourceNodes[pos_Solid][k].pos_bound = pos_bound;
							sourceNodes[pos_Solid][k].pos_fluid = pos_fluid;
							sourceNodes[pos_Solid][k].q = q;
							sourceNodes[pos_Solid][k].velocity = vel[i];
							sourceNodes[pos_Solid][k].isInlet = true;
						}
					}
				}
			}

		}
		
	}
}

template <typename T>
vec<T, 4> IBMethod_Specialisation<T, 2>::getCrossProductMat(vec<T, 3> v)
{
	return { 0.0, -v[2],
			v[2],  0.0 };
}

template <typename T>
vec<T, 9> IBMethod_Specialisation<T, 3>::getCrossProductMat(vec<T, 3> v)
{
	return { 0.0, -v[2],  v[1],
			v[2],  0.0,  -v[0],
			-v[1], v[0],  0.0 };
}
//---------------------------------
template<typename T, size_t D>
IBMethod<T, D>::IBMethod(const unsigned char ibm_tag) : IBMethod_Specialisation<T, D>(ibm_tag){}

////Base Methods:
template <typename T, size_t D>
void IBMethod<T, D>::addBody(std::unique_ptr<IM_BODY<T, D>> im_body) { im_list.push_back(std::move(im_body)); }

template <typename T, size_t D>
int IBMethod<T, D>::getBodyCount() const { return im_list.size(); }

template <typename T, size_t D>
const IM_BODY<T, D>* IBMethod<T, D>::getBodyAt(int idx) const { return im_list[idx].get(); }

//Base Methods:
template <typename T, size_t D>
const unsigned char IBMethod<T, D>::getTag() const
{
	return ibm_tag;
}

template <typename T, size_t D>
void IBMethod<T,D>::reset(T dh, T C_u, T rho)
{
	for (auto& body : im_list)
	{
		try
		{
			executeGenerator(*body, (double)dh, (double)C_u, (double)rho);
			maxPointCount += body->samples.size() + body->inletSamples.size();
		}
		catch (std::exception e)
		{
			std::cerr << "ERROR::RESET IMMERSED BOUNDARY:: " << e.what() << std::endl;
		}
	}
}

template <typename T, size_t D>
void IBMethod<T,D>::update(T timeStep)
{
	for (int i = 0; i < im_list.size(); ++i)
	{
		auto& body = im_list[i];
		/*for (int i = 0; i < body->samples.size(); ++i)
		{
			body->samples[i] = body->R * (body->samples[i] - oldPos) + body->position;
			body->velocities[i] = omega_star * (body->samples[i] - body->position) + body->velocity_center;

			if (!body->is_inlet.empty() && body->is_inlet[i])
			{
				body->inlet_velocity = body->R * body->inlet_velocity;
				body->velocities[i] += dot(body->inlet_velocity, body->normals[i])*body->normals[i];
			}
		}*/
	}
}

template <typename T, size_t D>
int IBMethod<T, D>::getMaxPointCount() const { return maxPointCount; }

template <typename T, size_t D>
void IBMethod<T, D>::setSharpness(unsigned char sharpness) { this->sharpness = sharpness; }

template<typename T, size_t D>
void IBMethod<T, D>::rescaleVelocities(const T& scale_u)
{
	for (int i = 0; i < im_list.size(); ++i)
	{
		im_list[i]->inlet_velocity *= scale_u;
		im_list[i]->velocity_center *= scale_u;
		for(int ID=0; ID < im_list[i]->velocities.size(); ++ID)
			im_list[i]->velocities[ID] *= scale_u;
	}
}