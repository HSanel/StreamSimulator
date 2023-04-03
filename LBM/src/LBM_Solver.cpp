#include "LBM_Solver.h"

template struct LBM_Solver_Specialisation<float, 2>;
template struct LBM_Solver_Specialisation<float, 3>;
template struct LBM_Solver_Specialisation<double, 2>;
template struct LBM_Solver_Specialisation<double, 3>;

template struct LBM_Solver<float,2>;
template struct LBM_Solver<float, 3>;
template struct LBM_Solver<double, 2>;
template struct LBM_Solver<double, 3>;

//for DEBUG
void printBIN(unsigned char a)
{
	unsigned char o = 1;
	std::stringstream s;
	
	for (int i = 0; i < sizeof(unsigned char)*8; i++)
	{
		unsigned char b = a & o;
		if (b != 0)
			s << 1;
		else
			s << 0;

		o = o << 1;
	}
	std::string rev_s(s.str());
	std::reverse(rev_s.begin(), rev_s.end());
	std::cout << rev_s << std::endl;
}

//specialised constructors
template<typename T>
LBM_Solver_Specialisation<T, 2>::LBM_Solver_Specialisation(SimDomain<T, 2> simDom, std::unique_ptr<IBMethod<T, 2>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 2>> writer)
	:sd(simDom), ibm(std::move(immersedBoundary)), writer(std::move(writer)), st(SimState<T, 2>{simDom, immersedBoundary != nullptr}) {}

template<typename T>
LBM_Solver_Specialisation<T, 3>::LBM_Solver_Specialisation(SimDomain<T, 3> simDom, std::unique_ptr<IBMethod<T, 3>> immersedBoundary, std::unique_ptr<LBM_Writer<T, 3>> writer)
	: sd(simDom), ibm(move(immersedBoundary)), writer(move(writer)), st(SimState<T, 3>{simDom, immersedBoundary != nullptr}) {}

//base constructors
template<typename T, size_t D>
LBM_Solver<T, D>::LBM_Solver(SimDomain<T, D> simDom, std::unique_ptr<IBMethod<T, D>> immersedBoundary, std::unique_ptr <LBM_Writer<T,D>> writer, double maxSimulationTime, bool unitTime , SimInitialiser<T, D> simInit)
	: LBM_Solver_Specialisation<T, D>(simDom, std::move(immersedBoundary), std::move(writer)), unitTime(unitTime)
{
	if (unitTime)
		this->maxSimulationTime = maxSimulationTime * sd.getTimeStep();
	else
		this->maxSimulationTime = maxSimulationTime;
	
	simulationStep = 0;

	if (!normilized_parameter)
	{
		simInit.rho /= simDom.getC_u();
		simInit.u  /= simDom.getC_u();
		simInit.F_ext /= simDom.getC_F();
	}

	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
	{
		st.rho_L[pos] = simInit.rho;
		st.u_L[pos] = simInit.u;
		st.F_ext_L[pos] = simInit.F_ext;

		for (int i = 0; i < sd.getQ(); ++i)
		{
			st.f_star[pos][i] = sd.w(i) * st.rho_L[pos] * ((T)1.0 + dot(sd.c(i), st.u_L[pos]) / cs_sq<T> 
				+dot(sd.c(i), st.u_L[pos]) * dot(sd.c(i), st.u_L[pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(st.u_L[pos] , st.u_L[pos]) / ((T)2.0 * cs_sq<T>));
		}
	}
}

template<typename T, size_t D>
LBM_Solver<T, D>::LBM_Solver(SimDomain<T, D> simDom, std::unique_ptr<LBM_Writer<T, D>> writer, double maxSimulationTime, bool unitTime, SimInitialiser<T, D> simInit)
	:LBM_Solver(simDom, nullptr, std::move(writer), maxSimulationTime, unitTime, simInit){}

template<typename T, size_t D>
LBM_Solver<T, D>::LBM_Solver(SimDomain<T, D> simDom, std::unique_ptr<IBMethod<T, D>> immersedBoundary, double maxSimulationTime, bool unitTime, SimInitialiser<T, D> simInit)
	: LBM_Solver(simDom, std::move(immersedBoundary),nullptr, maxSimulationTime, unitTime, simInit)  {}

template<typename T, size_t D>
LBM_Solver<T, D>::LBM_Solver(SimDomain<T, D> simDom, double maxSimulationTime, bool unitTime, SimInitialiser<T, D> simInit)
	: LBM_Solver(simDom, nullptr, nullptr, maxSimulationTime, unitTime, simInit){}

//specialised methods
template<typename T>
void LBM_Solver_Specialisation<T, 2>::streaming()
{
	for (int y = 0; y < sd.getGridDim_L(1); ++y)
		for (int x = 0; x < sd.getGridDim_L(0); ++x)
		{
			int pos = y * sd.getGridDim_L(0) + x;
			T rhoVB_border = st.rho_L[pos];
			T rhoPB_border = 1.0;
			vec<T, 2> uwVB{ 0.0, 0.0 }, uwPB{ 0.0, 0.0 };
			bool velBound_defined = false;
			bool pressBound_defined = false;

			int i;
			auto calcPos = [&](int idx) {	return (y + static_cast<int>(sd.c(idx)[1])) * sd.getGridDim_L(0) + x + static_cast<int>(sd.c(idx)[0]); };

			auto calc_f = [&](int idx) { return st.f_star[pos][idx]; };
			auto calc_f_with_velBound = [&](int idx) { return st.f_star[pos][idx] - 2.0 * sd.w(idx) * (rhoVB_border / cs_sq<T>) * (sd.c(idx)[0] * uwVB[0] + sd.c(idx)[1] * uwVB[1]); };
			auto calc_f_with_pressBound = [&](int idx)
			{
				T scalarProdSquared = (sd.c(idx)[0] * uwPB[0] + sd.c(idx)[1] * uwPB[1]) * (sd.c(idx)[0] * uwPB[0] + sd.c(idx)[1] * uwPB[1]);
				return -st.f_star[pos][idx] + 2.0 * sd.w(idx) * rhoPB_border * (1.0 + scalarProdSquared / (2.0 * cs_sq<T> *cs_sq<T>) - (uwPB[0] * uwPB[0] + uwPB[1] * uwPB[1]) * (uwPB[0] * uwPB[0] + uwPB[1] * uwPB[1]) / (2.0 * cs_sq<T>));
			};
			//Rand Auswertung ändern!!!
			for (int vbIdx = 0; vbIdx < sd.getVelBoundCount(); ++vbIdx)
			{
				auto& velBound = sd.getVelBound(vbIdx);

				if (x == 0 && velBound.side == left)	//left
				{
					if (y < sd.getGridDim_L(1))
					{
						velBound_defined = true;
						uwVB[0] = velBound.u_w/sd.getC_u();
						uwVB[1] = 0.0;
						break;
					}
				}
				else if (x == sd.getGridDim_L(0) - 1 && velBound.side == right)	//right
				{
					if (y < sd.getGridDim_L(1))
					{
						velBound_defined = true;
						uwVB[0] = -velBound.u_w / sd.getC_u();
						uwVB[1] = 0.0;
						break;
					}
				}
				else if (y == 0 && velBound.side == bottom)	//bottom
				{
					if (x < sd.getGridDim_L(0))
					{
						velBound_defined = true;
						uwVB[0] = 0.0;
						uwVB[1] = velBound.u_w / sd.getC_u();
						break;
					}
				}
				else if (y == sd.getGridDim_L(1) - 1 && velBound.side == top)	//top
				{
					if (x < sd.getGridDim_L(0))
					{
						velBound_defined = true;
						uwVB[0] = 0.0;
						uwVB[1] = -velBound.u_w / sd.getC_u();
						break;
					}
				}
			}

			for (int pbIdx = 0; pbIdx < sd.getPressBoundCount(); ++pbIdx)
			{
				auto& pressBound = sd.getPressBound(pbIdx);

				if (x == 0 && pressBound.side == left)	//left
				{
					if (y < sd.getGridDim_L(1))
					{
						pressBound_defined = true;
						int pos_next = y * sd.getGridDim_L(0) + x + 1;

						uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
						rhoPB_border = (pressBound.dp_w/sd.getC_p()) / cs_sq<T> +1.0;
						break;
					}
				}
				else if (x == sd.getGridDim_L(0) - 1 && pressBound.side == right)	//right
				{
					if (y < sd.getGridDim_L(1))
					{
						pressBound_defined = true;
						int pos_next = y * sd.getGridDim_L(0) + x - 1;

						uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
						rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
						break;
					}
				}
				else if (y == 0 && pressBound.side == bottom)	//bottom
				{
					if (x < sd.getGridDim_L(0))
					{
						pressBound_defined = true;
						int pos_next = (y + 1) * sd.getGridDim_L(0) + x;

						uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
						rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> + 1.0;
						break;
					}
				}
				else if (y == sd.getGridDim_L(1) - 1 && pressBound.side == top)	//top
				{
					if (x < sd.getGridDim_L(0))
					{
						pressBound_defined = true;
						int pos_next = (y - 1) * sd.getGridDim_L(0) + x;

						uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
						rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
						break;
					}
				}
			}



			if (velBound_defined && pressBound_defined)
				throw std::invalid_argument("ERROR:: Inlett and Outlett can't be defined at the same latticenode::");

			i = 0;					
			st.f[calcPos(i)][i] = calc_f(i);

			i = 1;
			if (x == sd.getGridDim_L(0) - 1)
			{
				if (velBound_defined)
					st.f[pos][3] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][3] = calc_f_with_pressBound(i);
				else
					st.f[pos][3] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 2;
			if (y == sd.getGridDim_L(1) - 1)
			{
				if (velBound_defined)
					st.f[pos][4] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][4] = calc_f_with_pressBound(i);
				else
					st.f[pos][4] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 3;				//L
			if (x == 0)
			{
				if (velBound_defined)
					st.f[pos][1] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][1] = calc_f_with_pressBound(i);
				else
					st.f[pos][1] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 4;
			if (y == 0)
			{
				if (velBound_defined)
					st.f[pos][2] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][2] = calc_f_with_pressBound(i);
				else
					st.f[pos][2] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 5;
			if (x == sd.getGridDim_L(0) - 1 || y == sd.getGridDim_L(1) - 1)
			{
				if (velBound_defined)
					st.f[pos][7] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][7] = calc_f_with_pressBound(i);
				else
					st.f[pos][7] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 6;
			if (x == 0 || y == sd.getGridDim_L(1) - 1)
			{
				if (velBound_defined)
					st.f[pos][8] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][8] = calc_f_with_pressBound(i);
				else
					st.f[pos][8] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 7;						//L
			if (x == 0 || y == 0)
			{
				if (velBound_defined)
					st.f[pos][5] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][5] = calc_f_with_pressBound(i);
				else
					st.f[pos][5] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);

			i = 8;
			if (x == sd.getGridDim_L(0) - 1 || y == 0)
			{
				if (velBound_defined)
					st.f[pos][6] = calc_f_with_velBound(i);
				else if (pressBound_defined)
					st.f[pos][6] = calc_f_with_pressBound(i);
				else
					st.f[pos][6] = calc_f(i);
			}
			else
				st.f[calcPos(i)][i] = calc_f(i);
		}
}

template<typename T>
void LBM_Solver_Specialisation<T, 3>::streaming()
{
	for (int z = 0; z < sd.getGridDim_L(2); ++z)
		for (int y = 0; y < sd.getGridDim_L(1); ++y)
			for (int x = 0; x < sd.getGridDim_L(0); ++x)
			{
				int pos = (z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x;
				T rhoVB_border = st.rho_L[pos];
				T rhoPB_border = 1.0;
				vec<T, 3> uwVB{ 0.0, 0.0, 0.0 }, uwPB{ 0.0, 0.0, 0.0 };
				bool velBound_defined = false;
				bool pressBound_defined = false;

				int i;
				auto calcPos = [&](int idx) {	return ((z + static_cast<int>(sd.c(idx)[2])) * sd.getGridDim_L(1) + y + static_cast<int>(sd.c(idx)[1])) * sd.getGridDim_L(0) + x + static_cast<int>(sd.c(idx)[0]); };
				auto calc_f = [&](int idx) { return st.f_star[pos][idx]; };
				auto calc_f_with_velBound = [&](int idx) { return st.f_star[pos][idx] - 2.0 * sd.w(idx) * (rhoVB_border / cs_sq<T>) * (sd.c(idx)[0] * uwVB[0] + sd.c(idx)[1] * uwVB[1] + sd.c(idx)[2] * uwVB[2]); };
				auto calc_f_with_pressBound = [&](int idx)
				{
					T scalarProdSquared = (sd.c(idx)[0] * uwPB[0] + sd.c(idx)[1] * uwPB[1] + sd.c(idx)[2] * uwPB[2]) * (sd.c(idx)[0] * uwPB[0] + sd.c(idx)[1] * uwPB[1] + sd.c(idx)[2] * uwPB[2]);
					return -st.f_star[pos][idx] + 2.0 * sd.w(idx) * rhoPB_border * (1.0 + scalarProdSquared / (2.0 * cs_sq<T> *cs_sq<T>) - (uwPB[0] * uwPB[0] + uwPB[1] * uwPB[1] + uwPB[2] * uwPB[2]) * (uwPB[0] * uwPB[0] + uwPB[1] * uwPB[1] + uwPB[2] * uwPB[2]) / (2.0 * cs_sq<T>));
				};



				for (int vbIdx = 0; vbIdx < sd.getVelBoundCount(); ++vbIdx)
				{
					auto& velBound = sd.getVelBound(vbIdx);

					if (x == 0 && velBound.side == left)	//left
					{
						if (y < sd.getGridDim_L(1) && z < sd.getGridDim_L(2))
						{
							velBound_defined = true;
							uwVB[0] = velBound.u_w / sd.getC_u();
							uwVB[1] = 0.0;
							uwVB[2] = 0.0;
							break;
						}
					}
					else if (x == sd.getGridDim_L(0) - 1 && velBound.side == right)	//right
					{
						if (y < sd.getGridDim_L(1) && z < sd.getGridDim_L(2))
						{
							velBound_defined = true;
							uwVB[0] = -velBound.u_w / sd.getC_u();
							uwVB[1] = 0.0;
							uwVB[2] = 0.0;
							break;
						}
					}
					else if (y == 0 && velBound.side == bottom)	//bottom
					{
						if (x < sd.getGridDim_L(0) && z < sd.getGridDim_L(2))
						{
							velBound_defined = true;
							uwVB[0] = 0.0;
							uwVB[1] = velBound.u_w / sd.getC_u();
							uwVB[2] = 0.0;
							break;
						}
					}
					else if (y == sd.getGridDim_L(1) - 1 && velBound.side == top)	//top
					{
						if (x < sd.getGridDim_L(0) && z < sd.getGridDim_L(2))
						{
							velBound_defined = true;
							uwVB[0] = 0.0;
							uwVB[1] = -velBound.u_w / sd.getC_u();
							uwVB[2] = 0.0;
							break;
						}
					}
					else if (z == 0 && velBound.side == back)	//back
					{
						if (x < sd.getGridDim_L(0) && y < sd.getGridDim_L(1))
						{
							velBound_defined = true;
							uwVB[0] = 0.0;
							uwVB[1] = 0.0;
							uwVB[2] = velBound.u_w / sd.getC_u();
							break;
						}
					}
					else if (z == sd.getGridDim_L(2) - 1 && velBound.side == front)	//front
					{
						if (x < sd.getGridDim_L(0) && y < sd.getGridDim_L(1))
						{
							velBound_defined = true;
							uwVB[0] = 0.0;
							uwVB[1] = 0.0;
							uwVB[2] = -velBound.u_w / sd.getC_u();
							break;
						}
					}
				}

				for (int pbIdx = 0; pbIdx < sd.getPressBoundCount(); pbIdx++)
				{
					auto& pressBound = sd.getPressBound(pbIdx);

					if (x == 0 && pressBound.side == left)	//left
					{
						if (y < sd.getGridDim_L(1) && z < sd.getGridDim_L(2))
						{
							pressBound_defined = true;
							int pos_next = (z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x + 1;

							uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
							rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
							break;
						}
					}
					else if (x == sd.getGridDim_L(0) - 1 && pressBound.side == right)	//right
					{
						if (y < sd.getGridDim_L(1) && z < sd.getGridDim_L(2))
						{
							pressBound_defined = true;
							int pos_next = (z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x - 1;

							uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
							rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
							break;
						}
					}
					else if (y == 0 && pressBound.side == bottom)	//bottom
					{
						if (x < sd.getGridDim_L(0) && z < sd.getGridDim_L(2))
						{
							pressBound_defined = true;
							int pos_next = (z * sd.getGridDim_L(1) + y + 1) * sd.getGridDim_L(0) + x;

							uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
							rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
							break;
						}
					}
					else if (y == sd.getGridDim_L(1) - 1 && pressBound.side == top)	//top
					{
						if (x < sd.getGridDim_L(0) && z < sd.getGridDim_L(2))
						{
							pressBound_defined = true;
							int pos_next = (z * sd.getGridDim_L(1) + y - 1) * sd.getGridDim_L(0) + x;

							uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
							rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
							break;
						}
					}
					else if (z == 0 && pressBound.side == back)	//back
					{
						if (x < sd.getGridDim_L(0) && y < sd.getGridDim_L(1))
						{
							pressBound_defined = true;
							int pos_next = ((z + 1) * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x;

							uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
							rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
							break;
						}
					}
					else if (z == sd.getGridDim_L(2) - 1 && pressBound.side == front)	//front
					{
						if (x < sd.getGridDim_L(0) && y < sd.getGridDim_L(1))
						{
							pressBound_defined = true;
							int pos_next = ((z - 1) * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x;

							uwPB = st.u_L[pos] + (T)0.5 * (st.u_L[pos] - st.u_L[pos_next]);
							rhoPB_border = (pressBound.dp_w / sd.getC_p()) / cs_sq<T> +1.0;
							break;
						}
					}
				}


				if (velBound_defined && pressBound_defined)
					throw std::invalid_argument("ERROR:: Inlett and Outlett can't be defined at the same latticenode::");

#pragma region BB
				i = 0;
				st.f[calcPos(i)][i] = calc_f(i);

				i = 1;
				if (x == sd.getGridDim_L(0) - 1)
				{
					if (velBound_defined)
						st.f[pos][2] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][2] = calc_f_with_pressBound(i);
					else
						st.f[pos][2] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 2;
				if (x == 0)
				{
					if (velBound_defined)
						st.f[pos][1] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][1] = calc_f_with_pressBound(i);
					else
						st.f[pos][1] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 3;
				if (y == sd.getGridDim_L(1) - 1)
				{
					if (velBound_defined)
						st.f[pos][4] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][4] = calc_f_with_pressBound(i);
					else
						st.f[pos][4] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 4;
				if (y == 0)
				{
					if (velBound_defined)
						st.f[pos][3] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][3] = calc_f_with_pressBound(i);
					else
						st.f[pos][3] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 5;
				if (z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][6] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][6] = calc_f_with_pressBound(i);
					else
						st.f[pos][6] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 6;
				if (z == 0)
				{
					if (velBound_defined)
						st.f[pos][5] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][5] = calc_f_with_pressBound(i);
					else
						st.f[pos][5] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 7;
				if (x == sd.getGridDim_L(0) - 1 || y == sd.getGridDim_L(1) - 1)
				{
					if (velBound_defined)
						st.f[pos][8] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][8] = calc_f_with_pressBound(i);
					else
						st.f[pos][8] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 8;
				if (x == 0 || y == 0)
				{
					if (velBound_defined)
						st.f[pos][7] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][7] = calc_f_with_pressBound(i);
					else
						st.f[pos][7] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 9;
				if (x == sd.getGridDim_L(0) - 1 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][10] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][10] = calc_f_with_pressBound(i);
					else
						st.f[pos][10] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 10;
				if (x == 0 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][9] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][9] = calc_f_with_pressBound(i);
					else
						st.f[pos][9] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 11;
				if (y == sd.getGridDim_L(1) - 1 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][12] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][12] = calc_f_with_pressBound(i);
					else
						st.f[pos][12] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 12;
				if (y == 0 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][11] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][11] = calc_f_with_pressBound(i);
					else
						st.f[pos][11] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 13;
				if (x == sd.getGridDim_L(0) - 1 || y == 0)
				{
					if (velBound_defined)
						st.f[pos][14] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][14] = calc_f_with_pressBound(i);
					else
						st.f[pos][14] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 14;
				if (x == 0 || y == sd.getGridDim_L(1) - 1)
				{
					if (velBound_defined)
						st.f[pos][13] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][13] = calc_f_with_pressBound(i);
					else
						st.f[pos][13] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 15;
				if (x == sd.getGridDim_L(0) - 1 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][16] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][16] = calc_f_with_pressBound(i);
					else
						st.f[pos][16] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 16;
				if (x == 0 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][15] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][15] = calc_f_with_pressBound(i);
					else
						st.f[pos][15] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 17;
				if (y == sd.getGridDim_L(1) - 1 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][18] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][18] = calc_f_with_pressBound(i);
					else
						st.f[pos][18] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 18;
				if (y == 0 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][17] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][17] = calc_f_with_pressBound(i);
					else
						st.f[pos][17] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 19;
				if (x == sd.getGridDim_L(0) - 1 || y == sd.getGridDim_L(1) - 1 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][20] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][20] = calc_f_with_pressBound(i);
					else
						st.f[pos][20] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 20;
				if (x == 0 || y == 0 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][19] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][19] = calc_f_with_pressBound(i);
					else
						st.f[pos][19] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 21;
				if (x == sd.getGridDim_L(0) - 1 || y == sd.getGridDim_L(1) - 1 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][22] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][22] = calc_f_with_pressBound(i);
					else
						st.f[pos][22] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 22;
				if (x == 0 || y == 0 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][21] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][21] = calc_f_with_pressBound(i);
					else
						st.f[pos][21] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 23;
				if (x == sd.getGridDim_L(0) - 1 || y == 0 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][24] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][24] = calc_f_with_pressBound(i);
					else
						st.f[pos][24] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 24;
				if (x == 0 || y == sd.getGridDim_L(1) - 1 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][23] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][23] = calc_f_with_pressBound(i);
					else
						st.f[pos][23] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 25;
				if (x == 0 || y == sd.getGridDim_L(1) - 1 || z == sd.getGridDim_L(2) - 1)
				{
					if (velBound_defined)
						st.f[pos][26] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][26] = calc_f_with_pressBound(i);
					else
						st.f[pos][26] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
				i = 26;
				if (x == sd.getGridDim_L(0) - 1 || y == 0 || z == 0)
				{
					if (velBound_defined)
						st.f[pos][25] = calc_f_with_velBound(i);
					else if (pressBound_defined)
						st.f[pos][25] = calc_f_with_pressBound(i);
					else
						st.f[pos][25] = calc_f(i);
				}
				else
					st.f[calcPos(i)][i] = calc_f(i);
#pragma endregion BB
			}
}

template<typename T>
void LBM_Solver_Specialisation<T, 2>::collision_CM()
{
	for (int y = 0; y < sd.getGridDim_L(1); ++y)
		for (int x = 0; x < sd.getGridDim_L(0); ++x)
		{
			int pos = y * sd.getGridDim_L(0) + x;

			T f0 = st.f[pos][0],
				f1 = st.f[pos][1],
				f2 = st.f[pos][2],
				f3 = st.f[pos][3],
				f4 = st.f[pos][4],
				f5 = st.f[pos][5],
				f6 = st.f[pos][6],
				f7 = st.f[pos][7],
				f8 = st.f[pos][8];

			T r0 = (T)0.0, 
				r1 = (T)2.0, 
				r2 = this->sd.getRelaxationConstant(),
				r3 = this->sd.getRelaxationConstant(),
				r4 = (T)1.0, 
				r5 = (T)1.0;

			T rho = st.rho_L[pos];
			T ux = st.u_L[pos][0];
			T uy = st.u_L[pos][1];
			T Fx = st.F_ext_L[pos][0];
			T Fy = st.F_ext_L[pos][1];

			T f_t0 = r0 * (f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 - rho);
			T f_t1 = -r1 * (f0 * ux + f2 * ux + f4 * ux + f1 * (ux - 1.0) + f3 * (ux + 1.0) + f5 * (ux - 1.0) + f6 * (ux + 1.0) + f7 * (ux + 1.0) + f8 * (ux - 1.0));
			T f_t2 = -r1 * (f0 * uy + f1 * uy + f3 * uy + f2 * (uy - 1.0) + f4 * (uy + 1.0) + f5 * (uy - 1.0) + f6 * (uy - 1.0) + f7 * (uy + 1.0) + f8 * (uy + 1.0));
			T f_t3 = r2 * (rho * (-2.0 / 3.0) + f1 * (pow(ux - 1.0, 2.0) + uy * uy) + f2 * (pow(uy - 1.0, 2.0) + ux * ux) + f3 * (pow(ux + 1.0, 2.0) + uy * uy) + f4 * (pow(uy + 1.0, 2.0) + ux * ux) + f5 * (pow(ux - 1.0, 2.0) + pow(uy - 1.0, 2.0)) + f6 * (pow(ux + 1.0, 2.0) + pow(uy - 1.0, 2.0)) + f7 * (pow(ux + 1.0, 2.0) + pow(uy + 1.0, 2.0)) + f8 * (pow(ux - 1.0, 2.0) + pow(uy + 1.0, 2.0)) + f0 * (ux * ux + uy * uy));
			T f_t4 = r2 * (f0 * (ux * ux - uy * uy) + f1 * (pow(ux - 1.0, 2.0) - uy * uy) - f2 * (pow(uy - 1.0, 2.0) - ux * ux) + f3 * (pow(ux + 1.0, 2.0) - uy * uy) - f4 * (pow(uy + 1.0, 2.0) - ux * ux) + f5 * (pow(ux - 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f6 * (pow(ux + 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f7 * (pow(ux + 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f8 * (pow(ux - 1.0, 2.0) - pow(uy + 1.0, 2.0)));
			T f_t5 = r3 * (f1 * uy * (ux - 1.0) + f2 * ux * (uy - 1.0) + f3 * uy * (ux + 1.0) + f4 * ux * (uy + 1.0) + f5 * (ux - 1.0) * (uy - 1.0) + f6 * (ux + 1.0) * (uy - 1.0) + f7 * (ux + 1.0) * (uy + 1.0) + f8 * (ux - 1.0) * (uy + 1.0) + f0 * ux * uy);
			T f_t6 = -r4 * (f0 * (ux * ux) * uy + f1 * uy * pow(ux - 1.0, 2.0) + f2 * (ux * ux) * (uy - 1.0) + f3 * uy * pow(ux + 1.0, 2.0) + f4 * (ux * ux) * (uy + 1.0) + f5 * pow(ux - 1.0, 2.0) * (uy - 1.0) + f6 * pow(ux + 1.0, 2.0) * (uy - 1.0) + f7 * pow(ux + 1.0, 2.0) * (uy + 1.0) + f8 * pow(ux - 1.0, 2.0) * (uy + 1.0));
			T f_t7 = -r4 * (f0 * ux * (uy * uy) + f1 * (uy * uy) * (ux - 1.0) + f2 * ux * pow(uy - 1.0, 2.0) + f3 * (uy * uy) * (ux + 1.0) + f4 * ux * pow(uy + 1.0, 2.0) + f5 * (ux - 1.0) * pow(uy - 1.0, 2.0) + f6 * (ux + 1.0) * pow(uy - 1.0, 2.0) + f7 * (ux + 1.0) * pow(uy + 1.0, 2.0) + f8 * (ux - 1.0) * pow(uy + 1.0, 2.0));
			T f_t8 = r5 * (rho * (-1.0 / 9.0) + f5 * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + f6 * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + f7 * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + f8 * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + f0 * (ux * ux) * (uy * uy) + f1 * (uy * uy) * pow(ux - 1.0, 2.0) + f2 * (ux * ux) * pow(uy - 1.0, 2.0) + f3 * (uy * uy) * pow(ux + 1.0, 2.0) + f4 * (ux * ux) * pow(uy + 1.0, 2.0));

			T Omega_0 = -f_t8 - f_t7 * ux * 2.0 - f_t6 * uy * 2.0 - f_t3 * ((ux * ux) / 2.0 + (uy * uy) / 2.0 - 1.0) + f_t4 * ((ux * ux) / 2.0 - (uy * uy) / 2.0) - f_t0 * ((ux * ux) * (uy * uy) - ux * ux - uy * uy + 1.0) + f_t1 * (ux * 2.0 - ux * (uy * uy) * 2.0) + f_t2 * (uy * 2.0 - (ux * ux) * uy * 2.0) - f_t5 * ux * uy * 4.0;
			T Omega_1 = f_t8 / 2.0 + f_t2 * (ux * uy + (ux * ux) * uy) + f_t6 * uy + f_t3 * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - f_t4 * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) - f_t1 * (ux - ux * (uy * uy) - (uy * uy) / 2.0 + 1.0 / 2.0) + f_t7 * (ux + 1.0 / 2.0) - f_t0 * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * ux) / 2.0) + f_t5 * (uy + ux * uy * 2.0);
			T Omega_2 = f_t8 / 2.0 + f_t1 * (ux * uy + ux * (uy * uy)) + f_t7 * ux + f_t3 * (uy / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) + f_t4 * (uy / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0 + 1.0 / 4.0) - f_t2 * (uy - (ux * ux) * uy - (ux * ux) / 2.0 + 1.0 / 2.0) + f_t6 * (uy + 1.0 / 2.0) - f_t0 * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * uy) / 2.0) + f_t5 * (ux + ux * uy * 2.0);
			T Omega_3 = f_t8 / 2.0 - f_t2 * (ux * uy - (ux * ux) * uy) + f_t6 * uy - f_t3 * (ux / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) + f_t4 * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - f_t1 * (ux - ux * (uy * uy) + (uy * uy) / 2.0 - 1.0 / 2.0) + f_t7 * (ux - 1.0 / 2.0) + f_t0 * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * ux) / 2.0) - f_t5 * (uy - ux * uy * 2.0);
			T Omega_4 = f_t8 / 2.0 - f_t1 * (ux * uy - ux * (uy * uy)) + f_t7 * ux - f_t3 * (uy / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) - f_t4 * (uy / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0 - 1.0 / 4.0) - f_t2 * (uy - (ux * ux) * uy + (ux * ux) / 2.0 - 1.0 / 2.0) + f_t6 * (uy - 1.0 / 2.0) + f_t0 * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * uy) / 2.0) - f_t5 * (ux - ux * uy * 2.0);
			T Omega_5 = f_t8 * (-1.0 / 4.0) - f_t5 * (ux / 2.0 + uy / 2.0 + ux * uy + 1.0 / 4.0) - f_t2 * (ux / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) - f_t1 * (uy / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0) - f_t3 * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) + f_t4 * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 + 1.0 / 4.0) - f_t6 * (uy / 2.0 + 1.0 / 4.0);
			T Omega_6 = f_t8 * (-1.0 / 4.0) - f_t5 * (ux / 2.0 - uy / 2.0 + ux * uy - 1.0 / 4.0) + f_t2 * (ux / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) + f_t1 * (uy / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0) - f_t3 * (ux * (-1.0 / 8.0) + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) - f_t4 * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 - 1.0 / 4.0) - f_t6 * (uy / 2.0 + 1.0 / 4.0);
			T Omega_7 = f_t8 * (-1.0 / 4.0) + f_t5 * (ux / 2.0 + uy / 2.0 - ux * uy - 1.0 / 4.0) - f_t2 * (ux / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) - f_t1 * (uy / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * uy) / 4.0) + f_t3 * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0) - f_t4 * (ux / 8.0 - uy / 8.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 - 1.0 / 4.0) - f_t6 * (uy / 2.0 - 1.0 / 4.0);
			T Omega_8 = f_t8 * (-1.0 / 4.0) + f_t2 * (ux / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) - f_t3 * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) + f_t4 * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0) - f_t7 * (ux / 2.0 + 1.0 / 4.0) - f_t6 * (uy / 2.0 - 1.0 / 4.0) + (f_t1 * (uy - uy * uy) * (ux * 2.0 + 1.0)) / 4.0 - (f_t5 * (ux * 2.0 + 1.0) * (uy * 2.0 - 1.0)) / 4.0;


			T G_0 = Fx * uy * (r4 / 2.0 - 1.0) * (-2.0 / 3.0) - Fy * ux * (r4 / 2.0 - 1.0) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * 2.0 - ux * (uy * uy) * 2.0) + Fy * (r1 / 2.0 - 1.0) * (uy * 2.0 - (ux * ux) * uy * 2.0);
			T G_1 = Fy * (r1 / 2.0 - 1.0) * (ux * uy + (ux * ux) * uy) + (Fx * uy * (r4 / 2.0 - 1.0)) / 3.0 - Fx * (r1 / 2.0 - 1.0) * (ux - ux * (uy * uy) - (uy * uy) / 2.0 + 1.0 / 2.0) + (Fy * (r4 / 2.0 - 1.0) * (ux + 1.0 / 2.0)) / 3.0;
			T G_2 = Fx * (r1 / 2.0 - 1.0) * (ux * uy + ux * (uy * uy)) + (Fy * ux * (r4 / 2.0 - 1.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (uy - (ux * ux) * uy - (ux * ux) / 2.0 + 1.0 / 2.0) + (Fx * (r4 / 2.0 - 1.0) * (uy + 1.0 / 2.0)) / 3.0;
			T G_3 = -Fy * (r1 / 2.0 - 1.0) * (ux * uy - (ux * ux) * uy) + (Fx * uy * (r4 / 2.0 - 1.0)) / 3.0 - Fx * (r1 / 2.0 - 1.0) * (ux - ux * (uy * uy) + (uy * uy) / 2.0 - 1.0 / 2.0) + (Fy * (r4 / 2.0 - 1.0) * (ux - 1.0 / 2.0)) / 3.0;
			T G_4 = -Fx * (r1 / 2.0 - 1.0) * (ux * uy - ux * (uy * uy)) + (Fy * ux * (r4 / 2.0 - 1.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (uy - (ux * ux) * uy + (ux * ux) / 2.0 - 1.0 / 2.0) + (Fx * (r4 / 2.0 - 1.0) * (uy - 1.0 / 2.0)) / 3.0;
			T G_5 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0);
			T G_6 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 3.0 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 + (uy * uy) / 4.0);
			T G_7 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 3.0 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * ux) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * uy) / 4.0);
			T G_8 = Fx * (r4 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0) * (-1.0 / 3.0) - (Fy * (r4 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 3.0 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 + (ux * ux) / 4.0) + (Fx * (uy - uy * uy) * (r1 / 2.0 - 1.0) * (ux * 2.0 + 1.0)) / 4.0;



			st.f_star[pos][0] = f0 + Omega_0 + G_0;
			st.f_star[pos][1] = f1 + Omega_1 + G_1;
			st.f_star[pos][2] = f2 + Omega_2 + G_2;
			st.f_star[pos][3] = f3 + Omega_3 + G_3;
			st.f_star[pos][4] = f4 + Omega_4 + G_4;
			st.f_star[pos][5] = f5 + Omega_5 + G_5;
			st.f_star[pos][6] = f6 + Omega_6 + G_6;
			st.f_star[pos][7] = f7 + Omega_7 + G_7;
			st.f_star[pos][8] = f8 + Omega_8 + G_8;
		}
}

template<typename T>
void LBM_Solver_Specialisation<T, 3>::collision_CM()
{
	for (int z = 0; z < sd.getGridDim_L(2); ++z)
		for (int y = 0; y < sd.getGridDim_L(1); ++y)
			for (int x = 0; x < sd.getGridDim_L(0); ++x)
			{
				int pos = (z * sd.getGridDim_L(1) + y) * sd.getGridDim_L(0) + x;
				T f0 = st.f[pos][0],
					f1 = st.f[pos][1],
					f2 = st.f[pos][2],
					f3 = st.f[pos][3],
					f4 = st.f[pos][4],
					f5 = st.f[pos][5],
					f6 = st.f[pos][6],
					f7 = st.f[pos][7],
					f8 = st.f[pos][8],
					f9 = st.f[pos][9],
					f10 = st.f[pos][10],
					f11 = st.f[pos][11],
					f12 = st.f[pos][12],
					f13 = st.f[pos][13],
					f14 = st.f[pos][14],
					f15 = st.f[pos][15],
					f16 = st.f[pos][16],
					f17 = st.f[pos][17],
					f18 = st.f[pos][18],
					f19 = st.f[pos][19],
					f20 = st.f[pos][20],
					f21 = st.f[pos][21],
					f22 = st.f[pos][22],
					f23 = st.f[pos][23],
					f24 = st.f[pos][24],
					f25 = st.f[pos][25],
					f26 = st.f[pos][26];
				//1.75
				T r0 = (T)0.0, 
					r1 = (T)2.0, 
					r2 = this->sd.getRelaxationConstant(),
					r3 = this->sd.getRelaxationConstant(),
					r4 = (T)1.0, 
					r5 = (T)1.0, 
					r6 = (T)1.0;

				T rho = st.rho_L[pos];
				T ux = st.u_L[pos][0];
				T uy = st.u_L[pos][1];
				T uz = st.u_L[pos][2];
				T Fx = st.F_ext_L[pos][0];
				T Fy = st.F_ext_L[pos][1];
				T Fz = st.F_ext_L[pos][2];

				T f_t0 = r0 * (f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19 + f20 + f21 + f22 + f23 + f24 + f25 + f26 - rho);
				T f_t1 = -r1 * (f0 * ux + f3 * ux + f4 * ux + f5 * ux + f6 * ux + f11 * ux + f12 * ux + f17 * ux + f18 * ux + f1 * (ux - 1.0) + f2 * (ux + 1.0) + f7 * (ux - 1.0) + f8 * (ux + 1.0) + f9 * (ux - 1.0) + f10 * (ux + 1.0) + f13 * (ux - 1.0) + f14 * (ux + 1.0) + f15 * (ux - 1.0) + f16 * (ux + 1.0) + f19 * (ux - 1.0) + f20 * (ux + 1.0) + f21 * (ux - 1.0) + f22 * (ux + 1.0) + f23 * (ux - 1.0) + f24 * (ux + 1.0) + f25 * (ux + 1.0) + f26 * (ux - 1.0));
				T f_t2 = -r1 * (f0 * uy + f1 * uy + f2 * uy + f5 * uy + f6 * uy + f9 * uy + f10 * uy + f15 * uy + f16 * uy + f3 * (uy - 1.0) + f4 * (uy + 1.0) + f7 * (uy - 1.0) + f8 * (uy + 1.0) + f11 * (uy - 1.0) + f12 * (uy + 1.0) + f13 * (uy + 1.0) + f14 * (uy - 1.0) + f17 * (uy - 1.0) + f18 * (uy + 1.0) + f19 * (uy - 1.0) + f20 * (uy + 1.0) + f21 * (uy - 1.0) + f22 * (uy + 1.0) + f23 * (uy + 1.0) + f24 * (uy - 1.0) + f25 * (uy - 1.0) + f26 * (uy + 1.0));
				T f_t3 = -r1 * (f0 * uz + f1 * uz + f2 * uz + f3 * uz + f4 * uz + f7 * uz + f8 * uz + f13 * uz + f14 * uz + f5 * (uz - 1.0) + f6 * (uz + 1.0) + f9 * (uz - 1.0) + f10 * (uz + 1.0) + f11 * (uz - 1.0) + f12 * (uz + 1.0) + f15 * (uz + 1.0) + f16 * (uz - 1.0) + f17 * (uz + 1.0) + f18 * (uz - 1.0) + f19 * (uz - 1.0) + f20 * (uz + 1.0) + f21 * (uz + 1.0) + f22 * (uz - 1.0) + f23 * (uz - 1.0) + f24 * (uz + 1.0) + f25 * (uz - 1.0) + f26 * (uz + 1.0));
				T f_t4 = r2 * (f1 * uy * (ux - 1.0) + f2 * uy * (ux + 1.0) + f3 * ux * (uy - 1.0) + f4 * ux * (uy + 1.0) + f9 * uy * (ux - 1.0) + f10 * uy * (ux + 1.0) + f11 * ux * (uy - 1.0) + f12 * ux * (uy + 1.0) + f15 * uy * (ux - 1.0) + f16 * uy * (ux + 1.0) + f17 * ux * (uy - 1.0) + f18 * ux * (uy + 1.0) + f7 * (ux - 1.0) * (uy - 1.0) + f8 * (ux + 1.0) * (uy + 1.0) + f13 * (ux - 1.0) * (uy + 1.0) + f14 * (ux + 1.0) * (uy - 1.0) + f19 * (ux - 1.0) * (uy - 1.0) + f20 * (ux + 1.0) * (uy + 1.0) + f21 * (ux - 1.0) * (uy - 1.0) + f22 * (ux + 1.0) * (uy + 1.0) + f23 * (ux - 1.0) * (uy + 1.0) + f24 * (ux + 1.0) * (uy - 1.0) + f25 * (ux + 1.0) * (uy - 1.0) + f26 * (ux - 1.0) * (uy + 1.0) + f0 * ux * uy + f5 * ux * uy + f6 * ux * uy);
				T f_t5 = r2 * (f1 * uz * (ux - 1.0) + f2 * uz * (ux + 1.0) + f5 * ux * (uz - 1.0) + f6 * ux * (uz + 1.0) + f7 * uz * (ux - 1.0) + f8 * uz * (ux + 1.0) + f11 * ux * (uz - 1.0) + f12 * ux * (uz + 1.0) + f13 * uz * (ux - 1.0) + f14 * uz * (ux + 1.0) + f17 * ux * (uz + 1.0) + f18 * ux * (uz - 1.0) + f9 * (ux - 1.0) * (uz - 1.0) + f10 * (ux + 1.0) * (uz + 1.0) + f15 * (ux - 1.0) * (uz + 1.0) + f16 * (ux + 1.0) * (uz - 1.0) + f19 * (ux - 1.0) * (uz - 1.0) + f20 * (ux + 1.0) * (uz + 1.0) + f21 * (ux - 1.0) * (uz + 1.0) + f22 * (ux + 1.0) * (uz - 1.0) + f23 * (ux - 1.0) * (uz - 1.0) + f24 * (ux + 1.0) * (uz + 1.0) + f25 * (ux + 1.0) * (uz - 1.0) + f26 * (ux - 1.0) * (uz + 1.0) + f0 * ux * uz + f3 * ux * uz + f4 * ux * uz);
				T f_t6 = r2 * (f3 * uz * (uy - 1.0) + f4 * uz * (uy + 1.0) + f5 * uy * (uz - 1.0) + f6 * uy * (uz + 1.0) + f7 * uz * (uy - 1.0) + f8 * uz * (uy + 1.0) + f9 * uy * (uz - 1.0) + f10 * uy * (uz + 1.0) + f13 * uz * (uy + 1.0) + f14 * uz * (uy - 1.0) + f15 * uy * (uz + 1.0) + f16 * uy * (uz - 1.0) + f11 * (uy - 1.0) * (uz - 1.0) + f12 * (uy + 1.0) * (uz + 1.0) + f17 * (uy - 1.0) * (uz + 1.0) + f18 * (uy + 1.0) * (uz - 1.0) + f19 * (uy - 1.0) * (uz - 1.0) + f20 * (uy + 1.0) * (uz + 1.0) + f21 * (uy - 1.0) * (uz + 1.0) + f22 * (uy + 1.0) * (uz - 1.0) + f23 * (uy + 1.0) * (uz - 1.0) + f24 * (uy - 1.0) * (uz + 1.0) + f25 * (uy - 1.0) * (uz - 1.0) + f26 * (uy + 1.0) * (uz + 1.0) + f0 * uy * uz + f1 * uy * uz + f2 * uy * uz);
				T f_t7 = r2 * (f0 * (ux * ux - uy * uy) + f5 * (ux * ux - uy * uy) + f6 * (ux * ux - uy * uy) + f1 * (pow(ux - 1.0, 2.0) - uy * uy) + f2 * (pow(ux + 1.0, 2.0) - uy * uy) - f3 * (pow(uy - 1.0, 2.0) - ux * ux) - f4 * (pow(uy + 1.0, 2.0) - ux * ux) + f9 * (pow(ux - 1.0, 2.0) - uy * uy) + f10 * (pow(ux + 1.0, 2.0) - uy * uy) - f11 * (pow(uy - 1.0, 2.0) - ux * ux) - f12 * (pow(uy + 1.0, 2.0) - ux * ux) + f15 * (pow(ux - 1.0, 2.0) - uy * uy) + f16 * (pow(ux + 1.0, 2.0) - uy * uy) - f17 * (pow(uy - 1.0, 2.0) - ux * ux) - f18 * (pow(uy + 1.0, 2.0) - ux * ux) + f7 * (pow(ux - 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f8 * (pow(ux + 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f13 * (pow(ux - 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f14 * (pow(ux + 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f19 * (pow(ux - 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) - pow(uy + 1.0, 2.0)));
				T f_t8 = r2 * (f0 * (ux * ux - uz * uz) + f3 * (ux * ux - uz * uz) + f4 * (ux * ux - uz * uz) + f1 * (pow(ux - 1.0, 2.0) - uz * uz) + f2 * (pow(ux + 1.0, 2.0) - uz * uz) - f5 * (pow(uz - 1.0, 2.0) - ux * ux) - f6 * (pow(uz + 1.0, 2.0) - ux * ux) + f7 * (pow(ux - 1.0, 2.0) - uz * uz) + f8 * (pow(ux + 1.0, 2.0) - uz * uz) - f11 * (pow(uz - 1.0, 2.0) - ux * ux) - f12 * (pow(uz + 1.0, 2.0) - ux * ux) + f13 * (pow(ux - 1.0, 2.0) - uz * uz) + f14 * (pow(ux + 1.0, 2.0) - uz * uz) - f17 * (pow(uz + 1.0, 2.0) - ux * ux) - f18 * (pow(uz - 1.0, 2.0) - ux * ux) + f9 * (pow(ux - 1.0, 2.0) - pow(uz - 1.0, 2.0)) + f10 * (pow(ux + 1.0, 2.0) - pow(uz + 1.0, 2.0)) + f15 * (pow(ux - 1.0, 2.0) - pow(uz + 1.0, 2.0)) + f16 * (pow(ux + 1.0, 2.0) - pow(uz - 1.0, 2.0)) + f19 * (pow(ux - 1.0, 2.0) - pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) - pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) - pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) - pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) - pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) - pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) - pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) - pow(uz + 1.0, 2.0)));
				T f_t9 = r2 * (-rho + f1 * (pow(ux - 1.0, 2.0) + uy * uy + uz * uz) + f2 * (pow(ux + 1.0, 2.0) + uy * uy + uz * uz) + f3 * (pow(uy - 1.0, 2.0) + ux * ux + uz * uz) + f4 * (pow(uy + 1.0, 2.0) + ux * ux + uz * uz) + f5 * (pow(uz - 1.0, 2.0) + ux * ux + uy * uy) + f6 * (pow(uz + 1.0, 2.0) + ux * ux + uy * uy) + f7 * (pow(ux - 1.0, 2.0) + pow(uy - 1.0, 2.0) + uz * uz) + f8 * (pow(ux + 1.0, 2.0) + pow(uy + 1.0, 2.0) + uz * uz) + f9 * (pow(ux - 1.0, 2.0) + pow(uz - 1.0, 2.0) + uy * uy) + f10 * (pow(ux + 1.0, 2.0) + pow(uz + 1.0, 2.0) + uy * uy) + f11 * (pow(uy - 1.0, 2.0) + pow(uz - 1.0, 2.0) + ux * ux) + f12 * (pow(uy + 1.0, 2.0) + pow(uz + 1.0, 2.0) + ux * ux) + f13 * (pow(ux - 1.0, 2.0) + pow(uy + 1.0, 2.0) + uz * uz) + f14 * (pow(ux + 1.0, 2.0) + pow(uy - 1.0, 2.0) + uz * uz) + f15 * (pow(ux - 1.0, 2.0) + pow(uz + 1.0, 2.0) + uy * uy) + f16 * (pow(ux + 1.0, 2.0) + pow(uz - 1.0, 2.0) + uy * uy) + f17 * (pow(uy - 1.0, 2.0) + pow(uz + 1.0, 2.0) + ux * ux) + f18 * (pow(uy + 1.0, 2.0) + pow(uz - 1.0, 2.0) + ux * ux) + f0 * (ux * ux + uy * uy + uz * uz) + f19 * (pow(ux - 1.0, 2.0) + pow(uy - 1.0, 2.0) + pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) + pow(uy + 1.0, 2.0) + pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) + pow(uy - 1.0, 2.0) + pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) + pow(uy + 1.0, 2.0) + pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) + pow(uy + 1.0, 2.0) + pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) + pow(uy - 1.0, 2.0) + pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) + pow(uy - 1.0, 2.0) + pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) + pow(uy + 1.0, 2.0) + pow(uz + 1.0, 2.0)));
				T f_t10 = -r3 * (f1 * ((uy * uy) * (ux - 1.0) + (uz * uz) * (ux - 1.0)) + f2 * ((uy * uy) * (ux + 1.0) + (uz * uz) * (ux + 1.0)) + f11 * (ux * pow(uy - 1.0, 2.0) + ux * pow(uz - 1.0, 2.0)) + f12 * (ux * pow(uy + 1.0, 2.0) + ux * pow(uz + 1.0, 2.0)) + f17 * (ux * pow(uy - 1.0, 2.0) + ux * pow(uz + 1.0, 2.0)) + f18 * (ux * pow(uy + 1.0, 2.0) + ux * pow(uz - 1.0, 2.0)) + f0 * (ux * (uy * uy) + ux * (uz * uz)) + f7 * ((uz * uz) * (ux - 1.0) + (ux - 1.0) * pow(uy - 1.0, 2.0)) + f8 * ((uz * uz) * (ux + 1.0) + (ux + 1.0) * pow(uy + 1.0, 2.0)) + f9 * ((uy * uy) * (ux - 1.0) + (ux - 1.0) * pow(uz - 1.0, 2.0)) + f10 * ((uy * uy) * (ux + 1.0) + (ux + 1.0) * pow(uz + 1.0, 2.0)) + f13 * ((uz * uz) * (ux - 1.0) + (ux - 1.0) * pow(uy + 1.0, 2.0)) + f14 * ((uz * uz) * (ux + 1.0) + (ux + 1.0) * pow(uy - 1.0, 2.0)) + f15 * ((uy * uy) * (ux - 1.0) + (ux - 1.0) * pow(uz + 1.0, 2.0)) + f16 * ((uy * uy) * (ux + 1.0) + (ux + 1.0) * pow(uz - 1.0, 2.0)) + f3 * (ux * pow(uy - 1.0, 2.0) + ux * (uz * uz)) + f4 * (ux * pow(uy + 1.0, 2.0) + ux * (uz * uz)) + f5 * (ux * pow(uz - 1.0, 2.0) + ux * (uy * uy)) + f6 * (ux * pow(uz + 1.0, 2.0) + ux * (uy * uy)) + f19 * ((ux - 1.0) * pow(uy - 1.0, 2.0) + (ux - 1.0) * pow(uz - 1.0, 2.0)) + f20 * ((ux + 1.0) * pow(uy + 1.0, 2.0) + (ux + 1.0) * pow(uz + 1.0, 2.0)) + f21 * ((ux - 1.0) * pow(uy - 1.0, 2.0) + (ux - 1.0) * pow(uz + 1.0, 2.0)) + f22 * ((ux + 1.0) * pow(uy + 1.0, 2.0) + (ux + 1.0) * pow(uz - 1.0, 2.0)) + f23 * ((ux - 1.0) * pow(uy + 1.0, 2.0) + (ux - 1.0) * pow(uz - 1.0, 2.0)) + f24 * ((ux + 1.0) * pow(uy - 1.0, 2.0) + (ux + 1.0) * pow(uz + 1.0, 2.0)) + f25 * ((ux + 1.0) * pow(uy - 1.0, 2.0) + (ux + 1.0) * pow(uz - 1.0, 2.0)) + f26 * ((ux - 1.0) * pow(uy + 1.0, 2.0) + (ux - 1.0) * pow(uz + 1.0, 2.0)));
				T f_t11 = -r3 * (f3 * ((ux * ux) * (uy - 1.0) + (uz * uz) * (uy - 1.0)) + f4 * ((ux * ux) * (uy + 1.0) + (uz * uz) * (uy + 1.0)) + f9 * (uy * pow(ux - 1.0, 2.0) + uy * pow(uz - 1.0, 2.0)) + f10 * (uy * pow(ux + 1.0, 2.0) + uy * pow(uz + 1.0, 2.0)) + f15 * (uy * pow(ux - 1.0, 2.0) + uy * pow(uz + 1.0, 2.0)) + f16 * (uy * pow(ux + 1.0, 2.0) + uy * pow(uz - 1.0, 2.0)) + f0 * ((ux * ux) * uy + uy * (uz * uz)) + f7 * ((uz * uz) * (uy - 1.0) + pow(ux - 1.0, 2.0) * (uy - 1.0)) + f8 * ((uz * uz) * (uy + 1.0) + pow(ux + 1.0, 2.0) * (uy + 1.0)) + f11 * ((ux * ux) * (uy - 1.0) + (uy - 1.0) * pow(uz - 1.0, 2.0)) + f12 * ((ux * ux) * (uy + 1.0) + (uy + 1.0) * pow(uz + 1.0, 2.0)) + f13 * ((uz * uz) * (uy + 1.0) + pow(ux - 1.0, 2.0) * (uy + 1.0)) + f14 * ((uz * uz) * (uy - 1.0) + pow(ux + 1.0, 2.0) * (uy - 1.0)) + f17 * ((ux * ux) * (uy - 1.0) + (uy - 1.0) * pow(uz + 1.0, 2.0)) + f18 * ((ux * ux) * (uy + 1.0) + (uy + 1.0) * pow(uz - 1.0, 2.0)) + f1 * (uy * pow(ux - 1.0, 2.0) + uy * (uz * uz)) + f2 * (uy * pow(ux + 1.0, 2.0) + uy * (uz * uz)) + f5 * (uy * pow(uz - 1.0, 2.0) + (ux * ux) * uy) + f6 * (uy * pow(uz + 1.0, 2.0) + (ux * ux) * uy) + f19 * (pow(ux - 1.0, 2.0) * (uy - 1.0) + (uy - 1.0) * pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) * (uy + 1.0) + (uy + 1.0) * pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) * (uy - 1.0) + (uy - 1.0) * pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) * (uy + 1.0) + (uy + 1.0) * pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) * (uy + 1.0) + (uy + 1.0) * pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) * (uy - 1.0) + (uy - 1.0) * pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) * (uy - 1.0) + (uy - 1.0) * pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) * (uy + 1.0) + (uy + 1.0) * pow(uz + 1.0, 2.0)));
				T f_t12 = -r3 * (f5 * ((ux * ux) * (uz - 1.0) + (uy * uy) * (uz - 1.0)) + f6 * ((ux * ux) * (uz + 1.0) + (uy * uy) * (uz + 1.0)) + f7 * (uz * pow(ux - 1.0, 2.0) + uz * pow(uy - 1.0, 2.0)) + f8 * (uz * pow(ux + 1.0, 2.0) + uz * pow(uy + 1.0, 2.0)) + f13 * (uz * pow(ux - 1.0, 2.0) + uz * pow(uy + 1.0, 2.0)) + f14 * (uz * pow(ux + 1.0, 2.0) + uz * pow(uy - 1.0, 2.0)) + f0 * ((ux * ux) * uz + (uy * uy) * uz) + f9 * ((uy * uy) * (uz - 1.0) + pow(ux - 1.0, 2.0) * (uz - 1.0)) + f10 * ((uy * uy) * (uz + 1.0) + pow(ux + 1.0, 2.0) * (uz + 1.0)) + f11 * ((ux * ux) * (uz - 1.0) + pow(uy - 1.0, 2.0) * (uz - 1.0)) + f12 * ((ux * ux) * (uz + 1.0) + pow(uy + 1.0, 2.0) * (uz + 1.0)) + f15 * ((uy * uy) * (uz + 1.0) + pow(ux - 1.0, 2.0) * (uz + 1.0)) + f16 * ((uy * uy) * (uz - 1.0) + pow(ux + 1.0, 2.0) * (uz - 1.0)) + f17 * ((ux * ux) * (uz + 1.0) + pow(uy - 1.0, 2.0) * (uz + 1.0)) + f18 * ((ux * ux) * (uz - 1.0) + pow(uy + 1.0, 2.0) * (uz - 1.0)) + f1 * (uz * pow(ux - 1.0, 2.0) + (uy * uy) * uz) + f2 * (uz * pow(ux + 1.0, 2.0) + (uy * uy) * uz) + f3 * (uz * pow(uy - 1.0, 2.0) + (ux * ux) * uz) + f4 * (uz * pow(uy + 1.0, 2.0) + (ux * ux) * uz) + f19 * (pow(ux - 1.0, 2.0) * (uz - 1.0) + pow(uy - 1.0, 2.0) * (uz - 1.0)) + f20 * (pow(ux + 1.0, 2.0) * (uz + 1.0) + pow(uy + 1.0, 2.0) * (uz + 1.0)) + f21 * (pow(ux - 1.0, 2.0) * (uz + 1.0) + pow(uy - 1.0, 2.0) * (uz + 1.0)) + f22 * (pow(ux + 1.0, 2.0) * (uz - 1.0) + pow(uy + 1.0, 2.0) * (uz - 1.0)) + f23 * (pow(ux - 1.0, 2.0) * (uz - 1.0) + pow(uy + 1.0, 2.0) * (uz - 1.0)) + f24 * (pow(ux + 1.0, 2.0) * (uz + 1.0) + pow(uy - 1.0, 2.0) * (uz + 1.0)) + f25 * (pow(ux + 1.0, 2.0) * (uz - 1.0) + pow(uy - 1.0, 2.0) * (uz - 1.0)) + f26 * (pow(ux - 1.0, 2.0) * (uz + 1.0) + pow(uy + 1.0, 2.0) * (uz + 1.0)));
				T f_t13 = -r3 * (f1 * ((uy * uy) * (ux - 1.0) - (uz * uz) * (ux - 1.0)) + f2 * ((uy * uy) * (ux + 1.0) - (uz * uz) * (ux + 1.0)) + f11 * (ux * pow(uy - 1.0, 2.0) - ux * pow(uz - 1.0, 2.0)) + f12 * (ux * pow(uy + 1.0, 2.0) - ux * pow(uz + 1.0, 2.0)) + f17 * (ux * pow(uy - 1.0, 2.0) - ux * pow(uz + 1.0, 2.0)) + f18 * (ux * pow(uy + 1.0, 2.0) - ux * pow(uz - 1.0, 2.0)) + f0 * (ux * (uy * uy) - ux * (uz * uz)) - f7 * ((uz * uz) * (ux - 1.0) - (ux - 1.0) * pow(uy - 1.0, 2.0)) - f8 * ((uz * uz) * (ux + 1.0) - (ux + 1.0) * pow(uy + 1.0, 2.0)) + f9 * ((uy * uy) * (ux - 1.0) - (ux - 1.0) * pow(uz - 1.0, 2.0)) + f10 * ((uy * uy) * (ux + 1.0) - (ux + 1.0) * pow(uz + 1.0, 2.0)) - f13 * ((uz * uz) * (ux - 1.0) - (ux - 1.0) * pow(uy + 1.0, 2.0)) - f14 * ((uz * uz) * (ux + 1.0) - (ux + 1.0) * pow(uy - 1.0, 2.0)) + f15 * ((uy * uy) * (ux - 1.0) - (ux - 1.0) * pow(uz + 1.0, 2.0)) + f16 * ((uy * uy) * (ux + 1.0) - (ux + 1.0) * pow(uz - 1.0, 2.0)) + f3 * (ux * pow(uy - 1.0, 2.0) - ux * (uz * uz)) + f4 * (ux * pow(uy + 1.0, 2.0) - ux * (uz * uz)) - f5 * (ux * pow(uz - 1.0, 2.0) - ux * (uy * uy)) - f6 * (ux * pow(uz + 1.0, 2.0) - ux * (uy * uy)) + f19 * ((ux - 1.0) * pow(uy - 1.0, 2.0) - (ux - 1.0) * pow(uz - 1.0, 2.0)) + f20 * ((ux + 1.0) * pow(uy + 1.0, 2.0) - (ux + 1.0) * pow(uz + 1.0, 2.0)) + f21 * ((ux - 1.0) * pow(uy - 1.0, 2.0) - (ux - 1.0) * pow(uz + 1.0, 2.0)) + f22 * ((ux + 1.0) * pow(uy + 1.0, 2.0) - (ux + 1.0) * pow(uz - 1.0, 2.0)) + f23 * ((ux - 1.0) * pow(uy + 1.0, 2.0) - (ux - 1.0) * pow(uz - 1.0, 2.0)) + f24 * ((ux + 1.0) * pow(uy - 1.0, 2.0) - (ux + 1.0) * pow(uz + 1.0, 2.0)) + f25 * ((ux + 1.0) * pow(uy - 1.0, 2.0) - (ux + 1.0) * pow(uz - 1.0, 2.0)) + f26 * ((ux - 1.0) * pow(uy + 1.0, 2.0) - (ux - 1.0) * pow(uz + 1.0, 2.0)));
				T f_t14 = -r3 * (f3 * ((ux * ux) * (uy - 1.0) - (uz * uz) * (uy - 1.0)) + f4 * ((ux * ux) * (uy + 1.0) - (uz * uz) * (uy + 1.0)) + f9 * (uy * pow(ux - 1.0, 2.0) - uy * pow(uz - 1.0, 2.0)) + f10 * (uy * pow(ux + 1.0, 2.0) - uy * pow(uz + 1.0, 2.0)) + f15 * (uy * pow(ux - 1.0, 2.0) - uy * pow(uz + 1.0, 2.0)) + f16 * (uy * pow(ux + 1.0, 2.0) - uy * pow(uz - 1.0, 2.0)) + f0 * ((ux * ux) * uy - uy * (uz * uz)) - f7 * ((uz * uz) * (uy - 1.0) - pow(ux - 1.0, 2.0) * (uy - 1.0)) - f8 * ((uz * uz) * (uy + 1.0) - pow(ux + 1.0, 2.0) * (uy + 1.0)) + f11 * ((ux * ux) * (uy - 1.0) - (uy - 1.0) * pow(uz - 1.0, 2.0)) + f12 * ((ux * ux) * (uy + 1.0) - (uy + 1.0) * pow(uz + 1.0, 2.0)) - f13 * ((uz * uz) * (uy + 1.0) - pow(ux - 1.0, 2.0) * (uy + 1.0)) - f14 * ((uz * uz) * (uy - 1.0) - pow(ux + 1.0, 2.0) * (uy - 1.0)) + f17 * ((ux * ux) * (uy - 1.0) - (uy - 1.0) * pow(uz + 1.0, 2.0)) + f18 * ((ux * ux) * (uy + 1.0) - (uy + 1.0) * pow(uz - 1.0, 2.0)) + f1 * (uy * pow(ux - 1.0, 2.0) - uy * (uz * uz)) + f2 * (uy * pow(ux + 1.0, 2.0) - uy * (uz * uz)) - f5 * (uy * pow(uz - 1.0, 2.0) - (ux * ux) * uy) - f6 * (uy * pow(uz + 1.0, 2.0) - (ux * ux) * uy) + f19 * (pow(ux - 1.0, 2.0) * (uy - 1.0) - (uy - 1.0) * pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) * (uy + 1.0) - (uy + 1.0) * pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) * (uy - 1.0) - (uy - 1.0) * pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) * (uy + 1.0) - (uy + 1.0) * pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) * (uy + 1.0) - (uy + 1.0) * pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) * (uy - 1.0) - (uy - 1.0) * pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) * (uy - 1.0) - (uy - 1.0) * pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) * (uy + 1.0) - (uy + 1.0) * pow(uz + 1.0, 2.0)));
				T f_t15 = -r3 * (f5 * ((ux * ux) * (uz - 1.0) - (uy * uy) * (uz - 1.0)) + f6 * ((ux * ux) * (uz + 1.0) - (uy * uy) * (uz + 1.0)) + f7 * (uz * pow(ux - 1.0, 2.0) - uz * pow(uy - 1.0, 2.0)) + f8 * (uz * pow(ux + 1.0, 2.0) - uz * pow(uy + 1.0, 2.0)) + f13 * (uz * pow(ux - 1.0, 2.0) - uz * pow(uy + 1.0, 2.0)) + f14 * (uz * pow(ux + 1.0, 2.0) - uz * pow(uy - 1.0, 2.0)) + f0 * ((ux * ux) * uz - (uy * uy) * uz) - f9 * ((uy * uy) * (uz - 1.0) - pow(ux - 1.0, 2.0) * (uz - 1.0)) - f10 * ((uy * uy) * (uz + 1.0) - pow(ux + 1.0, 2.0) * (uz + 1.0)) + f11 * ((ux * ux) * (uz - 1.0) - pow(uy - 1.0, 2.0) * (uz - 1.0)) + f12 * ((ux * ux) * (uz + 1.0) - pow(uy + 1.0, 2.0) * (uz + 1.0)) - f15 * ((uy * uy) * (uz + 1.0) - pow(ux - 1.0, 2.0) * (uz + 1.0)) - f16 * ((uy * uy) * (uz - 1.0) - pow(ux + 1.0, 2.0) * (uz - 1.0)) + f17 * ((ux * ux) * (uz + 1.0) - pow(uy - 1.0, 2.0) * (uz + 1.0)) + f18 * ((ux * ux) * (uz - 1.0) - pow(uy + 1.0, 2.0) * (uz - 1.0)) + f1 * (uz * pow(ux - 1.0, 2.0) - (uy * uy) * uz) + f2 * (uz * pow(ux + 1.0, 2.0) - (uy * uy) * uz) - f3 * (uz * pow(uy - 1.0, 2.0) - (ux * ux) * uz) - f4 * (uz * pow(uy + 1.0, 2.0) - (ux * ux) * uz) + f19 * (pow(ux - 1.0, 2.0) * (uz - 1.0) - pow(uy - 1.0, 2.0) * (uz - 1.0)) + f20 * (pow(ux + 1.0, 2.0) * (uz + 1.0) - pow(uy + 1.0, 2.0) * (uz + 1.0)) + f21 * (pow(ux - 1.0, 2.0) * (uz + 1.0) - pow(uy - 1.0, 2.0) * (uz + 1.0)) + f22 * (pow(ux + 1.0, 2.0) * (uz - 1.0) - pow(uy + 1.0, 2.0) * (uz - 1.0)) + f23 * (pow(ux - 1.0, 2.0) * (uz - 1.0) - pow(uy + 1.0, 2.0) * (uz - 1.0)) + f24 * (pow(ux + 1.0, 2.0) * (uz + 1.0) - pow(uy - 1.0, 2.0) * (uz + 1.0)) + f25 * (pow(ux + 1.0, 2.0) * (uz - 1.0) - pow(uy - 1.0, 2.0) * (uz - 1.0)) + f26 * (pow(ux - 1.0, 2.0) * (uz + 1.0) - pow(uy + 1.0, 2.0) * (uz + 1.0)));
				T f_t16 = -r3 * (f0 * ux * uy * uz + f19 * (ux - 1.0) * (uy - 1.0) * (uz - 1.0) + f20 * (ux + 1.0) * (uy + 1.0) * (uz + 1.0) + f21 * (ux - 1.0) * (uy - 1.0) * (uz + 1.0) + f22 * (ux + 1.0) * (uy + 1.0) * (uz - 1.0) + f23 * (ux - 1.0) * (uy + 1.0) * (uz - 1.0) + f24 * (ux + 1.0) * (uy - 1.0) * (uz + 1.0) + f25 * (ux + 1.0) * (uy - 1.0) * (uz - 1.0) + f26 * (ux - 1.0) * (uy + 1.0) * (uz + 1.0) + f1 * uy * uz * (ux - 1.0) + f2 * uy * uz * (ux + 1.0) + f3 * ux * uz * (uy - 1.0) + f4 * ux * uz * (uy + 1.0) + f5 * ux * uy * (uz - 1.0) + f6 * ux * uy * (uz + 1.0) + f7 * uz * (ux - 1.0) * (uy - 1.0) + f8 * uz * (ux + 1.0) * (uy + 1.0) + f9 * uy * (ux - 1.0) * (uz - 1.0) + f10 * uy * (ux + 1.0) * (uz + 1.0) + f11 * ux * (uy - 1.0) * (uz - 1.0) + f12 * ux * (uy + 1.0) * (uz + 1.0) + f13 * uz * (ux - 1.0) * (uy + 1.0) + f14 * uz * (ux + 1.0) * (uy - 1.0) + f15 * uy * (ux - 1.0) * (uz + 1.0) + f16 * uy * (ux + 1.0) * (uz - 1.0) + f17 * ux * (uy - 1.0) * (uz + 1.0) + f18 * ux * (uy + 1.0) * (uz - 1.0));
				T f_t17 = r4 * (rho * (-1.0 / 3.0) + f0 * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) + (uy * uy) * (uz * uz)) + f19 * (pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0) + pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0) + pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0) + pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0) + pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0) + pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0) + pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0) + pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0) + pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f7 * ((uz * uz) * pow(ux - 1.0, 2.0) + (uz * uz) * pow(uy - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0)) + f8 * ((uz * uz) * pow(ux + 1.0, 2.0) + (uz * uz) * pow(uy + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0)) + f9 * ((uy * uy) * pow(ux - 1.0, 2.0) + (uy * uy) * pow(uz - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f10 * ((uy * uy) * pow(ux + 1.0, 2.0) + (uy * uy) * pow(uz + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f11 * ((ux * ux) * pow(uy - 1.0, 2.0) + (ux * ux) * pow(uz - 1.0, 2.0) + pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f12 * ((ux * ux) * pow(uy + 1.0, 2.0) + (ux * ux) * pow(uz + 1.0, 2.0) + pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f13 * ((uz * uz) * pow(ux - 1.0, 2.0) + (uz * uz) * pow(uy + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0)) + f14 * ((uz * uz) * pow(ux + 1.0, 2.0) + (uz * uz) * pow(uy - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0)) + f15 * ((uy * uy) * pow(ux - 1.0, 2.0) + (uy * uy) * pow(uz + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f16 * ((uy * uy) * pow(ux + 1.0, 2.0) + (uy * uy) * pow(uz - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f17 * ((ux * ux) * pow(uy - 1.0, 2.0) + (ux * ux) * pow(uz + 1.0, 2.0) + pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f18 * ((ux * ux) * pow(uy + 1.0, 2.0) + (ux * ux) * pow(uz - 1.0, 2.0) + pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f1 * ((uy * uy) * (uz * uz) + (uy * uy) * pow(ux - 1.0, 2.0) + (uz * uz) * pow(ux - 1.0, 2.0)) + f2 * ((uy * uy) * (uz * uz) + (uy * uy) * pow(ux + 1.0, 2.0) + (uz * uz) * pow(ux + 1.0, 2.0)) + f3 * ((ux * ux) * (uz * uz) + (ux * ux) * pow(uy - 1.0, 2.0) + (uz * uz) * pow(uy - 1.0, 2.0)) + f4 * ((ux * ux) * (uz * uz) + (ux * ux) * pow(uy + 1.0, 2.0) + (uz * uz) * pow(uy + 1.0, 2.0)) + f5 * ((ux * ux) * (uy * uy) + (ux * ux) * pow(uz - 1.0, 2.0) + (uy * uy) * pow(uz - 1.0, 2.0)) + f6 * ((ux * ux) * (uy * uy) + (ux * ux) * pow(uz + 1.0, 2.0) + (uy * uy) * pow(uz + 1.0, 2.0)));
				T f_t18 = r4 * (rho * (-1.0 / 9.0) + f0 * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) - (uy * uy) * (uz * uz)) + f19 * (pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0) - pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0) - pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0) - pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0) - pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0) - pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0) - pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0) - pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0) - pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f7 * ((uz * uz) * pow(ux - 1.0, 2.0) - (uz * uz) * pow(uy - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0)) + f8 * ((uz * uz) * pow(ux + 1.0, 2.0) - (uz * uz) * pow(uy + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0)) + f9 * ((uy * uy) * pow(ux - 1.0, 2.0) - (uy * uy) * pow(uz - 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f10 * ((uy * uy) * pow(ux + 1.0, 2.0) - (uy * uy) * pow(uz + 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f11 * ((ux * ux) * pow(uy - 1.0, 2.0) + (ux * ux) * pow(uz - 1.0, 2.0) - pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f12 * ((ux * ux) * pow(uy + 1.0, 2.0) + (ux * ux) * pow(uz + 1.0, 2.0) - pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f13 * ((uz * uz) * pow(ux - 1.0, 2.0) - (uz * uz) * pow(uy + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0)) + f14 * ((uz * uz) * pow(ux + 1.0, 2.0) - (uz * uz) * pow(uy - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0)) + f15 * ((uy * uy) * pow(ux - 1.0, 2.0) - (uy * uy) * pow(uz + 1.0, 2.0) + pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f16 * ((uy * uy) * pow(ux + 1.0, 2.0) - (uy * uy) * pow(uz - 1.0, 2.0) + pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f17 * ((ux * ux) * pow(uy - 1.0, 2.0) + (ux * ux) * pow(uz + 1.0, 2.0) - pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f18 * ((ux * ux) * pow(uy + 1.0, 2.0) + (ux * ux) * pow(uz - 1.0, 2.0) - pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f1 * (-(uy * uy) * (uz * uz) + (uy * uy) * pow(ux - 1.0, 2.0) + (uz * uz) * pow(ux - 1.0, 2.0)) + f2 * (-(uy * uy) * (uz * uz) + (uy * uy) * pow(ux + 1.0, 2.0) + (uz * uz) * pow(ux + 1.0, 2.0)) + f3 * ((ux * ux) * (uz * uz) + (ux * ux) * pow(uy - 1.0, 2.0) - (uz * uz) * pow(uy - 1.0, 2.0)) + f4 * ((ux * ux) * (uz * uz) + (ux * ux) * pow(uy + 1.0, 2.0) - (uz * uz) * pow(uy + 1.0, 2.0)) + f5 * ((ux * ux) * (uy * uy) + (ux * ux) * pow(uz - 1.0, 2.0) - (uy * uy) * pow(uz - 1.0, 2.0)) + f6 * ((ux * ux) * (uy * uy) + (ux * ux) * pow(uz + 1.0, 2.0) - (uy * uy) * pow(uz + 1.0, 2.0)));
				T f_t19 = r4 * (f0 * ((ux * ux) * (uy * uy) - (ux * ux) * (uz * uz)) - f7 * ((uz * uz) * pow(ux - 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0)) - f8 * ((uz * uz) * pow(ux + 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0)) + f9 * ((uy * uy) * pow(ux - 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f10 * ((uy * uy) * pow(ux + 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0)) - f13 * ((uz * uz) * pow(ux - 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0)) - f14 * ((uz * uz) * pow(ux + 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0)) + f15 * ((uy * uy) * pow(ux - 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f16 * ((uy * uy) * pow(ux + 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0)) - f3 * ((ux * ux) * (uz * uz) - (ux * ux) * pow(uy - 1.0, 2.0)) - f4 * ((ux * ux) * (uz * uz) - (ux * ux) * pow(uy + 1.0, 2.0)) + f5 * ((ux * ux) * (uy * uy) - (ux * ux) * pow(uz - 1.0, 2.0)) + f6 * ((ux * ux) * (uy * uy) - (ux * ux) * pow(uz + 1.0, 2.0)) + f19 * (pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f20 * (pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f21 * (pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f22 * (pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f23 * (pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f24 * (pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f25 * (pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) - pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0)) + f26 * (pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) - pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0)) + f1 * ((uy * uy) * pow(ux - 1.0, 2.0) - (uz * uz) * pow(ux - 1.0, 2.0)) + f2 * ((uy * uy) * pow(ux + 1.0, 2.0) - (uz * uz) * pow(ux + 1.0, 2.0)) + f11 * ((ux * ux) * pow(uy - 1.0, 2.0) - (ux * ux) * pow(uz - 1.0, 2.0)) + f12 * ((ux * ux) * pow(uy + 1.0, 2.0) - (ux * ux) * pow(uz + 1.0, 2.0)) + f17 * ((ux * ux) * pow(uy - 1.0, 2.0) - (ux * ux) * pow(uz + 1.0, 2.0)) + f18 * ((ux * ux) * pow(uy + 1.0, 2.0) - (ux * ux) * pow(uz - 1.0, 2.0)));
				T f_t20 = r4 * (f7 * uz * pow(ux - 1.0, 2.0) * (uy - 1.0) + f8 * uz * pow(ux + 1.0, 2.0) * (uy + 1.0) + f9 * uy * pow(ux - 1.0, 2.0) * (uz - 1.0) + f10 * uy * pow(ux + 1.0, 2.0) * (uz + 1.0) + f11 * (ux * ux) * (uy - 1.0) * (uz - 1.0) + f12 * (ux * ux) * (uy + 1.0) * (uz + 1.0) + f13 * uz * pow(ux - 1.0, 2.0) * (uy + 1.0) + f14 * uz * pow(ux + 1.0, 2.0) * (uy - 1.0) + f15 * uy * pow(ux - 1.0, 2.0) * (uz + 1.0) + f16 * uy * pow(ux + 1.0, 2.0) * (uz - 1.0) + f17 * (ux * ux) * (uy - 1.0) * (uz + 1.0) + f18 * (ux * ux) * (uy + 1.0) * (uz - 1.0) + f0 * (ux * ux) * uy * uz + f19 * pow(ux - 1.0, 2.0) * (uy - 1.0) * (uz - 1.0) + f20 * pow(ux + 1.0, 2.0) * (uy + 1.0) * (uz + 1.0) + f21 * pow(ux - 1.0, 2.0) * (uy - 1.0) * (uz + 1.0) + f22 * pow(ux + 1.0, 2.0) * (uy + 1.0) * (uz - 1.0) + f23 * pow(ux - 1.0, 2.0) * (uy + 1.0) * (uz - 1.0) + f24 * pow(ux + 1.0, 2.0) * (uy - 1.0) * (uz + 1.0) + f25 * pow(ux + 1.0, 2.0) * (uy - 1.0) * (uz - 1.0) + f26 * pow(ux - 1.0, 2.0) * (uy + 1.0) * (uz + 1.0) + f1 * uy * uz * pow(ux - 1.0, 2.0) + f2 * uy * uz * pow(ux + 1.0, 2.0) + f3 * (ux * ux) * uz * (uy - 1.0) + f4 * (ux * ux) * uz * (uy + 1.0) + f5 * (ux * ux) * uy * (uz - 1.0) + f6 * (ux * ux) * uy * (uz + 1.0));
				T f_t21 = r4 * (f7 * uz * (ux - 1.0) * pow(uy - 1.0, 2.0) + f8 * uz * (ux + 1.0) * pow(uy + 1.0, 2.0) + f9 * (uy * uy) * (ux - 1.0) * (uz - 1.0) + f10 * (uy * uy) * (ux + 1.0) * (uz + 1.0) + f11 * ux * pow(uy - 1.0, 2.0) * (uz - 1.0) + f12 * ux * pow(uy + 1.0, 2.0) * (uz + 1.0) + f13 * uz * (ux - 1.0) * pow(uy + 1.0, 2.0) + f14 * uz * (ux + 1.0) * pow(uy - 1.0, 2.0) + f15 * (uy * uy) * (ux - 1.0) * (uz + 1.0) + f16 * (uy * uy) * (ux + 1.0) * (uz - 1.0) + f17 * ux * pow(uy - 1.0, 2.0) * (uz + 1.0) + f18 * ux * pow(uy + 1.0, 2.0) * (uz - 1.0) + f0 * ux * (uy * uy) * uz + f19 * (ux - 1.0) * pow(uy - 1.0, 2.0) * (uz - 1.0) + f20 * (ux + 1.0) * pow(uy + 1.0, 2.0) * (uz + 1.0) + f21 * (ux - 1.0) * pow(uy - 1.0, 2.0) * (uz + 1.0) + f22 * (ux + 1.0) * pow(uy + 1.0, 2.0) * (uz - 1.0) + f23 * (ux - 1.0) * pow(uy + 1.0, 2.0) * (uz - 1.0) + f24 * (ux + 1.0) * pow(uy - 1.0, 2.0) * (uz + 1.0) + f25 * (ux + 1.0) * pow(uy - 1.0, 2.0) * (uz - 1.0) + f26 * (ux - 1.0) * pow(uy + 1.0, 2.0) * (uz + 1.0) + f1 * (uy * uy) * uz * (ux - 1.0) + f2 * (uy * uy) * uz * (ux + 1.0) + f3 * ux * uz * pow(uy - 1.0, 2.0) + f4 * ux * uz * pow(uy + 1.0, 2.0) + f5 * ux * (uy * uy) * (uz - 1.0) + f6 * ux * (uy * uy) * (uz + 1.0));
				T f_t22 = r4 * (f7 * (uz * uz) * (ux - 1.0) * (uy - 1.0) + f8 * (uz * uz) * (ux + 1.0) * (uy + 1.0) + f9 * uy * (ux - 1.0) * pow(uz - 1.0, 2.0) + f10 * uy * (ux + 1.0) * pow(uz + 1.0, 2.0) + f11 * ux * (uy - 1.0) * pow(uz - 1.0, 2.0) + f12 * ux * (uy + 1.0) * pow(uz + 1.0, 2.0) + f13 * (uz * uz) * (ux - 1.0) * (uy + 1.0) + f14 * (uz * uz) * (ux + 1.0) * (uy - 1.0) + f15 * uy * (ux - 1.0) * pow(uz + 1.0, 2.0) + f16 * uy * (ux + 1.0) * pow(uz - 1.0, 2.0) + f17 * ux * (uy - 1.0) * pow(uz + 1.0, 2.0) + f18 * ux * (uy + 1.0) * pow(uz - 1.0, 2.0) + f0 * ux * uy * (uz * uz) + f19 * (ux - 1.0) * (uy - 1.0) * pow(uz - 1.0, 2.0) + f20 * (ux + 1.0) * (uy + 1.0) * pow(uz + 1.0, 2.0) + f21 * (ux - 1.0) * (uy - 1.0) * pow(uz + 1.0, 2.0) + f22 * (ux + 1.0) * (uy + 1.0) * pow(uz - 1.0, 2.0) + f23 * (ux - 1.0) * (uy + 1.0) * pow(uz - 1.0, 2.0) + f24 * (ux + 1.0) * (uy - 1.0) * pow(uz + 1.0, 2.0) + f25 * (ux + 1.0) * (uy - 1.0) * pow(uz - 1.0, 2.0) + f26 * (ux - 1.0) * (uy + 1.0) * pow(uz + 1.0, 2.0) + f1 * uy * (uz * uz) * (ux - 1.0) + f2 * uy * (uz * uz) * (ux + 1.0) + f3 * ux * (uz * uz) * (uy - 1.0) + f4 * ux * (uz * uz) * (uy + 1.0) + f5 * ux * uy * pow(uz - 1.0, 2.0) + f6 * ux * uy * pow(uz + 1.0, 2.0));
				T f_t23 = -r5 * (f19 * (ux - 1.0) * pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f20 * (ux + 1.0) * pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f21 * (ux - 1.0) * pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f22 * (ux + 1.0) * pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f23 * (ux - 1.0) * pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f24 * (ux + 1.0) * pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f25 * (ux + 1.0) * pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f26 * (ux - 1.0) * pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f1 * (uy * uy) * (uz * uz) * (ux - 1.0) + f2 * (uy * uy) * (uz * uz) * (ux + 1.0) + f3 * ux * (uz * uz) * pow(uy - 1.0, 2.0) + f4 * ux * (uz * uz) * pow(uy + 1.0, 2.0) + f5 * ux * (uy * uy) * pow(uz - 1.0, 2.0) + f6 * ux * (uy * uy) * pow(uz + 1.0, 2.0) + f7 * (uz * uz) * (ux - 1.0) * pow(uy - 1.0, 2.0) + f8 * (uz * uz) * (ux + 1.0) * pow(uy + 1.0, 2.0) + f9 * (uy * uy) * (ux - 1.0) * pow(uz - 1.0, 2.0) + f10 * (uy * uy) * (ux + 1.0) * pow(uz + 1.0, 2.0) + f11 * ux * pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f12 * ux * pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f13 * (uz * uz) * (ux - 1.0) * pow(uy + 1.0, 2.0) + f14 * (uz * uz) * (ux + 1.0) * pow(uy - 1.0, 2.0) + f15 * (uy * uy) * (ux - 1.0) * pow(uz + 1.0, 2.0) + f16 * (uy * uy) * (ux + 1.0) * pow(uz - 1.0, 2.0) + f17 * ux * pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f18 * ux * pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f0 * ux * (uy * uy) * (uz * uz));
				T f_t24 = -r5 * (f19 * pow(ux - 1.0, 2.0) * (uy - 1.0) * pow(uz - 1.0, 2.0) + f20 * pow(ux + 1.0, 2.0) * (uy + 1.0) * pow(uz + 1.0, 2.0) + f21 * pow(ux - 1.0, 2.0) * (uy - 1.0) * pow(uz + 1.0, 2.0) + f22 * pow(ux + 1.0, 2.0) * (uy + 1.0) * pow(uz - 1.0, 2.0) + f23 * pow(ux - 1.0, 2.0) * (uy + 1.0) * pow(uz - 1.0, 2.0) + f24 * pow(ux + 1.0, 2.0) * (uy - 1.0) * pow(uz + 1.0, 2.0) + f25 * pow(ux + 1.0, 2.0) * (uy - 1.0) * pow(uz - 1.0, 2.0) + f26 * pow(ux - 1.0, 2.0) * (uy + 1.0) * pow(uz + 1.0, 2.0) + f1 * uy * (uz * uz) * pow(ux - 1.0, 2.0) + f2 * uy * (uz * uz) * pow(ux + 1.0, 2.0) + f3 * (ux * ux) * (uz * uz) * (uy - 1.0) + f4 * (ux * ux) * (uz * uz) * (uy + 1.0) + f5 * (ux * ux) * uy * pow(uz - 1.0, 2.0) + f6 * (ux * ux) * uy * pow(uz + 1.0, 2.0) + f7 * (uz * uz) * pow(ux - 1.0, 2.0) * (uy - 1.0) + f8 * (uz * uz) * pow(ux + 1.0, 2.0) * (uy + 1.0) + f9 * uy * pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f10 * uy * pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f11 * (ux * ux) * (uy - 1.0) * pow(uz - 1.0, 2.0) + f12 * (ux * ux) * (uy + 1.0) * pow(uz + 1.0, 2.0) + f13 * (uz * uz) * pow(ux - 1.0, 2.0) * (uy + 1.0) + f14 * (uz * uz) * pow(ux + 1.0, 2.0) * (uy - 1.0) + f15 * uy * pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f16 * uy * pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f17 * (ux * ux) * (uy - 1.0) * pow(uz + 1.0, 2.0) + f18 * (ux * ux) * (uy + 1.0) * pow(uz - 1.0, 2.0) + f0 * (ux * ux) * uy * (uz * uz));
				T f_t25 = -r5 * (f19 * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) * (uz - 1.0) + f20 * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) * (uz + 1.0) + f21 * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) * (uz + 1.0) + f22 * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) * (uz - 1.0) + f23 * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) * (uz - 1.0) + f24 * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) * (uz + 1.0) + f25 * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) * (uz - 1.0) + f26 * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) * (uz + 1.0) + f1 * (uy * uy) * uz * pow(ux - 1.0, 2.0) + f2 * (uy * uy) * uz * pow(ux + 1.0, 2.0) + f3 * (ux * ux) * uz * pow(uy - 1.0, 2.0) + f4 * (ux * ux) * uz * pow(uy + 1.0, 2.0) + f5 * (ux * ux) * (uy * uy) * (uz - 1.0) + f6 * (ux * ux) * (uy * uy) * (uz + 1.0) + f7 * uz * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + f8 * uz * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + f9 * (uy * uy) * pow(ux - 1.0, 2.0) * (uz - 1.0) + f10 * (uy * uy) * pow(ux + 1.0, 2.0) * (uz + 1.0) + f11 * (ux * ux) * pow(uy - 1.0, 2.0) * (uz - 1.0) + f12 * (ux * ux) * pow(uy + 1.0, 2.0) * (uz + 1.0) + f13 * uz * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + f14 * uz * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + f15 * (uy * uy) * pow(ux - 1.0, 2.0) * (uz + 1.0) + f16 * (uy * uy) * pow(ux + 1.0, 2.0) * (uz - 1.0) + f17 * (ux * ux) * pow(uy - 1.0, 2.0) * (uz + 1.0) + f18 * (ux * ux) * pow(uy + 1.0, 2.0) * (uz - 1.0) + f0 * (ux * ux) * (uy * uy) * uz);
				T f_t26 = r6 * (rho * (-1.0 / 2.7E+1) + f7 * (uz * uz) * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) + f8 * (uz * uz) * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) + f9 * (uy * uy) * pow(ux - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f10 * (uy * uy) * pow(ux + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f11 * (ux * ux) * pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f12 * (ux * ux) * pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f13 * (uz * uz) * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) + f14 * (uz * uz) * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) + f15 * (uy * uy) * pow(ux - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f16 * (uy * uy) * pow(ux + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f17 * (ux * ux) * pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f18 * (ux * ux) * pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f0 * (ux * ux) * (uy * uy) * (uz * uz) + f19 * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f20 * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f21 * pow(ux - 1.0, 2.0) * pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f22 * pow(ux + 1.0, 2.0) * pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f23 * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) * pow(uz - 1.0, 2.0) + f24 * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) * pow(uz + 1.0, 2.0) + f25 * pow(ux + 1.0, 2.0) * pow(uy - 1.0, 2.0) * pow(uz - 1.0, 2.0) + f26 * pow(ux - 1.0, 2.0) * pow(uy + 1.0, 2.0) * pow(uz + 1.0, 2.0) + f1 * (uy * uy) * (uz * uz) * pow(ux - 1.0, 2.0) + f2 * (uy * uy) * (uz * uz) * pow(ux + 1.0, 2.0) + f3 * (ux * ux) * (uz * uz) * pow(uy - 1.0, 2.0) + f4 * (ux * ux) * (uz * uz) * pow(uy + 1.0, 2.0) + f5 * (ux * ux) * (uy * uy) * pow(uz - 1.0, 2.0) + f6 * (ux * ux) * (uy * uy) * pow(uz + 1.0, 2.0));

				T Omega_0 = f_t26 + f_t23 * ux * 2.0 + f_t24 * uy * 2.0 + f_t25 * uz * 2.0 + f_t10 * (ux * -2.0 + ux * (uy * uy) + ux * (uz * uz)) + f_t11 * (uy * -2.0 + (ux * ux) * uy + uy * (uz * uz)) + f_t12 * (uz * -2.0 + (ux * ux) * uz + (uy * uy) * uz) - f_t13 * (ux * (uy * uy) - ux * (uz * uz)) - f_t14 * ((ux * ux) * uy - uy * (uz * uz)) - f_t15 * ((ux * ux) * uz - (uy * uy) * uz) - f_t19 * ((uy * uy) / 2.0 - (uz * uz) / 2.0) - f_t4 * (ux * uy * 4.0 - ux * uy * (uz * uz) * 4.0) - f_t5 * (ux * uz * 4.0 - ux * (uy * uy) * uz * 4.0) - f_t6 * (uy * uz * 4.0 - (ux * ux) * uy * uz * 4.0) - f_t0 * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) + (uy * uy) * (uz * uz) - ux * ux - uy * uy - uz * uz - (ux * ux) * (uy * uy) * (uz * uz) + 1.0) + f_t17 * ((ux * ux) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0) + f_t18 * ((ux * ux) * (-1.0 / 2.0) + (uy * uy) / 4.0 + (uz * uz) / 4.0) + f_t1 * (ux * 2.0 - ux * (uy * uy) * 2.0 - ux * (uz * uz) * 2.0 + ux * (uy * uy) * (uz * uz) * 2.0) + f_t2 * (uy * 2.0 - (ux * ux) * uy * 2.0 - uy * (uz * uz) * 2.0 + (ux * ux) * uy * (uz * uz) * 2.0) + f_t3 * (uz * 2.0 - (ux * ux) * uz * 2.0 - (uy * uy) * uz * 2.0 + (ux * ux) * (uy * uy) * uz * 2.0) + f_t7 * (((ux * ux) * (uy * uy)) / 3.0 - (ux * ux) * (uz * uz) * (2.0 / 3.0) + ((uy * uy) * (uz * uz)) / 3.0 + (ux * ux) / 3.0 - (uy * uy) * (2.0 / 3.0) + (uz * uz) / 3.0) + f_t8 * ((ux * ux) * (uy * uy) * (-2.0 / 3.0) + ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 3.0 + (ux * ux) / 3.0 + (uy * uy) / 3.0 - (uz * uz) * (2.0 / 3.0)) + f_t9 * (((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 3.0 - (ux * ux) * (2.0 / 3.0) - (uy * uy) * (2.0 / 3.0) - (uz * uz) * (2.0 / 3.0) + 1.0) + f_t22 * ux * uy * 4.0 + f_t21 * ux * uz * 4.0 + f_t20 * uy * uz * 4.0 + f_t16 * ux * uy * uz * 8.0;
				T Omega_1 = f_t26 * (-1.0 / 2.0) - f_t6 * ((ux * ux) * uy * uz * 2.0 + ux * uy * uz * 2.0) - f_t7 * (ux / 6.0 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 3.0 + (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t8 * (ux / 6.0 - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 3.0 + (ux * (uz * uz)) / 6.0 + (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t9 * (ux * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t16 * (uy * uz * 2.0 + ux * uy * uz * 4.0) - f_t24 * uy - f_t25 * uz + f_t19 * ((uy * uy) / 4.0 - (uz * uz) / 4.0) - f_t17 * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0) + f_t18 * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 4.0) - f_t11 * (uy * (-1.0 / 2.0) + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) + f_t14 * (uy / 2.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) - f_t12 * (uz * (-1.0 / 2.0) + (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) + f_t15 * (uz / 2.0 + (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) - f_t23 * (ux + 1.0 / 2.0) + f_t4 * (uy + ux * uy * 2.0 - uy * (uz * uz) - ux * uy * (uz * uz) * 2.0) + f_t5 * (uz + ux * uz * 2.0 - (uy * uy) * uz - ux * (uy * uy) * uz * 2.0) - f_t10 * (-ux + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) + f_t2 * (ux * uy + (ux * ux) * uy - ux * uy * (uz * uz) - (ux * ux) * uy * (uz * uz)) + f_t3 * (ux * uz + (ux * ux) * uz - ux * (uy * uy) * uz - (ux * ux) * (uy * uy) * uz) - f_t1 * (ux + ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) - (uy * uy) / 2.0 - (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) + 1.0 / 2.0) + f_t13 * ((ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 - (uz * uz) / 4.0) - f_t22 * (uy + ux * uy * 2.0) - f_t21 * (uz + ux * uz * 2.0) - f_t0 * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t20 * uy * uz * 2.0;
				T Omega_2 = f_t26 * (-1.0 / 2.0) - f_t6 * ((ux * ux) * uy * uz * 2.0 - ux * uy * uz * 2.0) + f_t7 * (ux / 6.0 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 3.0 - ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 3.0 - (ux * ux) / 6.0 + (uy * uy) / 6.0 + (uz * uz) / 6.0 - 1.0 / 6.0) + f_t8 * (ux / 6.0 + ((ux * ux) * (uy * uy)) / 3.0 - ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 3.0 + (ux * (uz * uz)) / 6.0 - (ux * ux) / 6.0 + (uy * uy) / 6.0 + (uz * uz) / 6.0 - 1.0 / 6.0) - f_t9 * (ux / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + f_t16 * (uy * uz * 2.0 - ux * uy * uz * 4.0) - f_t24 * uy - f_t25 * uz + f_t10 * (ux - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) + f_t19 * ((uy * uy) / 4.0 - (uz * uz) / 4.0) - f_t17 * (ux * (-1.0 / 4.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0) - f_t18 * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0) + f_t11 * (uy / 2.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) + f_t14 * (uy / 2.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) + f_t12 * (uz / 2.0 + (ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) + f_t15 * (uz / 2.0 - (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) - f_t23 * (ux - 1.0 / 2.0) - f_t4 * (uy - ux * uy * 2.0 - uy * (uz * uz) + ux * uy * (uz * uz) * 2.0) - f_t5 * (uz - ux * uz * 2.0 - (uy * uy) * uz + ux * (uy * uy) * uz * 2.0) - f_t2 * (ux * uy - (ux * ux) * uy - ux * uy * (uz * uz) + (ux * ux) * uy * (uz * uz)) - f_t3 * (ux * uz - (ux * ux) * uz - ux * (uy * uy) * uz + (ux * ux) * (uy * uy) * uz) - f_t1 * (ux - ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) + (uy * uy) / 2.0 + (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) - 1.0 / 2.0) + f_t13 * ((ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 - (uy * uy) / 4.0 + (uz * uz) / 4.0) + f_t22 * (uy - ux * uy * 2.0) + f_t21 * (uz - ux * uz * 2.0) + f_t0 * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 - (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t20 * uy * uz * 2.0;
				T Omega_3 = f_t26 * (-1.0 / 2.0) - f_t5 * (ux * (uy * uy) * uz * 2.0 + ux * uy * uz * 2.0) - f_t7 * (uy * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uy) / 6.0 + (uy * (uz * uz)) / 6.0 + (ux * ux) / 3.0 - (uy * uy) / 3.0 + (uz * uz) / 3.0 - 1.0 / 3.0) - f_t8 * (uy / 6.0 - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 3.0 + (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 + (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t9 * (uy * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uy) / 6.0 + (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - f_t16 * (ux * uz * 2.0 + ux * uy * uz * 4.0) - f_t23 * ux - f_t25 * uz + f_t19 * (uy / 4.0 + (uy * uy) / 4.0 - (uz * uz) / 4.0 + 1.0 / 4.0) - f_t18 * (uy / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uy / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) - f_t10 * (ux * (-1.0 / 2.0) + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) + f_t13 * (ux / 2.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) - f_t12 * (uz * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) - f_t15 * (uz / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) - f_t24 * (uy + 1.0 / 2.0) + f_t4 * (ux + ux * uy * 2.0 - ux * (uz * uz) - ux * uy * (uz * uz) * 2.0) + f_t6 * (uz + uy * uz * 2.0 - (ux * ux) * uz - (ux * ux) * uy * uz * 2.0) - f_t11 * (-uy + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) + f_t1 * (ux * uy + ux * (uy * uy) - ux * uy * (uz * uz) - ux * (uy * uy) * (uz * uz)) + f_t3 * (uy * uz + (uy * uy) * uz - (ux * ux) * uy * uz - (ux * ux) * (uy * uy) * uz) - f_t2 * (uy + ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) - (ux * ux) / 2.0 - (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) + 1.0 / 2.0) + f_t14 * (((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 - (uz * uz) / 4.0) - f_t22 * (ux + ux * uy * 2.0) - f_t20 * (uz + uy * uz * 2.0) - f_t0 * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t21 * ux * uz * 2.0;
				T Omega_4 = f_t26 * (-1.0 / 2.0) - f_t5 * (ux * (uy * uy) * uz * 2.0 - ux * uy * uz * 2.0) - f_t7 * (uy / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 6.0 - (uy * (uz * uz)) / 6.0 + (ux * ux) / 3.0 - (uy * uy) / 3.0 + (uz * uz) / 3.0 - 1.0 / 3.0) + f_t8 * (uy / 6.0 + ((ux * ux) * (uy * uy)) / 3.0 - ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 3.0 + (uy * (uz * uz)) / 6.0 + (ux * ux) / 6.0 - (uy * uy) / 6.0 + (uz * uz) / 6.0 - 1.0 / 6.0) - f_t9 * (uy / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 6.0 - (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + f_t16 * (ux * uz * 2.0 - ux * uy * uz * 4.0) - f_t23 * ux - f_t25 * uz + f_t11 * (uy - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) - f_t19 * (uy / 4.0 - (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 4.0) - f_t18 * (uy * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uy * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) + f_t10 * (ux / 2.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) + f_t13 * (ux / 2.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) + f_t12 * (uz / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) - f_t15 * (uz / 2.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) - f_t24 * (uy - 1.0 / 2.0) - f_t4 * (ux - ux * uy * 2.0 - ux * (uz * uz) + ux * uy * (uz * uz) * 2.0) - f_t6 * (uz - uy * uz * 2.0 - (ux * ux) * uz + (ux * ux) * uy * uz * 2.0) - f_t1 * (ux * uy - ux * (uy * uy) - ux * uy * (uz * uz) + ux * (uy * uy) * (uz * uz)) - f_t3 * (uy * uz - (uy * uy) * uz - (ux * ux) * uy * uz + (ux * ux) * (uy * uy) * uz) - f_t2 * (uy - ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) + (ux * ux) / 2.0 + (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) - 1.0 / 2.0) + f_t14 * (((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 - (ux * ux) / 4.0 + (uz * uz) / 4.0) + f_t22 * (ux - ux * uy * 2.0) + f_t20 * (uz - uy * uz * 2.0) + f_t0 * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 - (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t21 * ux * uz * 2.0;
				T Omega_5 = f_t26 * (-1.0 / 2.0) - f_t4 * (ux * uy * (uz * uz) * 2.0 + ux * uy * uz * 2.0) - f_t8 * (uz * (-1.0 / 3.0) - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 6.0 + (ux * ux) / 3.0 + (uy * uy) / 3.0 - (uz * uz) / 3.0 - 1.0 / 3.0) - f_t7 * (uz / 6.0 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 3.0 + ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 + (uz * uz) / 6.0 + 1.0 / 6.0) - f_t9 * (uz * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) - f_t16 * (ux * uy * 2.0 + ux * uy * uz * 4.0) - f_t23 * ux - f_t24 * uy - f_t19 * (uz / 4.0 - (uy * uy) / 4.0 + (uz * uz) / 4.0 + 1.0 / 4.0) - f_t18 * (uz / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uz / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) - f_t10 * (ux * (-1.0 / 2.0) + (ux * uz) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) - f_t13 * (ux / 2.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) - f_t11 * (uy * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) - f_t14 * (uy / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) - f_t25 * (uz + 1.0 / 2.0) + f_t5 * (ux + ux * uz * 2.0 - ux * (uy * uy) - ux * (uy * uy) * uz * 2.0) + f_t6 * (uy + uy * uz * 2.0 - (ux * ux) * uy - (ux * ux) * uy * uz * 2.0) - f_t12 * (-uz + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) + f_t1 * (ux * uz + ux * (uz * uz) - ux * (uy * uy) * uz - ux * (uy * uy) * (uz * uz)) + f_t2 * (uy * uz + uy * (uz * uz) - (ux * ux) * uy * uz - (ux * ux) * uy * (uz * uz)) - f_t3 * (uz + ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz - (ux * ux) / 2.0 - (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz + 1.0 / 2.0) + f_t15 * (((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0) - f_t21 * (ux + ux * uz * 2.0) - f_t20 * (uy + uy * uz * 2.0) - f_t0 * (uz / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t22 * ux * uy * 2.0;
				T Omega_6 = f_t26 * (-1.0 / 2.0) - f_t4 * (ux * uy * (uz * uz) * 2.0 - ux * uy * uz * 2.0) - f_t8 * (uz / 3.0 - ((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 6.0 + (ux * ux) / 3.0 + (uy * uy) / 3.0 - (uz * uz) / 3.0 - 1.0 / 3.0) + f_t7 * (uz / 6.0 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 3.0 - ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 3.0 + ((uy * uy) * uz) / 6.0 + (ux * ux) / 6.0 + (uy * uy) / 6.0 - (uz * uz) / 6.0 - 1.0 / 6.0) - f_t9 * (uz / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) + f_t16 * (ux * uy * 2.0 - ux * uy * uz * 4.0) - f_t23 * ux - f_t24 * uy + f_t12 * (uz - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) + f_t19 * (uz / 4.0 + (uy * uy) / 4.0 - (uz * uz) / 4.0 - 1.0 / 4.0) - f_t18 * (uz * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t17 * (uz * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0) + f_t10 * (ux / 2.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) - f_t13 * (ux / 2.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) + f_t11 * (uy / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) - f_t14 * (uy / 2.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) - f_t25 * (uz - 1.0 / 2.0) - f_t5 * (ux - ux * uz * 2.0 - ux * (uy * uy) + ux * (uy * uy) * uz * 2.0) - f_t6 * (uy - uy * uz * 2.0 - (ux * ux) * uy + (ux * ux) * uy * uz * 2.0) - f_t1 * (ux * uz - ux * (uz * uz) - ux * (uy * uy) * uz + ux * (uy * uy) * (uz * uz)) - f_t2 * (uy * uz - uy * (uz * uz) - (ux * ux) * uy * uz + (ux * ux) * uy * (uz * uz)) - f_t3 * (uz - ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz + (ux * ux) / 2.0 + (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz - 1.0 / 2.0) + f_t15 * (((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0) + f_t21 * (ux - ux * uz * 2.0) + f_t20 * (uy - uy * uz * 2.0) + f_t0 * (uz / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0) - f_t22 * ux * uy * 2.0;
				T Omega_7 = f_t26 / 4.0 + f_t12 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) - f_t15 * ((ux * uz) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) + f_t22 * (ux / 2.0 + uy / 2.0 + ux * uy + 1.0 / 4.0) + (f_t25 * uz) / 2.0 + f_t21 * (uz / 2.0 + ux * uz) + f_t20 * (uz / 2.0 + uy * uz) + f_t7 * (ux / 6.0 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 6.0 - (uy * uy) / 1.2E+1) - f_t8 * (ux / 1.2E+1 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 + ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) + f_t9 * (ux * (-1.0 / 1.2E+1) - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t16 * (uz / 2.0 + ux * uz + uy * uz + ux * uy * uz * 2.0) - f_t19 * (uy / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) - f_t2 * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t3 * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (ux / 2.0 + uy / 2.0 + ux * uy - (ux * (uz * uz)) / 2.0 - (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 - ux * uy * (uz * uz) + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) + f_t10 * (ux * (-1.0 / 4.0) + uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t11 * (ux / 8.0 - uy / 4.0 + (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t13 * (ux / 4.0 + uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t14 * (ux / 8.0 + uy / 4.0 + (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t6 * ((ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) * uy * uz + ux * uy * uz) + f_t5 * ((uy * uz) / 2.0 + ((uy * uy) * uz) / 2.0 + ux * (uy * uy) * uz + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t24 * (uy / 2.0 + 1.0 / 4.0);
				T Omega_8 = f_t26 / 4.0 - f_t12 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) + f_t15 * ((ux * uz) / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) - f_t22 * (ux / 2.0 + uy / 2.0 - ux * uy - 1.0 / 4.0) + (f_t25 * uz) / 2.0 - f_t21 * (uz / 2.0 - ux * uz) - f_t20 * (uz / 2.0 - uy * uz) - f_t7 * (ux / 6.0 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 6.0 + (uy * uy) / 1.2E+1) + f_t8 * (ux / 1.2E+1 + uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 + ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t9 * (ux / 1.2E+1 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t16 * (uz / 2.0 - ux * uz - uy * uz + ux * uy * uz * 2.0) + f_t19 * (uy / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t17 * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 - uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t2 * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t3 * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (ux / 2.0 + uy / 2.0 - ux * uy - (ux * (uz * uz)) / 2.0 - (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 + ux * uy * (uz * uz) - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 - uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t11 * (ux / 8.0 - uy / 4.0 - (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t13 * (ux / 4.0 + uy / 8.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t14 * (ux / 8.0 + uy / 4.0 - (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t6 * ((ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) * uy * uz - ux * uy * uz) + f_t5 * ((uy * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + ux * (uy * uy) * uz - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t24 * (uy / 2.0 - 1.0 / 4.0);
				T Omega_9 = f_t26 / 4.0 + f_t11 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) - f_t14 * ((ux * uy) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) + f_t21 * (ux / 2.0 + uz / 2.0 + ux * uz + 1.0 / 4.0) + (f_t24 * uy) / 2.0 + f_t22 * (uy / 2.0 + ux * uy) + f_t20 * (uy / 2.0 + uy * uz) - f_t7 * (ux / 1.2E+1 + uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) + f_t8 * (ux / 6.0 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 6.0 - (uz * uz) / 1.2E+1) + f_t9 * (ux * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (uy / 2.0 + ux * uy + uy * uz + ux * uy * uz * 2.0) + f_t19 * (uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) - f_t3 * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t1 * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t2 * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (ux / 2.0 + uz / 2.0 + ux * uz - (ux * (uy * uy)) / 2.0 - ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 - ux * (uy * uy) * uz + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) + f_t10 * (ux * (-1.0 / 4.0) + uz / 8.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (ux / 8.0 - uz / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t13 * (ux / 4.0 + uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t15 * (ux / 8.0 + uz / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t6 * ((ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (ux * ux) * uy * uz + ux * uy * uz) + f_t4 * ((uy * uz) / 2.0 + (uy * (uz * uz)) / 2.0 + ux * uy * (uz * uz) + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
				T Omega_10 = f_t26 / 4.0 - f_t11 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) + f_t14 * ((ux * uy) / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) - f_t21 * (ux / 2.0 + uz / 2.0 - ux * uz - 1.0 / 4.0) + (f_t24 * uy) / 2.0 - f_t22 * (uy / 2.0 - ux * uy) - f_t20 * (uy / 2.0 - uy * uz) + f_t7 * (ux / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t8 * (ux / 6.0 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (ux / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (uy / 2.0 - ux * uy - uy * uz + ux * uy * uz * 2.0) - f_t19 * (uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) - f_t17 * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t3 * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t1 * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t2 * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (ux / 2.0 + uz / 2.0 - ux * uz - (ux * (uy * uy)) / 2.0 - ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 + ux * (uy * uy) * uz - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 - uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (ux / 8.0 - uz / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t13 * (ux / 4.0 + uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) - f_t15 * (ux / 8.0 + uz / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t6 * ((ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 + (ux * ux) * uy * uz - ux * uy * uz) + f_t4 * ((uy * uz) / 2.0 - (uy * (uz * uz)) / 2.0 + ux * uy * (uz * uz) - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
				T Omega_11 = f_t26 / 4.0 + f_t10 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) - f_t13 * ((ux * uy) / 4.0 - (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) + f_t20 * (uy / 2.0 + uz / 2.0 + uy * uz + 1.0 / 4.0) - f_t19 * (uy / 8.0 - uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 + f_t22 * (ux / 2.0 + ux * uy) + f_t21 * (ux / 2.0 + ux * uz) + f_t7 * (uy * (-1.0 / 1.2E+1) + uz / 6.0 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 6.0 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 + (uz * uz) / 6.0) + f_t8 * (uy / 6.0 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 6.0 - (uz * uz) / 1.2E+1) + f_t9 * (uy * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (ux / 2.0 + ux * uy + ux * uz + ux * uy * uz * 2.0) + f_t17 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0) + f_t18 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0) - f_t3 * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t2 * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t6 * (uy / 2.0 + uz / 2.0 + uy * uz - ((ux * ux) * uy) / 2.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 - (ux * ux) * uy * uz + 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) + f_t11 * (uy * (-1.0 / 4.0) + uz / 8.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (uy / 8.0 - uz / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t14 * (uy / 4.0 + uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t15 * (uy / 8.0 + uz / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 + 1.0 / 8.0) + f_t5 * ((ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + ux * (uy * uy) * uz + ux * uy * uz) + f_t4 * ((ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 + ux * uy * (uz * uz) + ux * uy * uz) + f_t24 * (uy / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
				T Omega_12 = f_t26 / 4.0 - f_t10 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) + f_t13 * ((ux * uy) / 4.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) - f_t20 * (uy / 2.0 + uz / 2.0 - uy * uz - 1.0 / 4.0) + f_t19 * (uy / 8.0 - uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 - f_t22 * (ux / 2.0 - ux * uy) - f_t21 * (ux / 2.0 - ux * uz) + f_t7 * (uy / 1.2E+1 - uz / 6.0 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 6.0 - (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 + (uz * uz) / 6.0) - f_t8 * (uy / 6.0 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (uy / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) + f_t16 * (ux / 2.0 - ux * uy - ux * uz + ux * uy * uz * 2.0) - f_t17 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 8.0) - f_t18 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 - 1.0 / 8.0) - f_t3 * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - f_t2 * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t6 * (uy / 2.0 + uz / 2.0 - uy * uz - ((ux * ux) * uy) / 2.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 + (ux * ux) * uy * uz - 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t11 * (uy / 4.0 - uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t12 * (uy / 8.0 - uz / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t14 * (uy / 4.0 + uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) + f_t15 * (uy / 8.0 + uz / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 - 1.0 / 8.0) + f_t5 * ((ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 + ux * (uy * uy) * uz - ux * uy * uz) + f_t4 * ((ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 + ux * uy * (uz * uz) - ux * uy * uz) + f_t24 * (uy / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
				T Omega_13 = f_t26 / 4.0 + f_t12 * ((ux * uz) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) - f_t15 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) - f_t22 * (ux / 2.0 - uy / 2.0 - ux * uy + 1.0 / 4.0) + (f_t25 * uz) / 2.0 + f_t21 * (uz / 2.0 + ux * uz) - f_t20 * (uz / 2.0 - uy * uz) + f_t7 * (ux / 6.0 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 - (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 6.0 - (uy * uy) / 1.2E+1) - f_t8 * (ux / 1.2E+1 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 - ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) - f_t9 * (ux / 1.2E+1 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) - f_t16 * (uz / 2.0 + ux * uz - uy * uz - ux * uy * uz * 2.0) + f_t19 * (uy / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t17 * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t2 * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t3 * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (ux / 2.0 - uy / 2.0 - ux * uy - (ux * (uz * uz)) / 2.0 + (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 + ux * uy * (uz * uz) + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t11 * (ux / 8.0 + uy / 4.0 - (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t13 * (ux / 4.0 - uy / 8.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t14 * (ux / 8.0 - uy / 4.0 - (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t6 * ((ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) * uy * uz - ux * uy * uz) - f_t5 * ((uy * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - ux * (uy * uy) * uz + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t24 * (uy / 2.0 - 1.0 / 4.0);
				T Omega_14 = f_t26 / 4.0 + f_t12 * (ux * uz * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) + f_t15 * ((ux * uz) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) + f_t22 * (ux / 2.0 - uy / 2.0 + ux * uy - 1.0 / 4.0) + (f_t25 * uz) / 2.0 - f_t21 * (uz / 2.0 - ux * uz) + f_t20 * (uz / 2.0 + uy * uz) - f_t7 * (ux / 6.0 + uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 6.0 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 6.0 + (uy * uy) / 1.2E+1) + f_t8 * (ux / 1.2E+1 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 6.0 + (ux * (uy * uy)) / 6.0 - ((ux * ux) * uy) / 6.0 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + f_t9 * (ux / 1.2E+1 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) - f_t16 * (uz / 2.0 - ux * uz + uy * uz - ux * uy * uz * 2.0) - f_t19 * (uy / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux * (-1.0 / 8.0) + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t2 * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t3 * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (ux / 2.0 - uy / 2.0 + ux * uy - (ux * (uz * uz)) / 2.0 + (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 - ux * uy * (uz * uz) - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uy / 8.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t11 * (ux / 8.0 + uy / 4.0 + (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t13 * (ux / 4.0 - uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) + f_t14 * (ux / 8.0 - uy / 4.0 + (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t6 * ((ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) * uy * uz + ux * uy * uz) - f_t5 * ((uy * uz) / 2.0 + ((uy * uy) * uz) / 2.0 - ux * (uy * uy) * uz - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t24 * (uy / 2.0 + 1.0 / 4.0);
				T Omega_15 = f_t26 / 4.0 + f_t11 * ((ux * uy) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) - f_t14 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) - f_t21 * (ux / 2.0 - uz / 2.0 - ux * uz + 1.0 / 4.0) + (f_t24 * uy) / 2.0 + f_t22 * (uy / 2.0 + ux * uy) - f_t20 * (uy / 2.0 - uy * uz) - f_t7 * (ux / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) + f_t8 * (ux / 6.0 + uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 6.0 - (uz * uz) / 1.2E+1) - f_t9 * (ux / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) - f_t16 * (uy / 2.0 + ux * uy - uy * uz - ux * uy * uz * 2.0) - f_t19 * (uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) + f_t17 * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) - f_t18 * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1) + f_t3 * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t1 * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + f_t2 * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (ux / 2.0 - uz / 2.0 - ux * uz - (ux * (uy * uy)) / 2.0 + ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 + ux * (uy * uy) * uz + 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t12 * (ux / 8.0 + uz / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t13 * (ux / 4.0 - uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t15 * (ux / 8.0 - uz / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) - f_t6 * ((ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * ux) * uy * uz - ux * uy * uz) - f_t4 * ((uy * uz) / 2.0 - (uy * (uz * uz)) / 2.0 - ux * uy * (uz * uz) + ux * uy * uz) + f_t23 * (ux / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
				T Omega_16 = f_t26 / 4.0 + f_t11 * (ux * uy * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) + f_t14 * ((ux * uy) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) + f_t21 * (ux / 2.0 - uz / 2.0 + ux * uz - 1.0 / 4.0) + (f_t24 * uy) / 2.0 - f_t22 * (uy / 2.0 - ux * uy) + f_t20 * (uy / 2.0 + uy * uz) + f_t7 * (ux / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 6.0 - (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t8 * (ux / 6.0 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (ux / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t16 * (uy / 2.0 - ux * uy + uy * uz - ux * uy * uz * 2.0) + f_t19 * (uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) + f_t17 * (ux * (-1.0 / 8.0) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t18 * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1) + f_t3 * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t1 * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - f_t2 * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (ux / 2.0 - uz / 2.0 + ux * uz - (ux * (uy * uy)) / 2.0 + ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 - ux * (uy * uy) * uz - 1.0 / 4.0) - f_t0 * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t10 * (ux / 4.0 + uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t12 * (ux / 8.0 + uz / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t13 * (ux / 4.0 - uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) + f_t15 * (ux / 8.0 - uz / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) - f_t6 * ((ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * ux) * uy * uz + ux * uy * uz) - f_t4 * ((uy * uz) / 2.0 + (uy * (uz * uz)) / 2.0 - ux * uy * (uz * uz) - ux * uy * uz) + f_t23 * (ux / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
				T Omega_17 = f_t26 / 4.0 + f_t10 * ((ux * uy) / 4.0 - (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) - f_t13 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) - f_t20 * (uy / 2.0 - uz / 2.0 - uy * uz + 1.0 / 4.0) - f_t19 * (uy / 8.0 + uz / 8.0 + (uy * uy) / 8.0 - (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 + f_t22 * (ux / 2.0 + ux * uy) - f_t21 * (ux / 2.0 - ux * uz) - f_t7 * (uy / 1.2E+1 + uz / 6.0 - ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 6.0 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 6.0 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 1.2E+1 - (uz * uz) / 6.0) + f_t8 * (uy / 6.0 + uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 6.0 - (uz * uz) / 1.2E+1) - f_t9 * (uy / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 1.2E+1 + (uz * uz) / 1.2E+1) - f_t16 * (ux / 2.0 + ux * uy - ux * uz - ux * uy * uz * 2.0) + f_t17 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0) + f_t18 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0) + f_t3 * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t2 * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + f_t1 * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) + f_t6 * (uy / 2.0 - uz / 2.0 - uy * uz - ((ux * ux) * uy) / 2.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 + (ux * ux) * uy * uz + 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t11 * (uy / 4.0 + uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) - f_t12 * (uy / 8.0 + uz / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) + f_t14 * (uy / 4.0 - uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0) - f_t15 * (uy / 8.0 - uz / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0 + 1.0 / 8.0) - f_t5 * ((ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - ux * (uy * uy) * uz - ux * uy * uz) - f_t4 * ((ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ux * uy * (uz * uz) + ux * uy * uz) + f_t24 * (uy / 2.0 + 1.0 / 4.0) + f_t25 * (uz / 2.0 - 1.0 / 4.0);
				T Omega_18 = f_t26 / 4.0 + f_t10 * (ux * uy * (-1.0 / 4.0) + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) + f_t13 * ((ux * uy) / 4.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) + f_t20 * (uy / 2.0 - uz / 2.0 + uy * uz - 1.0 / 4.0) + f_t19 * (uy / 8.0 + uz / 8.0 - (uy * uy) / 8.0 + (uz * uz) / 8.0) + (f_t23 * ux) / 2.0 - f_t22 * (ux / 2.0 - ux * uy) + f_t21 * (ux / 2.0 + ux * uz) + f_t7 * (uy / 1.2E+1 + uz / 6.0 + ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 6.0 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 + (uz * uz) / 6.0) - f_t8 * (uy / 6.0 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 6.0 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 6.0 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 6.0 + (uz * uz) / 1.2E+1) + f_t9 * (uy / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) - f_t16 * (ux / 2.0 - ux * uy + ux * uz - ux * uy * uz * 2.0) + f_t17 * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0) + f_t18 * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0) + f_t3 * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + f_t2 * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) - f_t1 * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t6 * (uy / 2.0 - uz / 2.0 + uy * uz - ((ux * ux) * uy) / 2.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 - (ux * ux) * uy * uz - 1.0 / 4.0) - f_t0 * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0) - f_t11 * (uy / 4.0 + uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) - f_t12 * (uy / 8.0 + uz / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) + f_t14 * (uy / 4.0 - uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 - (uz * uz) / 8.0 - 1.0 / 8.0) - f_t15 * (uy / 8.0 - uz / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0 - 1.0 / 8.0) - f_t5 * ((ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - ux * (uy * uy) * uz + ux * uy * uz) - f_t4 * ((ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ux * uy * (uz * uz) - ux * uy * uz) + f_t24 * (uy / 2.0 - 1.0 / 4.0) + f_t25 * (uz / 2.0 + 1.0 / 4.0);
				T Omega_19 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 + uy / 4.0 + (ux * uy) / 2.0 + 1.0 / 8.0) - f_t21 * (ux / 4.0 + uz / 4.0 + (ux * uz) / 2.0 + 1.0 / 8.0) - f_t20 * (uy / 4.0 + uz / 4.0 + (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t8 * ((ux * ux) * (uy * uy) * (-1.0 / 1.2E+1) + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 + uy / 4.0 + uz / 4.0 + (ux * uy) / 2.0 + (ux * uz) / 2.0 + (uy * uz) / 2.0 + ux * uy * uz + 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 + (ux * uy) / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 + (ux * uy) / 4.0 + (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 + (ux * uz) / 4.0 + (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
				T Omega_20 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 + uy / 4.0 - (ux * uy) / 2.0 - 1.0 / 8.0) + f_t21 * (ux / 4.0 + uz / 4.0 - (ux * uz) / 2.0 - 1.0 / 8.0) + f_t20 * (uy / 4.0 + uz / 4.0 - (uy * uz) / 2.0 - 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t17 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 + uy / 4.0 + uz / 4.0 - (ux * uy) / 2.0 - (ux * uz) / 2.0 - (uy * uz) / 2.0 + ux * uy * uz - 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 - (ux * uy) / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 - (ux * uy) / 4.0 - (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 - (ux * uz) / 4.0 - (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);
				T Omega_21 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 + uy / 4.0 + (ux * uy) / 2.0 + 1.0 / 8.0) + f_t21 * (ux / 4.0 - uz / 4.0 - (ux * uz) / 2.0 + 1.0 / 8.0) + f_t20 * (uy / 4.0 - uz / 4.0 - (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t18 * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 + uy / 4.0 - uz / 4.0 + (ux * uy) / 2.0 - (ux * uz) / 2.0 - (uy * uz) / 2.0 - ux * uy * uz + 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 + (ux * uy) / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 + (ux * uy) / 4.0 - (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 + (ux * uz) / 4.0 + (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);
				T Omega_22 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 + uy / 4.0 - (ux * uy) / 2.0 - 1.0 / 8.0) - f_t21 * (ux / 4.0 - uz / 4.0 + (ux * uz) / 2.0 - 1.0 / 8.0) - f_t20 * (uy / 4.0 - uz / 4.0 + (uy * uz) / 2.0 - 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t11 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t12 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) + f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t17 * (ux * (-1.0 / 1.6E+1) - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 + uy / 4.0 - uz / 4.0 - (ux * uy) / 2.0 + (ux * uz) / 2.0 + (uy * uz) / 2.0 - ux * uy * uz - 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 - (ux * uy) / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 - (ux * uy) / 4.0 + (uy * uz) / 4.0 + (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 - (ux * uz) / 4.0 - (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
				T Omega_23 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 - uy / 4.0 - (ux * uy) / 2.0 + 1.0 / 8.0) - f_t21 * (ux / 4.0 + uz / 4.0 + (ux * uz) / 2.0 + 1.0 / 8.0) - f_t20 * (uy / 4.0 - uz / 4.0 + (uy * uz) / 2.0 - 1.0 / 8.0) + f_t7 * ((ux * ux) * (uy * uy) * (-1.0 / 2.4E+1) + ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - f_t8 * ((ux * ux) * (uy * uy) * (-1.0 / 1.2E+1) + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t10 * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 - uy / 4.0 + uz / 4.0 - (ux * uy) / 2.0 + (ux * uz) / 2.0 - (uy * uz) / 2.0 - ux * uy * uz + 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 - (ux * uy) / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 + (ux * uy) / 4.0 + (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 + (ux * uz) / 4.0 - (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
				T Omega_24 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 - uy / 4.0 + (ux * uy) / 2.0 - 1.0 / 8.0) + f_t21 * (ux / 4.0 + uz / 4.0 - (ux * uz) / 2.0 - 1.0 / 8.0) + f_t20 * (uy / 4.0 - uz / 4.0 - (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 1.2E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 1.2E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t10 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t11 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t12 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 + uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t16 * (ux / 4.0 - uy / 4.0 + uz / 4.0 + (ux * uy) / 2.0 - (ux * uz) / 2.0 + (uy * uz) / 2.0 - ux * uy * uz - 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 + (ux * uy) / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 - (ux * uy) / 4.0 - (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 - (ux * uz) / 4.0 + (uy * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);
				T Omega_25 = f_t26 * (-1.0 / 8.0) - f_t22 * (ux / 4.0 - uy / 4.0 + (ux * uy) / 2.0 - 1.0 / 8.0) - f_t21 * (ux / 4.0 - uz / 4.0 + (ux * uz) / 2.0 - 1.0 / 8.0) - f_t20 * (uy / 4.0 + uz / 4.0 + (uy * uz) / 2.0 + 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t8 * ((ux * ux) * (uy * uy) * (-1.0 / 1.2E+1) + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 1.2E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t11 * (ux * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t12 * (ux * (-1.0 / 1.6E+1) + uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) + f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 + (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t17 * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 - uy / 4.0 - uz / 4.0 + (ux * uy) / 2.0 + (ux * uz) / 2.0 - (uy * uz) / 2.0 + ux * uy * uz - 1.0 / 8.0) - f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) + f_t6 * (ux / 8.0 + (ux * uy) / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * ux) / 8.0 - ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) + f_t5 * (uy / 8.0 - (ux * uy) / 4.0 + (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 + (uy * uy) / 8.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) + f_t4 * (uz / 8.0 - (ux * uz) / 4.0 + (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 + (uz * uz) / 8.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 - 1.0 / 8.0) - f_t24 * (uy / 4.0 + 1.0 / 8.0) - f_t25 * (uz / 4.0 + 1.0 / 8.0);
				T Omega_26 = f_t26 * (-1.0 / 8.0) + f_t22 * (ux / 4.0 - uy / 4.0 - (ux * uy) / 2.0 + 1.0 / 8.0) + f_t21 * (ux / 4.0 - uz / 4.0 - (ux * uz) / 2.0 + 1.0 / 8.0) + f_t20 * (uy / 4.0 + uz / 4.0 - (uy * uz) / 2.0 - 1.0 / 8.0) - f_t7 * (((ux * ux) * (uy * uy)) / 2.4E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 1.2E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t8 * (((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 2.4E+1 - ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 1.2E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) - f_t9 * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + f_t10 * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) + f_t11 * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) + f_t12 * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) - f_t13 * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t14 * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) - f_t15 * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) - f_t19 * (uy / 1.6E+1 - uz / 1.6E+1 - (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) - f_t17 * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1) + f_t18 * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1) - f_t16 * (ux / 4.0 - uy / 4.0 - uz / 4.0 - (ux * uy) / 2.0 - (ux * uz) / 2.0 + (uy * uz) / 2.0 + ux * uy * uz + 1.0 / 8.0) + f_t3 * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + f_t2 * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - f_t1 * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + f_t0 * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0) - f_t6 * (ux / 8.0 - (ux * uy) / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * ux) / 8.0 + ((ux * ux) * uy * uz) / 2.0 + (ux * uy * uz) / 2.0) - f_t5 * (uy / 8.0 + (ux * uy) / 4.0 - (uy * uz) / 4.0 - (ux * (uy * uy)) / 4.0 + ((uy * uy) * uz) / 4.0 - (uy * uy) / 8.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - f_t4 * (uz / 8.0 + (ux * uz) / 4.0 - (uy * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * (uz * uz)) / 4.0 - (uz * uz) / 8.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - f_t23 * (ux / 4.0 + 1.0 / 8.0) - f_t24 * (uy / 4.0 - 1.0 / 8.0) - f_t25 * (uz / 4.0 - 1.0 / 8.0);

				T G_0 = Fx * ux * (r5 / 2.0 - 1.0) * (2.0 / 9.0) + Fy * uy * (r5 / 2.0 - 1.0) * (2.0 / 9.0) + Fz * uz * (r5 / 2.0 - 1.0) * (2.0 / 9.0) + Fx * (r3 / 2.0 - 1.0) * (ux * -2.0 + ux * (uy * uy) + ux * (uz * uz)) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (uy * -2.0 + (ux * ux) * uy + uy * (uz * uz)) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (uz * -2.0 + (ux * ux) * uz + (uy * uy) * uz) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * 2.0 - ux * (uy * uy) * 2.0 - ux * (uz * uz) * 2.0 + ux * (uy * uy) * (uz * uz) * 2.0) + Fy * (r1 / 2.0 - 1.0) * (uy * 2.0 - (ux * ux) * uy * 2.0 - uy * (uz * uz) * 2.0 + (ux * ux) * uy * (uz * uz) * 2.0) + Fz * (r1 / 2.0 - 1.0) * (uz * 2.0 - (ux * ux) * uz * 2.0 - (uy * uy) * uz * 2.0 + (ux * ux) * (uy * uy) * uz * 2.0);
				T G_1 = Fy * uy * (r5 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r5 / 2.0 - 1.0)) / 9.0 - Fy * (r3 / 2.0 - 1.0) * (uy * (-1.0 / 2.0) + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (uz * (-1.0 / 2.0) + (ux * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fx * (r5 / 2.0 - 1.0) * (ux + 1.0 / 2.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (-ux + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fy * (r1 / 2.0 - 1.0) * (ux * uy + (ux * ux) * uy - ux * uy * (uz * uz) - (ux * ux) * uy * (uz * uz)) + Fz * (r1 / 2.0 - 1.0) * (ux * uz + (ux * ux) * uz - ux * (uy * uy) * uz - (ux * ux) * (uy * uy) * uz) - Fx * (r1 / 2.0 - 1.0) * (ux + ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) - (uy * uy) / 2.0 - (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) + 1.0 / 2.0);
				T G_2 = Fy * uy * (r5 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r5 / 2.0 - 1.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * (ux - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (uy / 2.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (uz / 2.0 + (ux * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fx * (r5 / 2.0 - 1.0) * (ux - 1.0 / 2.0)) / 9.0 - Fy * (r1 / 2.0 - 1.0) * (ux * uy - (ux * ux) * uy - ux * uy * (uz * uz) + (ux * ux) * uy * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (ux * uz - (ux * ux) * uz - ux * (uy * uy) * uz + (ux * ux) * (uy * uy) * uz) - Fx * (r1 / 2.0 - 1.0) * (ux - ((uy * uy) * (uz * uz)) / 2.0 - ux * (uy * uy) - ux * (uz * uz) + (uy * uy) / 2.0 + (uz * uz) / 2.0 + ux * (uy * uy) * (uz * uz) - 1.0 / 2.0);
				T G_3 = Fx * ux * (r5 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r5 / 2.0 - 1.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (ux * (-1.0 / 2.0) + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (uz * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fy * (r5 / 2.0 - 1.0) * (uy + 1.0 / 2.0)) / 9.0 - Fy * (r3 / 2.0 - 1.0) * (-uy + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * uy + ux * (uy * uy) - ux * uy * (uz * uz) - ux * (uy * uy) * (uz * uz)) + Fz * (r1 / 2.0 - 1.0) * (uy * uz + (uy * uy) * uz - (ux * ux) * uy * uz - (ux * ux) * (uy * uy) * uz) - Fy * (r1 / 2.0 - 1.0) * (uy + ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) - (ux * ux) / 2.0 - (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) + 1.0 / 2.0);
				T G_4 = Fx * ux * (r5 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fz * uz * (r5 / 2.0 - 1.0)) / 9.0 + Fy * (r3 / 2.0 - 1.0) * (uy - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (ux * ux) / 4.0 + (uz * uz) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r3 / 2.0 - 1.0) * (ux / 2.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (uz / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0) * (2.0 / 3.0) - (Fy * (r5 / 2.0 - 1.0) * (uy - 1.0 / 2.0)) / 9.0 - Fx * (r1 / 2.0 - 1.0) * (ux * uy - ux * (uy * uy) - ux * uy * (uz * uz) + ux * (uy * uy) * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (uy * uz - (uy * uy) * uz - (ux * ux) * uy * uz + (ux * ux) * (uy * uy) * uz) - Fy * (r1 / 2.0 - 1.0) * (uy - ((ux * ux) * (uz * uz)) / 2.0 - (ux * ux) * uy - uy * (uz * uz) + (ux * ux) / 2.0 + (uz * uz) / 2.0 + (ux * ux) * uy * (uz * uz) - 1.0 / 2.0);
				T G_5 = Fx * ux * (r5 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fy * uy * (r5 / 2.0 - 1.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (ux * (-1.0 / 2.0) + (ux * uz) / 2.0 + (ux * (uy * uy)) / 2.0 + (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (uy * (-1.0 / 2.0) + (uy * uz) / 2.0 + ((ux * ux) * uy) / 2.0 + (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) - (Fz * (r5 / 2.0 - 1.0) * (uz + 1.0 / 2.0)) / 9.0 - Fz * (r3 / 2.0 - 1.0) * (-uz + ((ux * ux) * uz) / 2.0 + ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r1 / 2.0 - 1.0) * (ux * uz + ux * (uz * uz) - ux * (uy * uy) * uz - ux * (uy * uy) * (uz * uz)) + Fy * (r1 / 2.0 - 1.0) * (uy * uz + uy * (uz * uz) - (ux * ux) * uy * uz - (ux * ux) * uy * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (uz + ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz - (ux * ux) / 2.0 - (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz + 1.0 / 2.0);
				T G_6 = Fx * ux * (r5 / 2.0 - 1.0) * (-1.0 / 9.0) - (Fy * uy * (r5 / 2.0 - 1.0)) / 9.0 + Fz * (r3 / 2.0 - 1.0) * (uz - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 2.0) * (2.0 / 3.0) + Fx * (r3 / 2.0 - 1.0) * (ux / 2.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (uy / 2.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0) * (2.0 / 3.0) - (Fz * (r5 / 2.0 - 1.0) * (uz - 1.0 / 2.0)) / 9.0 - Fx * (r1 / 2.0 - 1.0) * (ux * uz - ux * (uz * uz) - ux * (uy * uy) * uz + ux * (uy * uy) * (uz * uz)) - Fy * (r1 / 2.0 - 1.0) * (uy * uz - uy * (uz * uz) - (ux * ux) * uy * uz + (ux * ux) * uy * (uz * uz)) - Fz * (r1 / 2.0 - 1.0) * (uz - ((ux * ux) * (uy * uy)) / 2.0 - (ux * ux) * uz - (uy * uy) * uz + (ux * ux) / 2.0 + (uy * uy) / 2.0 + (ux * ux) * (uy * uy) * uz - 1.0 / 2.0);
				T G_7 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + Fz * (r3 / 2.0 - 1.0) * ((ux * uz) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r5 / 2.0 - 1.0)) / 1.8E+1 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) + Fx * (r3 / 2.0 - 1.0) * (ux * (-1.0 / 4.0) + uy / 8.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (ux / 8.0 - uy / 4.0 + (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
				T G_8 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 - Fz * (r3 / 2.0 - 1.0) * ((ux * uz) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r5 / 2.0 - 1.0)) / 1.8E+1 - Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 + (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r3 / 2.0 - 1.0) * (ux / 4.0 - uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (ux / 8.0 - uy / 4.0 - (ux * uy) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
				T G_9 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fy * (r3 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r5 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) + Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) + Fx * (r3 / 2.0 - 1.0) * (ux * (-1.0 / 4.0) + uz / 8.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (ux / 8.0 - uz / 4.0 + (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
				T G_10 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 - Fy * (r3 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r5 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 + (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0) - Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r3 / 2.0 - 1.0) * (ux / 4.0 - uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (ux / 8.0 - uz / 4.0 - (ux * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
				T G_11 = (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r5 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) + Fy * (r3 / 2.0 - 1.0) * (uy * (-1.0 / 4.0) + uz / 8.0 + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (uy / 8.0 - uz / 4.0 + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
				T G_12 = (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * ((ux * uy) / 4.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r5 / 2.0 - 1.0)) / 1.8E+1 - Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 + ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0) - Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 + (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fy * (r3 / 2.0 - 1.0) * (uy / 4.0 - uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (uy / 8.0 - uz / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
				T G_13 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 + Fz * (r3 / 2.0 - 1.0) * ((ux * uz) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r5 / 2.0 - 1.0)) / 1.8E+1 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 + (ux * ux) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 - (uy * uy) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r3 / 2.0 - 1.0) * (ux / 4.0 + uy / 8.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux / 8.0 + uy / 4.0 - (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
				T G_14 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + Fz * (r3 / 2.0 - 1.0) * (ux * uz * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uz) / 4.0 + ((uy * uy) * uz) / 4.0) * (2.0 / 3.0) + (Fz * uz * (r5 / 2.0 - 1.0)) / 1.8E+1 + Fy * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (ux * uy) / 2.0 - ((ux * ux) * uy) / 2.0 - (ux * (uz * uz)) / 4.0 - (ux * ux) / 4.0 - (ux * uy * (uz * uz)) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uy) / 2.0 - (ux * (uy * uy)) / 2.0 - (uy * (uz * uz)) / 4.0 + (uy * uy) / 4.0 + (ux * uy * (uz * uz)) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - Fz * (r1 / 2.0 - 1.0) * ((ux * (uy * uy) * uz) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * (uy * uy) * uz) / 2.0 + (ux * uy * uz) / 2.0) - Fx * (r3 / 2.0 - 1.0) * (ux / 4.0 + uy / 8.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux / 8.0 + uy / 4.0 + (ux * uy) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
				T G_15 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 + Fy * (r3 / 2.0 - 1.0) * ((ux * uy) / 4.0 - (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r5 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 + (ux * ux) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((uy * uy) * (uz * uz)) / 4.0 + (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 - (uz * uz) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) + Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fx * (r3 / 2.0 - 1.0) * (ux / 4.0 + uz / 8.0 + (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux / 8.0 + uz / 4.0 - (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
				T G_16 = (Fx * (r5 / 2.0 - 1.0) * (ux / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fy * (r3 / 2.0 - 1.0) * (ux * uy * (-1.0 / 4.0) + (uy * uz) / 4.0 + ((ux * ux) * uy) / 4.0 + (uy * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fy * uy * (r5 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (ux / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (ux * uz) / 2.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uz) / 2.0 - (ux * ux) / 4.0 - (ux * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fx * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((uy * uy) * (uz * uz)) / 4.0 - (ux * uz) / 2.0 - (ux * (uz * uz)) / 2.0 - ((uy * uy) * uz) / 4.0 + (uz * uz) / 4.0 + (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0) - Fy * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - ((ux * ux) * uy * uz) / 2.0 - ((ux * ux) * uy * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - Fx * (r3 / 2.0 - 1.0) * (ux / 4.0 + uz / 8.0 - (ux * uz) / 4.0 - (ux * (uy * uy)) / 4.0 - (ux * (uz * uz)) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux / 8.0 + uz / 4.0 + (ux * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
				T G_17 = (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 + 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 - 1.0 / 4.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * ((ux * uy) / 4.0 - (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r5 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 - ((ux * ux) * (uy * uy)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 + (uy * uy) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 + ((ux * ux) * (uz * uz)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 - (uz * uz) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) + Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - (ux * uy * uz) / 2.0) - Fy * (r3 / 2.0 - 1.0) * (uy / 4.0 + uz / 8.0 + (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 - (ux * ux) / 8.0 - (uz * uz) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (uy / 8.0 + uz / 4.0 - (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0);
				T G_18 = (Fy * (r5 / 2.0 - 1.0) * (uy / 2.0 - 1.0 / 4.0)) / 9.0 + (Fz * (r5 / 2.0 - 1.0) * (uz / 2.0 + 1.0 / 4.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * (ux * uy * (-1.0 / 4.0) + (ux * uz) / 4.0 + (ux * (uy * uy)) / 4.0 + (ux * (uz * uz)) / 4.0) * (2.0 / 3.0) + (Fx * ux * (r5 / 2.0 - 1.0)) / 1.8E+1 + Fz * (r1 / 2.0 - 1.0) * (uy / 4.0 + ((ux * ux) * (uy * uy)) / 4.0 + (uy * uz) / 2.0 - ((ux * ux) * uy) / 4.0 - ((uy * uy) * uz) / 2.0 - (uy * uy) / 4.0 - ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0) + Fy * (r1 / 2.0 - 1.0) * (uz / 4.0 - ((ux * ux) * (uz * uz)) / 4.0 - (uy * uz) / 2.0 - ((ux * ux) * uz) / 4.0 - (uy * (uz * uz)) / 2.0 + (uz * uz) / 4.0 + ((ux * ux) * uy * uz) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0) - Fx * (r1 / 2.0 - 1.0) * ((ux * uy * (uz * uz)) / 2.0 - (ux * (uy * uy) * uz) / 2.0 - (ux * (uy * uy) * (uz * uz)) / 2.0 + (ux * uy * uz) / 2.0) - Fy * (r3 / 2.0 - 1.0) * (uy / 4.0 + uz / 8.0 - (uy * uz) / 4.0 - ((ux * ux) * uy) / 4.0 - (uy * (uz * uz)) / 4.0 + (ux * ux) / 8.0 + (uz * uz) / 8.0 - 1.0 / 8.0) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (uy / 8.0 + uz / 4.0 + (uy * uz) / 4.0 - ((ux * ux) * uz) / 4.0 - ((uy * uy) * uz) / 4.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0 + 1.0 / 8.0) * (2.0 / 3.0);
				T G_19 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);
				T G_20 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
				T G_21 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 - uz / 1.6E+1 + (ux * uy) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 - (ux * uz) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
				T G_22 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uy / 1.6E+1 + (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 + (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);
				T G_23 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 - Fx * (r3 / 2.0 - 1.0) * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 + (ux * (uy * uy)) / 8.0 + (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 - uy / 1.6E+1 + (ux * uz) / 8.0 - (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
				T G_24 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 - (ux * ux) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * ux) / 1.6E+1 - (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 + (ux * (uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 + (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 - (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);
				T G_25 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 - 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 + 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 + 1.0 / 8.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * uy) / 8.0 - (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fy * (r3 / 2.0 - 1.0) * (ux * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uy) / 8.0 + (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) - Fz * (r3 / 2.0 - 1.0) * (ux * (-1.0 / 1.6E+1) + uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 + ((ux * ux) * uz) / 8.0 + ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) - Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 - (ux * (uy * uy)) / 8.0 + ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - (ux * uy * uz) / 4.0) - Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 - (ux * (uz * uz)) / 8.0 + ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0) + Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 + (uy * (uz * uz)) / 8.0 + ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - (ux * uy * uz) / 4.0);
				T G_26 = Fx * (r5 / 2.0 - 1.0) * (ux / 4.0 + 1.0 / 8.0) * (-1.0 / 9.0) - (Fy * (r5 / 2.0 - 1.0) * (uy / 4.0 - 1.0 / 8.0)) / 9.0 - (Fz * (r5 / 2.0 - 1.0) * (uz / 4.0 - 1.0 / 8.0)) / 9.0 + Fx * (r3 / 2.0 - 1.0) * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * uy) / 8.0 + (ux * uz) / 8.0 - (ux * (uy * uy)) / 8.0 - (ux * (uz * uz)) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fy * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 - uz / 1.6E+1 - (ux * uy) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uy) / 8.0 - (uy * (uz * uz)) / 8.0 + (ux * ux) / 1.6E+1 + (uz * uz) / 1.6E+1) * (2.0 / 3.0) + Fz * (r3 / 2.0 - 1.0) * (ux / 1.6E+1 - uy / 1.6E+1 - (ux * uz) / 8.0 + (uy * uz) / 8.0 - ((ux * ux) * uz) / 8.0 - ((uy * uy) * uz) / 8.0 + (ux * ux) / 1.6E+1 + (uy * uy) / 1.6E+1) * (2.0 / 3.0) + Fz * (r1 / 2.0 - 1.0) * (((ux * ux) * (uy * uy)) / 8.0 - (ux * uy) / 8.0 + (ux * (uy * uy)) / 8.0 - ((ux * ux) * uy) / 8.0 - (ux * (uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 + (ux * uy * uz) / 4.0) + Fy * (r1 / 2.0 - 1.0) * (((ux * ux) * (uz * uz)) / 8.0 - (ux * uz) / 8.0 + (ux * (uz * uz)) / 8.0 - ((ux * ux) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0) - Fx * (r1 / 2.0 - 1.0) * (((uy * uy) * (uz * uz)) / 8.0 + (uy * uz) / 8.0 - (uy * (uz * uz)) / 8.0 - ((uy * uy) * uz) / 8.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + (ux * uy * uz) / 4.0);

				st.f_star[pos][0] = f0 + Omega_0 + G_0;
				st.f_star[pos][1] = f1 + Omega_1 + G_1;
				st.f_star[pos][2] = f2 + Omega_2 + G_2;
				st.f_star[pos][3] = f3 + Omega_3 + G_3;
				st.f_star[pos][4] = f4 + Omega_4 + G_4;
				st.f_star[pos][5] = f5 + Omega_5 + G_5;
				st.f_star[pos][6] = f6 + Omega_6 + G_6;
				st.f_star[pos][7] = f7 + Omega_7 + G_7;
				st.f_star[pos][8] = f8 + Omega_8 + G_8;
				st.f_star[pos][9] = f9 + Omega_9 + G_9;
				st.f_star[pos][10] = f10 + Omega_10 + G_10;
				st.f_star[pos][11] = f11 + Omega_11 + G_11;
				st.f_star[pos][12] = f12 + Omega_12 + G_12;
				st.f_star[pos][13] = f13 + Omega_13 + G_13;
				st.f_star[pos][14] = f14 + Omega_14 + G_14;
				st.f_star[pos][15] = f15 + Omega_15 + G_15;
				st.f_star[pos][16] = f16 + Omega_16 + G_16;
				st.f_star[pos][17] = f17 + Omega_17 + G_17;
				st.f_star[pos][18] = f18 + Omega_18 + G_18;
				st.f_star[pos][19] = f19 + Omega_19 + G_19;
				st.f_star[pos][20] = f20 + Omega_20 + G_20;
				st.f_star[pos][21] = f21 + Omega_21 + G_21;
				st.f_star[pos][22] = f22 + Omega_22 + G_22;
				st.f_star[pos][23] = f23 + Omega_23 + G_23;
				st.f_star[pos][24] = f24 + Omega_24 + G_24;
				st.f_star[pos][25] = f25 + Omega_25 + G_25;
				st.f_star[pos][26] = f26 + Omega_26 + G_26;
			}
}

template<typename T>
T LBM_Solver_Specialisation<T, 2>::calc_f_eq_CM(int i, T rho, vec<T, 2> u)
{
	T ux = u[0], uy = u[1];
	switch (i)
	{
	case 0:
		return rho / 9.0 + rho * ((ux * ux) / 2.0 + (uy * uy) / 2.0 - 1.0) * (2.0 / 3.0) + rho * ((ux * ux) * (uy * uy) - ux * ux - uy * uy + 1.0);
	case 1:
		return rho * (-1.0 / 1.8E+1) - rho * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) * (2.0 / 3.0) + rho * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 + (ux * ux) / 2.0);
	case 2:
		return rho * (-1.0 / 1.8E+1) - rho * (uy / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) * (2.0 / 3.0) + rho * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 + (uy * uy) / 2.0);
	case 3:
		return rho * (-1.0 / 1.8E+1) + rho * (ux / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) * (2.0 / 3.0) - rho * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * ux) / 2.0);
	case 4:
		return rho * (-1.0 / 1.8E+1) + rho * (uy / 4.0 - (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) * (2.0 / 3.0) - rho * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * uy) / 2.0);
	case 5:
		return rho / 3.6E+1 + rho * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0);
	case 6:
		return rho / 3.6E+1 + rho * (ux * (-1.0 / 8.0) + uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0);
	case 7:
		return rho / 3.6E+1 - rho * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 - (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0);
	case 8:
		return rho / 3.6E+1 + rho * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) * (2.0 / 3.0) + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0);
	default:
		return 0;
	}
}

template<typename T>
T LBM_Solver_Specialisation<T, 3>::calc_f_eq_CM(int i, T rho, vec<T, 3> u)
{
	T ux = u[0], uy = u[1], uz = u[2];
	switch (i)
	{
		case 0:
			return rho * (-1.0 / 2.7E+1) + rho * ((ux * ux) * (uy * uy) + (ux * ux) * (uz * uz) + (uy * uy) * (uz * uz) - ux * ux - uy * uy - uz * uz - (ux * ux) * (uy * uy) * (uz * uz) + 1.0) - (rho * ((ux * ux) * (-1.0 / 2.0) + (uy * uy) / 4.0 + (uz * uz) / 4.0)) / 9.0 - (rho * ((ux * ux) / 2.0 + (uy * uy) / 4.0 + (uz * uz) / 4.0 - 1.0)) / 3.0 - rho * (((ux * ux) * (uy * uy)) / 3.0 + ((ux * ux) * (uz * uz)) / 3.0 + ((uy * uy) * (uz * uz)) / 3.0 - (ux * ux) * (2.0 / 3.0) - (uy * uy) * (2.0 / 3.0) - (uz * uz) * (2.0 / 3.0) + 1.0);
		case 1:
			return rho / 5.4E+1 + rho * (ux * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + (ux * (uy * uy)) / 6.0 + (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) - (rho * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 8.0 - (uz * uz) / 8.0 + 1.0 / 4.0)) / 9.0 + (rho * (ux / 4.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0)) / 3.0 + rho * (ux / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 + (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
		case 2:
			return rho / 5.4E+1 + rho * (ux / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - (ux * (uy * uy)) / 6.0 - (ux * (uz * uz)) / 6.0 - (ux * ux) / 3.0 - (uy * uy) / 6.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + (rho * (ux * (-1.0 / 4.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0)) / 3.0 + (rho * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 1.0 / 4.0)) / 9.0 - rho * (ux / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 - (ux * (uy * uy)) / 2.0 - (ux * (uz * uz)) / 2.0 - (ux * ux) / 2.0 + (ux * (uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
		case 3:
			return rho / 5.4E+1 + rho * (uy * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uy) / 6.0 + (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + (rho * (uy / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uy / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 + rho * (uy / 2.0 - ((ux * ux) * (uy * uy)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 + (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
		case 4:
			return rho / 5.4E+1 + rho * (uy / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uy) / 6.0 - (uy * (uz * uz)) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 3.0 - (uz * uz) / 6.0 + 1.0 / 6.0) + (rho * (uy * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uy * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 - rho * (uy / 2.0 + ((ux * ux) * (uy * uy)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uy) / 2.0 - (uy * (uz * uz)) / 2.0 - (uy * uy) / 2.0 + ((ux * ux) * uy * (uz * uz)) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
		case 5:
			return rho / 5.4E+1 + rho * (uz * (-1.0 / 3.0) + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 + ((ux * ux) * uz) / 6.0 + ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) + (rho * (uz / 8.0 - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uz / 8.0 + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 + rho * (uz / 2.0 - ((ux * ux) * (uz * uz)) / 2.0 - ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 + (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
		case 6:
			return rho / 5.4E+1 + rho * (uz / 3.0 + ((ux * ux) * (uy * uy)) / 6.0 + ((ux * ux) * (uz * uz)) / 6.0 + ((uy * uy) * (uz * uz)) / 6.0 - ((ux * ux) * uz) / 6.0 - ((uy * uy) * uz) / 6.0 - (ux * ux) / 6.0 - (uy * uy) / 6.0 - (uz * uz) / 3.0 + 1.0 / 6.0) + (rho * (uz * (-1.0 / 8.0) - (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 + 1.0 / 8.0)) / 9.0 + (rho * (uz * (-1.0 / 8.0) + (ux * ux) / 4.0 + (uy * uy) / 8.0 + (uz * uz) / 8.0 - 3.0 / 8.0)) / 3.0 - rho * (uz / 2.0 + ((ux * ux) * (uz * uz)) / 2.0 + ((uy * uy) * (uz * uz)) / 2.0 - ((ux * ux) * uz) / 2.0 - ((uy * uy) * uz) / 2.0 - (uz * uz) / 2.0 + ((ux * ux) * (uy * uy) * uz) / 2.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 2.0);
		case 7:
			return rho * (-1.0 / 1.08E+2) - rho * (ux * (-1.0 / 1.2E+1) - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) + (rho * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 - (rho * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 8:
			return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 + uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) - (rho * (ux / 8.0 - uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + (rho * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 + (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 - (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 9:
			return rho * (-1.0 / 1.08E+2) - rho * (ux * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 + (ux * (uy * uy)) / 1.2E+1 + (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) + (rho * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 - (rho * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 10:
			return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (ux / 8.0 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + (rho * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 3.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 + (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 - (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 11:
			return rho * (-1.0 / 1.08E+2) - rho * (uy * (-1.0 / 1.2E+1) - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 9.0 - (rho * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 3.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 12:
			return rho * (-1.0 / 1.08E+2) - rho * (uy / 1.2E+1 + uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 - ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 - ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) + (rho * (uy / 1.6E+1 + uz / 1.6E+1 - (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 3.0 + (rho * (uy / 1.6E+1 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 9.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 + (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 - ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 13:
			return rho * (-1.0 / 1.08E+2) + rho * (ux / 1.2E+1 - uy / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uy * uy) / 1.2E+1) - (rho * (ux / 8.0 - uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + (rho * (ux / 8.0 + uy / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 + (ux * (uy * uy)) / 4.0 - ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 14:
			return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 - uy / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uy) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 + ((ux * ux) * uy) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + (uy * (uz * uz)) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uy * uy) / 1.2E+1) - (rho * (ux * (-1.0 / 8.0) + uy / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 - (rho * (ux / 8.0 + uy / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uy * uy)) / 4.0 - (ux * uy) / 4.0 - (ux * (uy * uy)) / 4.0 + ((ux * ux) * uy) / 4.0 + (ux * uy * (uz * uz)) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 15:
			return rho * (-1.0 / 1.08E+2) + rho * (ux / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (ux * ux) / 1.2E+1 + (uz * uz) / 1.2E+1) - (rho * (ux / 8.0 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 + (rho * (ux / 8.0 + uz / 1.6E+1 + (ux * ux) / 8.0 - (uy * uy) / 1.6E+1 - (uz * uz) / 1.6E+1 + 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 + (ux * (uz * uz)) / 4.0 - ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 - (ux * (uy * uy) * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 16:
			return rho * (-1.0 / 1.08E+2) - rho * (ux / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (ux * uz) / 1.2E+1 - (ux * (uy * uy)) / 1.2E+1 - (ux * (uz * uz)) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (ux * ux) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (ux * (-1.0 / 8.0) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 3.0 - (rho * (ux / 8.0 + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 1.6E+1)) / 9.0 + rho * (((ux * ux) * (uz * uz)) / 4.0 - (ux * uz) / 4.0 - (ux * (uz * uz)) / 4.0 + ((ux * ux) * uz) / 4.0 + (ux * (uy * uy) * uz) / 4.0 + (ux * (uy * uy) * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 17:
			return rho * (-1.0 / 1.08E+2) + rho * (uy / 1.2E+1 - uz / 1.2E+1 - ((ux * ux) * (uy * uy)) / 1.2E+1 - ((ux * ux) * (uz * uz)) / 1.2E+1 - ((uy * uy) * (uz * uz)) / 1.2E+1 + (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 + (uy * uy) / 1.2E+1 + (uz * uz) / 1.2E+1) - (rho * (uy / 1.6E+1 - uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 9.0 - (rho * (uy / 1.6E+1 - uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 3.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 + (uy * (uz * uz)) / 4.0 - ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 - ((ux * ux) * uy * (uz * uz)) / 4.0 + ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 18:
			return rho * (-1.0 / 1.08E+2) - rho * (uy / 1.2E+1 - uz / 1.2E+1 + ((ux * ux) * (uy * uy)) / 1.2E+1 + ((ux * ux) * (uz * uz)) / 1.2E+1 + ((uy * uy) * (uz * uz)) / 1.2E+1 - (uy * uz) / 1.2E+1 - ((ux * ux) * uy) / 1.2E+1 + ((ux * ux) * uz) / 1.2E+1 - (uy * (uz * uz)) / 1.2E+1 + ((uy * uy) * uz) / 1.2E+1 - (uy * uy) / 1.2E+1 - (uz * uz) / 1.2E+1) - (rho * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 - (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 + 1.0 / 8.0)) / 9.0 - (rho * (uy * (-1.0 / 1.6E+1) + uz / 1.6E+1 + (ux * ux) / 8.0 + (uy * uy) / 1.6E+1 + (uz * uz) / 1.6E+1 - 1.0 / 8.0)) / 3.0 + rho * (((uy * uy) * (uz * uz)) / 4.0 - (uy * uz) / 4.0 - (uy * (uz * uz)) / 4.0 + ((uy * uy) * uz) / 4.0 + ((ux * ux) * uy * uz) / 4.0 + ((ux * ux) * uy * (uz * uz)) / 4.0 - ((ux * ux) * (uy * uy) * uz) / 4.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 4.0);
		case 19:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 20:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + (rho * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 - (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 3.0 + rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 21:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) - (rho * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 9.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 22:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 + (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + (rho * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 + rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 + ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 23:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux / 1.6E+1 - uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 - (rho * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 24:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 + (ux * uz) / 2.4E+1 - (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 - uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 - (ux * (uy * uy) * uz) / 8.0 + ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 25:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 - (ux * (uy * uy)) / 2.4E+1 + ((ux * ux) * uy) / 2.4E+1 - (ux * (uz * uz)) / 2.4E+1 + ((ux * ux) * uz) / 2.4E+1 + (uy * (uz * uz)) / 2.4E+1 + ((uy * uy) * uz) / 2.4E+1) + (rho * (ux * (-1.0 / 1.6E+1) + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 + (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 - (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 + (ux * (uy * uy) * (uz * uz)) / 8.0 - ((ux * ux) * uy * (uz * uz)) / 8.0 - ((ux * ux) * (uy * uy) * uz) / 8.0 + (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		case 26:
			return rho / 2.16E+2 + rho * (((ux * ux) * (uy * uy)) / 2.4E+1 + ((ux * ux) * (uz * uz)) / 2.4E+1 + ((uy * uy) * (uz * uz)) / 2.4E+1 - (ux * uy) / 2.4E+1 - (ux * uz) / 2.4E+1 + (uy * uz) / 2.4E+1 + (ux * (uy * uy)) / 2.4E+1 - ((ux * ux) * uy) / 2.4E+1 + (ux * (uz * uz)) / 2.4E+1 - ((ux * ux) * uz) / 2.4E+1 - (uy * (uz * uz)) / 2.4E+1 - ((uy * uy) * uz) / 2.4E+1) + (rho * (ux / 1.6E+1 - uy / 3.2E+1 - uz / 3.2E+1 + (ux * ux) / 1.6E+1 + (uy * uy) / 3.2E+1 + (uz * uz) / 3.2E+1)) / 3.0 - (rho * (ux / 1.6E+1 + uy / 3.2E+1 + uz / 3.2E+1 + (ux * ux) / 1.6E+1 - (uy * uy) / 3.2E+1 - (uz * uz) / 3.2E+1)) / 9.0 - rho * ((ux * uy * (uz * uz)) / 8.0 + (ux * (uy * uy) * uz) / 8.0 - ((ux * ux) * uy * uz) / 8.0 - (ux * (uy * uy) * (uz * uz)) / 8.0 + ((ux * ux) * uy * (uz * uz)) / 8.0 + ((ux * ux) * (uy * uy) * uz) / 8.0 - (ux * uy * uz) / 8.0 - ((ux * ux) * (uy * uy) * (uz * uz)) / 8.0);
		default:
			return 0;
	}
}

//--------------
template<typename T, size_t D>
void LBM_Solver<T, D>::collision_BGK()
{
	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
	{
		for (int i = 0; i < sd.getQ(); ++i)
		{


			//Extern Forces at i
			T F_i = ((T)1.0 - this->sd.getRelaxationConstant() / (T)2.0) * sd.w(i) *
				dot((sd.c(i) - st.u_L[pos]) / cs_sq<T> +dot(sd.c(i), st.u_L[pos]) / (cs_sq<T> *cs_sq<T>) * sd.c(i), st.F_ext_L[pos]);

			//Equlibrium at i
			T fi_eq = sd.w(i) * st.rho_L[pos] * ((T)1.0 + dot(sd.c(i), st.u_L[pos]) / cs_sq<T> +dot(sd.c(i), st.u_L[pos]) *
				dot(sd.c(i), st.u_L[pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(st.u_L[pos], st.u_L[pos]) / ((T)2.0 * cs_sq<T>));

			//Particle Distribution


			st.f_star[pos][i] = st.f[pos][i] - this->sd.getRelaxationConstant() * (st.f[pos][i] - fi_eq) + F_i;
		}
	}
}

template<typename T, size_t D>
void LBM_Solver<T, D>::collision_ghost_CM()
{
	for (int x_S_pos = 0; x_S_pos < sd.getMaxNodeCount(); ++x_S_pos)
	{
		for (int i = 0; i < sd.getQ(); ++i)
		{
			if (st.inletGhosCells[x_S_pos][i].isInlet)
			{
				int x_B_pos = st.inletGhosCells[x_S_pos][i].pos_bound;
				int x_F_pos = st.inletGhosCells[x_S_pos][i].pos_fluid;
				vec<T, D> velocity = st.inletGhosCells[x_S_pos][i].velocity;
				T q = st.inletGhosCells[x_S_pos][i].q;

				vec<T, D> u_solid;

				if (q >= (T)3.0 / 4.0)
				{
					u_solid = ((q - (T)1.0) * st.u_L[x_B_pos] + velocity) / q;
				}
				else
				{
					u_solid = (q - (T)1.0) * st.u_L[x_B_pos] + velocity + ((T)1.0 - q) * ((q - (T)1.0) * st.u_L[x_F_pos] + (T)2.0 * velocity) / ((T)1.0 + q);
				}

				T f_s_eq = calc_f_eq_CM(i, st.rho_L[x_B_pos], u_solid);

				T f_s_neq;

				if (q > (T)3.0 / 4.0)
				{
					T f_eq_B = calc_f_eq_CM(i, st.rho_L[x_B_pos], st.u_L[x_B_pos]);

					f_s_neq = st.f[x_B_pos][i] - f_eq_B;
				}
				else
				{
					T f_eq_B = calc_f_eq_CM(i, st.rho_L[x_B_pos], st.u_L[x_B_pos]);

					T f_eq_F = calc_f_eq_CM(i, st.rho_L[x_F_pos], st.u_L[x_F_pos]);

					f_s_neq = q * (st.f[x_B_pos][i] - f_eq_B) + ((T)1.0 - q) * (st.f[x_F_pos][i] - f_eq_F);

				}

				st.f_star[x_S_pos][i] = f_s_neq + f_s_eq;

			}
			st.inletGhosCells[x_S_pos][i].isInlet = false;
		}
	}
}

template<typename T, size_t D>
void LBM_Solver<T, D>::collision_ghost_BGK()
{
	for (int x_S_pos = 0; x_S_pos < sd.getMaxNodeCount(); ++x_S_pos)
	{
		for (int i = 0; i < sd.getQ(); ++i)
		{
			if (st.inletGhosCells[x_S_pos][i].isInlet)
			{
				int x_B_pos = st.inletGhosCells[x_S_pos][i].pos_bound;
				int x_F_pos = st.inletGhosCells[x_S_pos][i].pos_fluid;
				vec<T, D> velocity = st.inletGhosCells[x_S_pos][i].velocity;
				T q = st.inletGhosCells[x_S_pos][i].q;

				vec<T, D> u_solid;
				T rho = st.rho_L[x_B_pos];

				if (q >= (T)3.0 / 4.0)
				{
					u_solid = ((q - (T)1.0) * st.u_L[x_B_pos] + velocity) / q;
				}
				else
				{
					u_solid = (q - (T)1.0) * st.u_L[x_B_pos] + velocity + ((T)1.0 - q) * ((q - (T)1.0) * st.u_L[x_F_pos] + (T)2.0 * velocity) / ((T)1.0 + q);
				}


				T f_s_eq = sd.w(i) * st.rho_L[x_B_pos] * ((T)1.0 + dot(sd.c(i), u_solid) / cs_sq<T> +dot(sd.c(i), u_solid) *
					dot(sd.c(i), u_solid) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_solid, u_solid) / ((T)2.0 * cs_sq<T>));

				T f_s_neq;

				if (q > (T) 3.0 / 4.0)
				{
					T f_eq_B = sd.w(i) * st.rho_L[x_B_pos] * ((T)1.0 + dot(sd.c(i), st.u_L[x_B_pos]) / cs_sq<T> +dot(sd.c(i), st.u_L[x_B_pos]) *
						dot(sd.c(i), st.u_L[x_B_pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(st.u_L[x_B_pos], st.u_L[x_B_pos]) / ((T)2.0 * cs_sq<T>));


					f_s_neq = st.f[x_B_pos][i] - f_eq_B;
				}
				else
				{
					T f_eq_B = sd.w(i) * st.rho_L[x_B_pos] * ((T)1.0 + dot(sd.c(i), st.u_L[x_B_pos]) / cs_sq<T> +dot(sd.c(i), st.u_L[x_B_pos]) *
						dot(sd.c(i), st.u_L[x_B_pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(st.u_L[x_B_pos], st.u_L[x_B_pos]) / ((T)2.0 * cs_sq<T>));

					T f_eq_F = sd.w(i) * st.rho_L[x_F_pos] * ((T)1.0 + dot(sd.c(i), st.u_L[x_F_pos]) / cs_sq<T> +dot(sd.c(i), st.u_L[x_F_pos]) *
						dot(sd.c(i), st.u_L[x_F_pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(st.u_L[x_F_pos], st.u_L[x_F_pos]) / ((T)2.0 * cs_sq<T>));

					f_s_neq = q * (st.f[x_B_pos][i] - f_eq_B) + ((T)1.0 - q) * (st.f[x_F_pos][i] - f_eq_F);

				}

				st.f_star[x_S_pos][i] = f_s_neq + f_s_eq;
				
			}
			st.inletGhosCells[x_S_pos][i].isInlet = false;
		}
	}
}

template<typename T, size_t D>
void LBM_Solver<T,D>::updateMoments()
{
	CustomVecLength<T, D> vecLength;
	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
		{
			st.rho_L[pos] = 0.0;

			if (!ibm)
			{
				st.u_L[pos] = {};

				for (int i = 0; i < sd.getQ(); ++i)
				{
					st.rho_L[pos] += st.f[pos][i];
					st.u_L[pos] += sd.c(i) * st.f[pos][i];
				}

				st.u_L[pos] = (st.u_L[pos] + (T)0.5 * st.F_ext_L[pos]) / st.rho_L[pos];
			}
			else
			{
				st.u_unc_L[pos] = {};

				for (int i = 0; i < sd.getQ(); ++i)
				{
					st.rho_L[pos] += st.f[pos][i];
					st.u_unc_L[pos] += sd.c(i) * st.f[pos][i];
				}
				
				st.u_unc_L[pos] /= st.rho_L[pos];
				st.u_L[pos] = st.u_unc_L[pos] + (T)0.5 * st.F_ext_L[pos]/ st.rho_L[pos];
				st.F_ext_L[pos] = {};
			}

			st.uMag[pos] = vecLength.length(st.u_L[pos]);

		}

	if (ibm)
	{
		std::vector<Fs_DATA<T, D>> Fs_dataList;
		ibm->calcFs(sd, st.u_unc_L.get(), st.rho_L.get(), Fs_dataList);

		for (int i = 0; i < Fs_dataList.size(); ++i)
		{
			int pos_L = Fs_dataList.at(i).pos_L;
			vec<T,D> Fs = Fs_dataList.at(i).Fs;
			st.F_ext_L[pos_L] += Fs;
		}	
	}
}

template<typename T, size_t D>
void LBM_Solver<T, D>::setParticleGenerator(std::unique_ptr<ParticleGenerator<T, D>> particleGenerator, bool showParticleDensity)
{
	this->showParticleDensity = showParticleDensity;
	this->particleGenerator = std::move(particleGenerator);
	this->st.allocateParticleDensity();
}

template<typename T, size_t D>
void LBM_Solver<T, D>::enableBGK()
{
	this->BGKused = true;
}

template<typename T, size_t D>
void LBM_Solver<T, D>::find_UMax(T& uMax_mag)
{
	uMax_mag = 0;
	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
		if (uMax_mag < st.uMag[pos])
			uMax_mag = st.uMag[pos];
}

template<typename T, size_t D>
void LBM_Solver<T, D>::disableAdaptiveTimeStep() { this->adaptiveTimeStep = false; }

template<typename T, size_t D>
void LBM_Solver<T, D>::rescaleMoments_BGK(const T& scale_u, const T& scale_F)
{
	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
	{
		vec<T, D> u_L_old = st.u_L[pos];
		st.u_L[pos] *= scale_u;
		st.F_ext_L[pos] *= scale_F;

		for (int i = 0; i < sd.getQ(); ++i)
		{
			T f_old = st.f[pos][i];
			T f_eq_old = sd.w(i) * st.rho_L[pos] * ((T)1.0 + dot(sd.c(i), u_L_old) / cs_sq<T> +dot(sd.c(i), u_L_old) *
				dot(sd.c(i), u_L_old) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(u_L_old, u_L_old) / ((T)2.0 * cs_sq<T>));
			st.f[pos][i] = sd.w(i) * st.rho_L[pos] * ((T)1.0 + dot(sd.c(i), st.u_L[pos]) / cs_sq<T> +dot(sd.c(i), st.u_L[pos]) *
				dot(sd.c(i), st.u_L[pos]) / ((T)2.0 * cs_sq<T> *cs_sq<T>) - dot(st.u_L[pos], st.u_L[pos]) / ((T)2.0 * cs_sq<T>));

			st.f[pos][i] += ((f_eq_old - f_old) / f_eq_old) * st.f[pos][i];
		}
	}
}

template<typename T, size_t D>
void LBM_Solver<T, D>::rescaleMoments_CM(const T& scale_u, const T& scale_F)
{
	for (int pos = 0; pos < sd.getMaxNodeCount(); ++pos)
	{
		vec<T, D> u_L_old = st.u_L[pos];
		st.u_L[pos] *= scale_u;
		st.F_ext_L[pos] *= scale_F;

		for (int i = 0; i < sd.getQ(); ++i)
		{
			T f_old = st.f[pos][i];
			T f_eq_old = calc_f_eq_CM(i, st.rho_L[pos], u_L_old);
			st.f[pos][i] = calc_f_eq_CM(i, st.rho_L[pos], st.u_L[pos]);

			st.f[pos][i] += ((f_eq_old - f_old) / f_eq_old) * st.f[pos][i];
		}
	}
}

template<typename T, size_t D>
const std::string LBM_Solver<T, D>::getClockOutput() const
{
	return oStringStr.str();
}

template<typename T, size_t D>
void LBM_Solver<T, D>::solve()
{
	using namespace std::chrono;
	bool visualData = false;
	int percent = 0;
	std::string loadingPoints = "";
	T dt = this->sd.getTimeStep();
	T dh = this->sd.getGridSize();
	T rho = this->sd.getRho();
	T C_u = this->sd.getC_u();
	T viscosity_L = this->sd.getViscosity_L();
	vec<grid_size_t, 3> gridDim_L{ this->sd.getGridDim_L(0), this->sd.getGridDim_L(1), this->sd.getGridDim_L(2) };
	T uMax_mag = 0, scale_u = (T)1.0, scale_F = (T)1.0;
	T currentSimulationTime = 0;
	auto t_solve_0 = steady_clock::now();

	oStringStr << "\n\nVERISON WITH SEQUENTIAL CODE\n-----------------\n";
	try
	{
		if (this->ibm)
			this->ibm->reset(dh, C_u, rho);

		std::cout << "Calculating: \n" << loadingPoints << " 0%";
		while (currentSimulationTime <= maxSimulationTime)
		{
			int maxSimulationTimeSteps = (int)(maxSimulationTime / this->sd.getTimeStep());

			int percentTemp = (int)((currentSimulationTime / maxSimulationTime)*100.0);
			if (percent != percentTemp)
			{
				percent = percentTemp;
				loadingPoints += ".";
				std::cout << "\r";
				std::cout << loadingPoints << " " << percent << "%";
			}

			streaming();
			updateMoments();

			if (this->writer && simulationStep % this->writer->getStepsPerFrame() == 0)
			{
				if (this->writer->getDestination() == VTK_FILE)
					visualData = this->writer-> writeMomentsToVTKFile(this->sd, this->st, currentSimulationTime, simulationStep);
				else 
					visualData = this->writer->writeMomentsToTXTFile(this->sd, this->st, currentSimulationTime, simulationStep);

				if (this->ibm)
					this->writer->writeImBody(this->sd, *this->ibm, simulationStep);

				if(particleGenerator)
					this->writer->writeParticles(this->sd, this->st, *this->particleGenerator, currentSimulationTime, simulationStep, this->showParticleDensity);
			}

			if (particleGenerator)
			{
				particleGenerator->updateParticles(simulationStep, this->st.u_L.get(), this->st.rho_L.get(), viscosity_L, dh, dt, gridDim_L);
				if(this->showParticleDensity)
					particleGenerator->calcParticleDensity(simulationStep, this->st.particleDensity.get(), this->sd.getMaxNodeCount(), gridDim_L);
			}
			
			if (this->adaptiveTimeStep)
			{
				find_UMax(uMax_mag);

				if (uMax_mag > 0 && (uMax_mag < sd.get_uRef_L() - 0.02 || uMax_mag > sd.get_uRef_L() + 0.02))
				{
					this->sd.rescaleConFactors(uMax_mag, scale_u, scale_F);

					if (BGKused)
						this->rescaleMoments_BGK(scale_u, scale_F);
					else
						this->rescaleMoments_CM(scale_u, scale_F);

					if (ibm)
						ibm->rescaleVelocities(scale_u);

					if (this->particleGenerator)
						this->particleGenerator->rescaleParticleData(scale_u);
				}
			}

			if (BGKused)
			{
				collision_BGK();

				if(this->ibm)
					ibm->calcInlet(sd, st.inletGhosCells.get());

				collision_ghost_BGK();
			}
			else
			{
				collision_CM();

				if (this->ibm)
					ibm->calcInlet(sd, st.inletGhosCells.get());

				collision_ghost_CM();
			}

			
			
			if (this->ibm && ibm->getTag() == IBM_DYNAMIC)
			{
				this->ibm->update(simulationStep);
			}

			++simulationStep;
			currentSimulationTime += this->sd.getTimeStep();
		}
		loadingPoints += ".";
		std::cout << "\r";
		std::cout << loadingPoints << " " << 100 << "%";
		if (visualData && this->ibm)
			this->writer->writePVDFile(*this->ibm, this->showParticleDensity);
		else if (visualData)
			this->writer->writePVDFile(this->showParticleDensity);
	}
	catch (const std::exception e)
	{
		std::cerr << e.what() << " Timestep: " << simulationStep << std::endl;
	}

	oStringStr << "Maximal-Simulation-Steps: " << simulationStep << "\n";

	auto t_solve_1 = steady_clock::now();
	oStringStr << "Duration of the whole calculation: " << duration<double, std::milli>{ t_solve_1 - t_solve_0 }.count() << "ms\n";
}
