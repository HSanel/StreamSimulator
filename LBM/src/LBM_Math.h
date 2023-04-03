#pragma once
#include <iostream>
#include <array>
#include <string>

#define E3	{ 1, 0, 0,\
			  0, 1, 0,\
			  0, 0, 1}

#define E9 { 1, 0, 0, 0, 0, 0, 0, 0, 0,\
			 0, 1, 0, 0, 0, 0, 0, 0, 0,\
			 0, 0, 1, 0, 0, 0, 0, 0, 0,\
			 0, 0, 0, 1, 0, 0, 0, 0, 0,\
			 0, 0, 0, 0, 1, 0, 0, 0, 0,\
			 0, 0, 0, 0, 0, 1, 0, 0, 0,\
			 0, 0, 0, 0, 0, 0, 1, 0, 0,\
			 0, 0, 0, 0, 0, 0, 0, 1, 0,\
			 0, 0, 0, 0, 0, 0, 0, 0, 1 }

#define E27 { 1, 0, 0,\
			  0, 1, 0,\
			  0, 0, 1}

#define INIT_L(q) ((q==9) ? (3) : (2))

namespace LbmMath
{

	template <typename T,int N>
	void  LU_Dec(std::array<T, N*N>& M, std::array<T, N*N>& L)
	{
		for (int i = 0; i < N - 1; i++)
		{
			T Mii = M.at(i * N + i);
			if (Mii == (T)0.0)
			{
				throw std::invalid_argument("Divide by Zero in LU-Decomposition");
			}

			for (int k = i + 1; k < N; k++)
			{
				L.at(k * N + i) = M.at(k * N + i) / Mii;

				for (int j = i; j < N; j++)
					M.at(k * N + j) = M.at(k * N + j) - L.at(k * N + i) * M.at(i * N + j);
			}
		}
	}

	template <typename T, int N>
	void calc_y(std::array<T, N>& y, std::array<T, N>& b, std::array<T, N*N>& L)
	{
		y.at(0) = b.at(0) / L.at(0);

		for (int i = 1; i < N; i++)
		{
			y.at(i) = (T)0.0;
			for (int k = 0; k < i; k++)
			{
				y.at(i) += L.at(i * N + k) * y.at(k);
			}

			y.at(i) = (b.at(i) - y.at(i)) / L.at(i * N + i);
		}
	};

	template <typename T, int N>
	void calc_x(std::array<T, N>& x, std::array<T, N>& y, std::array<T, N* N>& R)
	{
		x.at(N - 1) = y.at(N - 1) / R.at((N - 1) * N + N - 1);

		for (int i = N - 2; i >= 0; i--)
		{
			x.at(i) = (T)0.0;
			for (int k = i + 1; k < N; k++)
			{
				x.at(i) += R.at(i * N + k) * x.at(k);
			}

			x.at(i) = (y.at(i) - x.at(i)) / R.at(i * N + i);
		}
	}

	template <size_t N>
	void printMat(std::array<float, N*N>& m, std::string name)
	{


		std::cout << "\nMatrix: " << name << "\n-------------------------\n";
		for (int i = 0; i < N; ++i)
		{
			std::cout << "| ";
			for (int j = 0; j < N; ++j)
			{
				std::cout << m.at(i * N + j) << "	";
			}
			std::cout << "|" << "\n";
		}

		std::cout << "\n";
	}


};
