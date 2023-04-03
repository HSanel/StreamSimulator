// ========================================================================= //
//                                                                           //
// Filename: locationArrayMath.cpp                                      
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2020 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 07.2020 Andreas Giebel                                     
//                                                                           //
// ========================================================================= //
#include <location.h>
#include "locationArrayMath.h"

#include <random>
#include <limits>
#include <omp.h>

#include <cstring> // for memset

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
Real dot(location_cpu, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> *, int numNodes)
{
	Real sum = Real(0);
#pragma omp parallel for reduction(+ : sum)
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			sum += vec1[i][e] * vec2[i][e];
		}
	}
	return sum;
}

#define INSTANTIATE_DOT(Real, E) template Real dot<Real, E>(location_cpu, array<Real, E> const *, array<Real, E> const *, array<Real, E> *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_DOT);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
Real sum(location_cpu, array<Real, E> const * vec, int numNodes)
{
	Real sum = Real(0);
#pragma omp parallel for reduction(+ : sum)
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			sum += vec[i][e];
		}
	}
	return sum;
}

#define INSTANTIATE_SUM(Real, E) template Real sum<Real, E>(location_cpu, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SUM);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
Real max(location_cpu, array<Real, E> const * vec, int numNodes)
{
	Real maximum = -std::numeric_limits<Real>::max();

	//allocate an array with the size of the max number of threads
	const int threads = omp_get_max_threads();
	Real * threadMax = new Real[threads];
	for (int i = 0; i < threads; i++) {
		threadMax[i] = maximum;
	}

	//Visual Studio doesn't support OpenMP max reduction operator
	//creating own max reduction
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		int && thread = omp_get_thread_num();
		for (int e = 0; e < E; e++) {
			const Real & elem = vec[i][e];
			if (elem > threadMax[thread])
				threadMax[thread] = elem;
		}
	}

	//reduce thread local maxima
	for (int i = 0; i < threads; i++) {
		Real & elem = threadMax[i];
		if (elem > maximum)
			maximum = elem;
	}
	delete[] threadMax;
	return maximum;
}

#define INSTANTIATE_MAX(Real, E) template Real max<Real, E>(location_cpu, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MAX);

template<typename Real, int E>
void abs(location_cpu, array<Real, E> * result, array<Real, E> const * vec, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = std::abs(vec[i][e]);
		}
	}
}

#define INSTANTIATE_ABS(Real, E) template void abs<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_ABS);

template<typename Real, int E>
void addVector(location_cpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = vec1[i][e] + vec2[i][e];
		}
	}
}

#define INSTANTIATE_ADDVECTOR(Real, E) template void addVector<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_ADDVECTOR);


/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void dotDevice(location_cpu, Real * result, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp1, array<Real, E> *, int numNodes)
{
	*result = dot<Real, E>(location_cpu(), vec1, vec2, temp1, numNodes);
}

#define INSTANTIATE_DOT_DEV(Real, E) template void dotDevice<Real, E>(location_cpu, Real *, array<Real, E> const *, array<Real, E> const *, array<Real, E> *, array<Real, E> *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_DOT_DEV);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void sumDevice(location_cpu, Real * result, array<Real, E> const * vec, array<Real, E> *, int numNodes)
{
	*result = sum<Real, E>(location_cpu(), vec, numNodes);
}

#define INSTANTIATE_SUM_DEV(Real, E) template void sumDevice<Real, E>(location_cpu, Real *, array<Real, E> const *, array<Real, E> *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SUM_DEV);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void multScalarDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = *scalar * vec[i][e];
		}
	}
}

#define INSTANTIATE_MULTSCALARDEV(Real, E) template void multScalarDevice<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MULTSCALARDEV);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void multScalar(location_cpu, array<Real, E> * result, array<Real, E> const * vec, Real scalar, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = scalar * vec[i][e];
		}
	}
}

#define INSTANTIATE_MULTSCALAR(Real, E) template void multScalar<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, Real, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MULTSCALAR);

template<typename Real, int E>
void multScalar(location_cpu, array<Real, E> * result, Real scalar, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] *= scalar;
		}
	}
}

#define INSTANTIATE_MULTSCALAR2(Real, E) template void multScalar<Real, E>(location_cpu, array<Real, E> *, Real, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MULTSCALAR2);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void divScalarDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = vec[i][e] / *scalar;
		}
	}
}


#define INSTANTIATE_DIVSCALAR_DEV(Real, E) template void divScalarDevice<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_DIVSCALAR_DEV);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void subVector(location_cpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = vec1[i][e] - vec2[i][e];
		}
	}
}

#define INSTANTIATE_SUBVECTOR(Real, E) template void subVector<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SUBVECTOR);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void alphaXplusYDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = *alpha * vecX[i][e] + vecY[i][e];
		}
	}
}

#define INSTANTIATE_AXPY_DEV(Real, E) template void alphaXplusYDevice<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_AXPY_DEV);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void alphaXplusY(location_cpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real alpha, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = alpha * vecX[i][e] + vecY[i][e];
		}
	}
}

#define INSTANTIATE_AXPY(Real, E) template void alphaXplusY<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, Real, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_AXPY);

/**
* All pointers reside in the same memory Location
*/
template<typename Real, int E>
void minusAlphaXplusYDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			result[i][e] = -*alpha * vecX[i][e] + vecY[i][e];
		}
	}
}

#define INSTANTIATE_MAXPY(Real, E) template void minusAlphaXplusYDevice<Real, E>(location_cpu, array<Real, E> *, array<Real, E> const *, array<Real, E> const *, Real *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_MAXPY);

template<typename Real, int E>
void setZero(location_cpu, array<Real, E> * vec, int numNodes)
{
	std::memset(vec, 0, numNodes * E * sizeof(Real));
}

#define INSTANTIATE_SETZERO(Real, E) template void setZero<Real, E>(location_cpu, array<Real, E> *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_SETZERO);

template<typename Real, int E>
void fill(location_cpu, array<Real, E> * vec, Real value, int numNodes)
{
#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		for (int e = 0; e < E; e++) {
			vec[i][e] = value;
		}
	}
}

#define INSTANTIATE_FILL(Real, E) template void fill<Real, E>(location_cpu, array<Real, E> *, Real, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_FILL);

template<typename Real, int E>
void randomize(location_cpu, array<Real, E> * vec, int numNodes)
{
	const int threads = omp_get_max_threads(); //Possibility of allocating for too many threads, but ensures that always enough is being allocated
	std::default_random_engine * gen = new std::default_random_engine[threads];
	std::uniform_real_distribution<Real> * dist = new std::uniform_real_distribution<Real>[threads];
	for (int i = 0; i < threads; i++) {
		gen[i] = std::default_random_engine(i);
		dist[i] = std::uniform_real_distribution<Real>(Real(-1.), Real(1.));
	}

#pragma omp parallel for
	for (int i = 0; i < numNodes; i++) {
		int && thread = omp_get_thread_num();
		for (int e = 0; e < E; e++)
			vec[i][e] = dist[thread](gen[thread]);
	}
	delete[] gen;
	delete[] dist;
}

#define INSTANTIATE_RANDOM(Real, E) template void randomize<Real, E>(location_cpu, array<Real, E> *, int);

EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(INSTANTIATE_RANDOM);

#ifdef WITH_DOUBLE_PRECISION
template<int E>
void convertDoubleToFloatArray(location_cpu, array<float, E> * vecOut, array<double, E> const * vecIn, int numNodes)
{
	for(int i = 0; i < numNodes; ++i)
	{
		for(int j = 0; j < E; ++j)
			vecOut[i][j] = static_cast<float>(vecIn[i][j]);
	}
}

template void convertDoubleToFloatArray<3>(location_cpu, array<float, 3> *, array<double, 3> const *, int);

#endif
