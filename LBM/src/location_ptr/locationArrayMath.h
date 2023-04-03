// ========================================================================= //
//                                                                           //
// Filename: locationArrayMath.h                                      
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
#pragma once

#include "array.h"
#include "location.h"

/*
* Device functions require all pointers to be from the same device (either all in CPU or GPU memory), even the pointer for scalars like in axpy etc.
* On the GPU this eliminates the need of a memcpy from device to host or vice versa.
*/

template<typename Real, int E>
void dotDevice(location_cpu, Real * result, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp, array<Real, E> * temp2, int numNodes);

template<typename Real, int E>
Real dot(location_cpu, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp, int numNodes);

template<typename Real, int E>
Real sum(location_cpu, array<Real, E> const * vec, int numNodes);

template<typename Real, int E>
Real max(location_cpu, array<Real, E> const * vec, int numNodes);

template<typename Real, int E>
void abs(location_cpu, array<Real, E> * result, array<Real, E> const * vec, int numNodes);

template <typename Real, int E>
void addVector(location_cpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes);

template<typename Real, int E>
void sumDevice(location_cpu, Real * result, array<Real, E> const * vec, array<Real, E> * temp, int numNodes);

template<typename Real, int E>
void alphaXplusYDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes);

template<typename Real, int E>
void alphaXplusY(location_cpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real alpha, int numNodes);

template<typename Real, int E>
void minusAlphaXplusYDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes);

template<typename Real, int E>
void multScalarDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes);

template<typename Real, int E>
void multScalar(location_cpu, array<Real, E> * result, array<Real, E> const * vec, Real scalar, int numNodes);

template<typename Real, int E>
void multScalar(location_cpu, array<Real, E> * result, Real scalar, int numNodes);

template<typename Real, int E>
void divScalarDevice(location_cpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes);

template<typename Real, int E>
void subVector(location_cpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes);

template<typename Real, int E>
void setZero(location_cpu, array<Real, E> * vec, int numNodes);

template<typename Real, int E>
void fill(location_cpu, array<Real, E> * vec, Real value, int numNodes);

template<typename Real, int E>
void randomize(location_cpu, array<Real, E> * vec, int numNodes);

#ifdef WITH_DOUBLE_PRECISION
template<int E>
void convertDoubleToFloatArray(location_cpu, array<float, E>* vecOut, array<double, E> const* vecIn, int numNodes);
#endif

#ifdef WITH_CUDA
template<typename Real, int E>
void dotDevice(location_gpu, Real * result, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp, array<Real, E> * temp2, int numNodes);

template<typename Real, int E>
Real dot(location_gpu, array<Real, E> const * vec1, array<Real, E> const * vec2, array<Real, E> * temp, int numNodes);

template<typename Real, int E>
Real sum(location_gpu, array<Real, E> const * vec, int numNodes);

template<typename Real, int E>
Real max(location_gpu, array<Real, E> const * vec, int numNodes);

template<typename Real, int E>
void abs(location_gpu, array<Real, E> * result, array<Real, E> const * vec, int numNodes);

template <typename Real, int E>
void addVector(location_gpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes);

template<typename Real, int E>
void sumDevice(location_gpu, Real * result, array<Real, E> const * vec, array<Real, E> * temp, int numNodes);

template<typename Real, int E>
void alphaXplusYDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes);

template<typename Real, int E>
void alphaXplusY(location_gpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real alpha, int numNodes);

template<typename Real, int E>
void minusAlphaXplusYDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vecX, array<Real, E> const * vecY, Real * alpha, int numNodes);

template<typename Real, int E>
void multScalarDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes);

template<typename Real, int E>
void multScalar(location_gpu, array<Real, E> * result, array<Real, E> const * vec, Real scalar, int numNodes);

template<typename Real, int E>
void multScalar(location_gpu, array<Real, E> * result, Real scalar, int numNodes);

template<typename Real, int E>
void divScalarDevice(location_gpu, array<Real, E> * result, array<Real, E> const * vec, Real * scalar, int numNodes);

template<typename Real, int E>
void subVector(location_gpu, array<Real, E> * result, array<Real, E> const * vec1, array<Real, E> const * vec2, int numNodes);

template<typename Real, int E>
void setZero(location_gpu, array<Real, E> * vec, int numNodes);

template<typename Real, int E>
void fill(location_gpu, array<Real, E> * vec, Real value, int numNodes);

template<typename Real, int E>
void randomize(location_gpu, array<Real, E> * vec, int numNodes);

#ifdef WITH_DOUBLE_PRECISION
template<int E>
void convertDoubleToFloatArray(location_gpu, array<float, E> * vecOut, array<double, E> const * vecIn, int numNodes);
#endif

#endif // WITH_CUDA

template<typename Real, int E>
void applyPermutation(location_gpu, array<Real, E> * vec, array<Real, E> const * vecIn, int const * permutation, int numNodes);

void computeInversePermutation(location_gpu, int * inversePermutation, int const * permutation, int numNodes);


#define EXECUTE_MACRO_FOR_ALL_ARRAY_TYPES(macrotext) macrotext(float, 3) macrotext(float, 6) macrotext(double, 3) macrotext(double, 6)