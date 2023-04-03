// ========================================================================= //
//                                                                           //
// Filename: Real.h                                                     
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2013 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 12.2017 Tim Grasser
//                                                                           //
// ========================================================================= //

#pragma once

#ifndef CPUGPU
#ifdef __CUDACC__
#define CPUGPU __device__ __host__
#else
#define CPUGPU
#endif
#endif

//Dont use this, unless you know exactly what you are doing.
//It might destroy integrity of derivatives, if used wrongly.
//Mit anderen Worten: Unueberlegte Nutzung steht unter zweifacher Todesstrafe! :)
template<typename real> CPUGPU float unsafe_Convert_ToFloat(real in)
{
	return static_cast<float>(in);
}

template<typename real> CPUGPU double unsafe_Convert_ToDouble(real in)
{
	return static_cast<double>(in);
}


//Kombinatorisch möchte ich das nicht machen. Entweder oder.
#if defined(WITH_ADOL_C) && !defined(__CUDACC__)

//known issue: Warning	C4244	'argument': conversion from 'std::size_t' to 'double', possible loss of data
//here you cannot cast from size_t to adouble. you have to cast like this: static_cast<real>(static_cast<double>(numDOF_))
#pragma warning(push)
#pragma warning (disable: 4267)
#include <adolc/adouble.h> 
#pragma warning(pop)
#define TEMPLATE_INSTANTIATE_FOR_SCALARS(txt)  template txt<float>;	template txt<double>; template txt<adouble>;
#define EXECUTE_MACRO_FOR_ALL_SCALARS(macrotext) macrotext(float) macrotext(double) macrotext(adouble)
float unsafe_Convert_ToFloat(adouble const & in);
double unsafe_Convert_ToDouble(adouble const & in);

#else
#if defined(WITH_CPPAD) && !defined(__CUDACC__)
#pragma warning(push)
#pragma warning(disable: 4396)
#include <Eigen/Core>
#include <cppad/example/cppad_eigen.hpp>
#include <cppad/utility/sparse2eigen.hpp>
//#include <cppad/ipopt/solve.hpp>
//#include <cppad/cppad.hpp>
#pragma warning(pop)
#define TEMPLATE_INSTANTIATE_FOR_SCALARS(txt)  template txt<float>;	template txt<double>; template txt<CppAD::AD<double>>;
#define EXECUTE_MACRO_FOR_ALL_SCALARS(macrotext) macrotext(float) macrotext(double) macrotext(CppAD::AD<double>)
float unsafe_Convert_ToFloat(CppAD::AD<double> const & in);
double unsafe_Convert_ToDouble(CppAD::AD<double> const & in);
std::vector<double> unsafe_Convert_ToDoubleVec(std::vector < CppAD::AD<double>> const & in);


#else //just basic scalars

#define TEMPLATE_INSTANTIATE_FOR_SCALARS(txt)  template txt<float>;	template txt<double>;
#define EXECUTE_MACRO_FOR_ALL_SCALARS(macrotext) macrotext(float) macrotext(double)

#endif
#endif

