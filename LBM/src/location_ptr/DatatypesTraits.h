// ========================================================================= //
//                                                                           //
// Filename: DatatypesTraits.h
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2012 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 09.2012 Daniel Weber
//                                                                           //
// ========================================================================= //

#ifndef DATATYPESTRAITS_H_INCLUDED
#define DATATYPESTRAITS_H_INCLUDED

#ifndef CPUGPU
#ifdef __CUDACC__
#define CPUGPU __device__ __host__
#else
#define CPUGPU
#endif
#endif

#ifndef __CUDACC__
#include "Real.h"
#endif 

#include <cmath>
#include <vector>

template <typename Real>
struct ScalarTrait
{
	typedef Real BuiltInType; //Without AutoDiff or similar
};

#if defined(WITH_CPPAD) && !defined(__CUDACC__) // cudacompiler cannot handle cppad and should not.
template <>
struct ScalarTrait<CppAD::AD<double>>
{
	typedef double BuiltInType; //Without AutoDiff or similar
};
#endif

#if defined(WITH_ADOL_C) && !defined(__CUDACC__)
template <>
struct ScalarTrait<adouble>
{
	typedef double BuiltInType; //Without AutoDiff or similar
};
#endif

template <typename real>
struct Trait
{
	typedef real mType; //mType = VectorTyp - klar ne
	typedef typename ScalarTrait<real>::BuiltInType mTypeElementar;
	typedef float constraintEncodingType;
	typedef real scalar;	
		
	static const unsigned int mTypeSize;
	static const unsigned int size;
	
	static const real zeroElem;
	static const real identityElem;

	static CPUGPU inline real abs(const real& rhs) {
		using std::fabs;
		return fabs(rhs);
	}
	static inline real max(const real& rhs, const real& lhs) {
		return rhs > lhs ? rhs : lhs;
	}

	static inline real transposedMult(const real& element, const real& rhs) {
		return element *rhs;
	}
	static inline real dot (real a, real b) {
		return a * b;
	}
	static inline void getTransposed(const real& element, real& result) {
		result = element;
	}
	static inline void addAbsInRow(const real& element, real& result) {
		result += fabs(element);
	}
	static inline bool allElementsLargerEqual(const real& a, const real& b) {
		return (a>=b);
	}
	static inline void transpose(real & ioValue) {}
	static inline real transposed(const real &inValue) {
		return inValue;
	}
	static inline real getValue(const real &inValue, int inRow, int inCol) {
		return inValue;
	}
	//static inline void smoothWithEps(real & ioValue, real inEps) {
	//    ioValue = abs(ioValue) < inEps ? 0.0f : ioValue;
	//}
	static inline void multInvDiagX(const real &inDiagElement, const real &inRhs, real &outResult) {
		outResult = (1.0f / inDiagElement) * inRhs;
	}
	static inline std::vector<real> diag(const std::vector<real> &inDiag) {
		return inDiag;
	}
	static inline void AddMatMultXPtr(const real &inMat, const real *inRhs, real *outResult) {
		(*outResult) += inMat * (*inRhs);
	}
	static inline void multInvDiagXPtr(const real &inDiagElement, const real *inRhs, real *outResult) {
		(*outResult) = (1.0f / inDiagElement) * (*inRhs);
	}
};



#include "Vector3.h"
#include "Matrix3.h"
//-------------------------------------------------------
// MATRIX3 TRAITS
//-------------------------------------------------------

template <typename real>
struct Trait<Matrix3T<real>>
{
	typedef Vector3T<real> mType;
	typedef Vector3T<typename ScalarTrait<real>::BuiltInType> mTypeElementar;
	typedef Vector3T<float> constraintEncodingType;
	typedef real scalar;

	static const unsigned int mTypeSize;
	static const unsigned int size;
	static const Matrix3T<real> zeroElem;
	static const Matrix3T<real> identityElem;

	static inline Matrix3T<real> abs(const Matrix3T<real>& rhs) {
		return rhs.abs();
	}
	static inline Vector3T<real> transposedMult(const Matrix3T<real>& element, const Vector3T<real>& rhs) {
		return element.transposedMult(rhs);
	}
	static inline void getTransposed(const Matrix3T<real>& element, Matrix3T<real>& result) {
		element.transposed(result);
	}
	static inline void addAbsInRow(const Matrix3T<real>& element, Vector3T<real>& result) {
		result[0] += fabs(element[0][0])+fabs(element[0][1])+fabs(element[0][2]);
		result[1] += fabs(element[1][0])+fabs(element[1][1])+fabs(element[1][2]);
		result[2] += fabs(element[2][0])+fabs(element[2][1])+fabs(element[2][2]);
	}
	static inline bool allElementsLargerEqual(const Vector3T<real>& a, const Vector3T<real>& b) {
		return ((a[0]>=b[0])&&(a[1]>=b[1])&&(a[2]>=b[2]));
	}
	static inline void transpose(Matrix3T<real> & ioValue) { ioValue.transpose(); }
	
	static inline Matrix3T<real> transposed(const Matrix3T<real> &inValue) {
		Matrix3T<real> t(inValue);
		t.transpose();
		return t;
	}
	static inline real getValue(const Matrix3T<real> &inValue, int inRow, int inCol) {
		return inValue[inRow][inCol];
	}
	static inline void multInvDiagX(const Matrix3T<real> &inDiagElement, const Vector3T<real> &inRhs,
		Vector3T<real> &outResult) {
		outResult[0] = (1.0f / inDiagElement[0][0]) * inRhs[0];
		outResult[1] = (1.0f / inDiagElement[1][1]) * inRhs[1];
		outResult[2] = (1.0f / inDiagElement[2][2]) * inRhs[2];
	}
	static inline std::vector<real> diag(const std::vector<Matrix3T<real>> &inDiag) {
	   std::vector<real> result(inDiag.size() * 3);
	   for (size_t y = 0; y < inDiag.size(); ++y) {
		   result[3*y + 0] = inDiag[y][0][0];
		   result[3*y + 1] = inDiag[y][1][1];
		   result[3*y + 2] = inDiag[y][2][2];
	   }
		return result;
	}

	static inline void AddMatMultXPtr(const Matrix3T<real> &inMat, const real *inRhs, real *outResult) {
	   outResult[0] +=   inMat[0][0] * inRhs[0]
					   + inMat[0][1] * inRhs[1]
					   + inMat[0][2] * inRhs[2];

	   outResult[1] +=   inMat[1][0] * inRhs[0]
					   + inMat[1][1] * inRhs[1]
					   + inMat[1][2] * inRhs[2];

	   outResult[2] +=   inMat[2][0] * inRhs[0]
					   + inMat[2][1] * inRhs[1]
					   + inMat[2][2] * inRhs[2];
	}
	static inline void multInvDiagXPtr(const Matrix3T<real> &inDiagElement,
									  const real *inRhs,
		real *outResult) {
		outResult[0] = ( 1.0f / inDiagElement[0][0] ) * inRhs[0];
		outResult[1] = ( 1.0f / inDiagElement[1][1] ) * inRhs[1];
		outResult[2] = ( 1.0f / inDiagElement[2][2] ) * inRhs[2];
	}

};


//-------------------------------------------------------
// Vector3 TRAITS
//-------------------------------------------------------


template <typename real>
struct Trait<Vector3T<real>>
{
	static const unsigned int size;
	const static Vector3T<real> zeroElem;
	const static Vector3T<real> identityElem;
	typedef real scalar;
	static inline real dot(const Vector3T<real>& a, const Vector3T<real>& b) {
		return a.dot(b);
	}
};

#endif

