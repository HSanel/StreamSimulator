// ========================================================================= //
//                                                                           //
// Filename: Vector3T.h
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

#pragma once

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>

#ifndef CPUGPU
#ifdef __CUDACC__
#define CPUGPU __device__ __host__
#else
#define CPUGPU
#endif
#endif

//#include <boost/operators.hpp>
template<typename real>
class Vector3T
{
private:
	/** data */
	real values[3];
public:
	/** identity and zero vectors */
	static Vector3T e1, e2, e3;
	typedef real scalar;

	static CPUGPU Vector3T zero() { return Vector3T(); }

	/** constructors */
	CPUGPU inline Vector3T();
	CPUGPU inline Vector3T(real x, real y, real z);
	CPUGPU inline Vector3T(real value);
	Vector3T(real *adress);

	CPUGPU inline real& operator [](int index);
	CPUGPU inline const real& operator [](int index) const;
	CPUGPU inline const real* getVector() const;
	CPUGPU inline real* getVector();

	CPUGPU inline Vector3T elementWiseMultiply(Vector3T const & in) const;

	/** to string */
	inline std::string str() const;

	/** length operations */
	CPUGPU inline real length() const;
	CPUGPU inline real sqrLength() const;
	CPUGPU inline void normalize();
	CPUGPU inline Vector3T normalized() const;
	CPUGPU inline real maxAbsElement()const;

	/** cross product */
	CPUGPU inline void cross(const Vector3T& rhs, Vector3T& result) const;
	CPUGPU inline Vector3T cross(const Vector3T& rhs) const;

	/** dot product */
	CPUGPU inline real dot(const Vector3T& rhs) const;

	/** operators */
	CPUGPU inline Vector3T operator -(const Vector3T& rhs) const;
	CPUGPU inline Vector3T operator -() const;
	CPUGPU inline Vector3T operator +(const Vector3T& rhs) const;
	CPUGPU inline Vector3T operator *(const real rhs) const;
	CPUGPU inline Vector3T operator /(const real rhs) const;
	CPUGPU inline real operator *(const Vector3T& rhs) const;
	CPUGPU inline Vector3T& operator +=(const Vector3T& rhs);
	CPUGPU inline Vector3T& operator -=(const Vector3T& rhs);
	CPUGPU inline Vector3T& operator *=(const real rhs);
	CPUGPU inline Vector3T& operator /=(const real rhs);

	/** comparison */
	CPUGPU inline bool operator == (const Vector3T& vlg) const;
	CPUGPU inline bool operator !=(const Vector3T& inOther) const;

	/** special */
	CPUGPU inline static void generateComplementBasis(Vector3T& u, Vector3T& v, const Vector3T& w);

	/** stream */
	template<typename realX>
	friend std::ostream& operator<<(std::ostream& os, const Vector3T<realX>& rhs);
	template<typename realX>
	friend std::istream& operator >> (std::istream& is, Vector3T<realX>& rhs);

	template<typename RealA, typename RealB>
	friend CPUGPU inline auto operator *(RealA inLeft, Vector3T<RealB> const & inRight)
		-> typename std::enable_if<std::is_convertible<RealA, RealB>::value, Vector3T<RealB>>::type;
};

template<typename real>
inline Vector3T<real> cross(const Vector3T<real> &L, const Vector3T<real> &R);
template<typename real>
inline real dot(const Vector3T<real> &L, const Vector3T<real> &R);

template<typename real>
void constructOrthonormalBasis(const Vector3T<real>& inVec, Vector3T<real>& dir2, Vector3T<real>& dir3);

#include "Vector3.inl"

//If this pops "C2371: redefinition...", you are using a simple "class Vector3;" forward declaration
//Use "template<typename real> class Vector3T; typedef Vector3T<float> Vector3;" instead
typedef Vector3T<float> Vector3; 

