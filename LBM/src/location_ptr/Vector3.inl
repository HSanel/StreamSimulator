// ========================================================================= //
//                                                                           //
// Filename: Vector3T.inl
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

#include <cstdio>
#include "DatatypesTraits.h"

template<typename real>
Vector3T<real>::Vector3T()
{
	values[0] = values[1] = values[2] = 0.0f;
}

template<typename real>
Vector3T<real>::Vector3T(real x, real y, real z)
{
	values[0] = x;
	values[1] = y;
	values[2] = z;
}

template<typename real>
Vector3T<real>::Vector3T(real value)
{
	values[0] = values[1] = values[2] = value;
}

template<typename real>
const real* Vector3T<real>::getVector() const
{
	return &values[0];
}

template<typename real>
real* Vector3T<real>::getVector()
{
	return &values[0];
}

template<typename real>
real Vector3T<real>::length() const
{
	const real d = sqrLength();
	return sqrt(d);
}

template<typename real>
real Vector3T<real>::sqrLength() const
{
	return values[0] * values[0] +
		   values[1] * values[1] +
		   values[2] * values[2];
}

template<typename real>
void Vector3T<real>::cross(const Vector3T& rhs, Vector3T& result) const
{
	result[0] = values[1] * rhs.values[2] - values[2] * rhs.values[1];
	result[1] = values[2] * rhs.values[0] - values[0] * rhs.values[2];
	result[2] = values[0] * rhs.values[1] - values[1] * rhs.values[0];
}

template<typename real>
void constructOrthonormalBasis(const Vector3T<real>& inVec, Vector3T<real>& dir2, Vector3T<real>& dir3)
{
	Vector3T<real> dir1 = inVec;
	dir1.normalize();

	if (fabs(dir1[0]) > fabs(dir1[2]))
		dir3 = Vector3T<real>(-dir1[1], dir1[0], 0.0f);
	else
		dir3 = Vector3T<real>(0.0f, -dir1[2], dir1[1]);
	dir3.normalize();
	dir2 = cross(dir3, dir1);
}


template<typename real>
Vector3T<real> Vector3T<real>::cross(const Vector3T<real>& rhs) const
{
	return Vector3T<real>(values[1] * rhs.values[2] - values[2] * rhs.values[1],
				   values[2] * rhs.values[0] - values[0] * rhs.values[2],
				   values[0] * rhs.values[1] - values[1] * rhs.values[0]);
}

template<typename real>
inline Vector3T<real> cross(const Vector3T<real> &L, const Vector3T<real> &R)
{
	return L.cross(R);
}

template<typename real>
inline real dot(const Vector3T<real> &L, const Vector3T<real> &R)
{
	return L.dot(R);
}

template<typename real>
real Vector3T<real>::dot(const Vector3T<real>& rhs) const
{
	return   values[0] * rhs.values[0]
		   + values[1] * rhs.values[1]
		   + values[2] * rhs.values[2];
}


template<typename real>
Vector3T<real> Vector3T<real>::operator -(const Vector3T<real>& rhs) const
{
	return Vector3T<real>(values[0] - rhs.values[0],
				   values[1] - rhs.values[1],
				   values[2] - rhs.values[2]);
}

template<typename real>
Vector3T<real> Vector3T<real>::operator -() const
{
	return Vector3T<real>(-values[0], -values[1], -values[2]);
}


template<typename real>
Vector3T<real> Vector3T<real>::operator +(const Vector3T<real>& rhs) const
{
	return Vector3T<real>(values[0] + rhs.values[0],
				   values[1] + rhs.values[1],
				   values[2] + rhs.values[2]);
}

template<typename real>
void Vector3T<real>::normalize(void)
{
	const real kNormalize = 1.0f / length();
	values[0] *= kNormalize;
	values[1] *= kNormalize;
	values[2] *= kNormalize;
}

template<typename real>
Vector3T<real> Vector3T<real>::normalized() const
{
	Vector3T<real> result(*this);
	result.normalize();
	return result;
}

template<typename real>
Vector3T<real> Vector3T<real>::operator* (const real rhs) const
{
	return Vector3T<real>(values[0] * rhs,
				   values[1] * rhs,
				   values[2] * rhs);
}

template<typename real>
Vector3T<real> Vector3T<real>::operator/ (const real rhs) const
{
	const real inverse = 1.0f / rhs; //should be faster
	return *this * inverse;
}

template<typename real>
real Vector3T<real>::operator* (const Vector3T<real>& rhs) const
{
	return   values[0] * rhs.values[0]
		   + values[1] * rhs.values[1]
		   + values[2] * rhs.values[2];
}

template<typename real>
Vector3T<real>& Vector3T<real>::operator +=(const Vector3T<real>& rhs)
{
	values[0] += rhs.values[0];
	values[1] += rhs.values[1];
	values[2] += rhs.values[2];
	return *(this);
}


template<typename real>
Vector3T<real> Vector3T<real>::elementWiseMultiply(const Vector3T<real>& in) const 
{
	return Vector3T<real>(values[0] * in[0], values[1] * in[1], values[2] * in[2]);
}

template<typename real>
Vector3T<real>& Vector3T<real>::operator /=(const real rhs)
{
	values[0] /= rhs;
	values[1] /= rhs;
	values[2] /= rhs;
	return *(this);
}

template<typename real>
Vector3T<real>& Vector3T<real>::operator -=(const Vector3T<real> &rhs)
{
	values[0] -= rhs.values[0];
	values[1] -= rhs.values[1];
	values[2] -= rhs.values[2];
	return *this;
}

template<typename real>
Vector3T<real>& Vector3T<real>::operator *=(const real rhs)
{
	values[0] *= rhs;
	values[1] *= rhs;
	values[2] *= rhs;
	return *this;
}

template<typename real>
bool Vector3T<real>::operator==(const Vector3T<real>& vgl) const
{
	static const real kEpsilon = 1e-6f;

	for(int i = 0; i < 3; i++)
	{
		if(Trait<real>::abs(values[i] - vgl.values[i]) > kEpsilon) {
			return false;
		}
	}
	return true;
}

template<typename real>
bool Vector3T<real>::operator !=(const Vector3T<real>& inOther) const
{
	return !(*this == inOther);
}

template<typename RealA, typename RealB>
auto inline operator *(RealA inLeft, Vector3T<RealB> const & inRight)
	-> typename std::enable_if<std::is_convertible<RealA, RealB>::value, Vector3T<RealB>>::type
{
	return Vector3T<RealB>(
		inLeft * inRight[0],
		inLeft * inRight[1],
		inLeft * inRight[2]
	);
}

template<typename real>
real& Vector3T<real>::operator [](int index)
{
	return values[index];
}

template<typename real>
const real& Vector3T<real>::operator [](int index) const
{
	return values[index];
}

template<typename real>
void Vector3T<real>::generateComplementBasis(Vector3T<real>& u, Vector3T<real>& v, const Vector3T<real>& w)
{
	if(fabs(w[0]) >= fabs(w[1]))
	{
		real invLength = 1.0f / sqrt(w[0]*w[0] + w[2]*w[2]);
		u[0] = -w[2] * invLength;
		u[1] = 0.0f;
		u[2] = w[0] * invLength;
		v[0] = w[1]*u[2];
		v[1] = w[2] * u[0] - w[0] * u[2];
		v[2] = - w[1] * u[0];
	}
	else
	{
		real invLength = 1.0f / sqrt(w[1]*w[1] + w[2]*w[2]);
		u[0] = 0.0f;
		u[1] = w[2] * invLength;
		u[2] = -w[1] * invLength;
		v[0] = w[1] * u[2] - w[2] * u[1];
		v[1] = -w[0] * u[2];
		v[2] = w[0] * u[1];
	}

}


template<typename realX>
std::ostream& operator<<(std::ostream& os, const Vector3T<realX>& rhs)
{
	os << rhs.values[0] << '\t'
		<< rhs.values[1] << '\t'
		<< rhs.values[2];
	return os;
}

template<typename realX>
std::istream& operator >> (std::istream& is, Vector3T<realX>& rhs)
{
	is >> rhs.values[0] >> rhs.values[1] >> rhs.values[2];
	return is;
}
