// ========================================================================= //
//                                                                           //
// Filename: Matrix3.h                                                       
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

template<typename real>
class Matrix3T;
#include "Vector3.h"
#include "Real.h"

#include <iostream>
#include <cmath>

template<typename real>
class Matrix3T
{
private:
	Vector3T<real> m_rows[3];
public:
	Matrix3T() = default;
	CPUGPU inline Matrix3T(const Vector3T<real>& row1, const Vector3T<real>& row2, const Vector3T<real>& row3);
	typedef real scalar;

	template<typename real2>
	CPUGPU inline Vector3T<real2> operator*(const Vector3T<real2>& rhs) const;

	CPUGPU inline Vector3T<real> leftMult(const Vector3T<real>& rhs) const;

	CPUGPU inline Vector3T<real> transposedMult(const Vector3T<real>& rhs) const;
	CPUGPU inline void transposedAdd(const Matrix3T& rhs);
	CPUGPU inline Matrix3T transposedMult() const;
	// treat diag as diagonal matrix and perform:
	// this = this * diag
	// returns this
	CPUGPU inline Matrix3T diagMultRight(const Vector3T<real>& diag) const;

	CPUGPU inline Matrix3T& operator *=(const real rhs);
	CPUGPU inline Matrix3T& operator /=(const real rhs);

	CPUGPU inline Matrix3T& operator *=(const Matrix3T& rhs);
	CPUGPU inline Matrix3T& operator +=(const Matrix3T& rhs);
	CPUGPU inline Matrix3T& operator -=(const Matrix3T& rhs);

	CPUGPU inline Matrix3T operator *(const Matrix3T& rhs) const;
	CPUGPU inline Matrix3T operator +(const Matrix3T& rhs) const;
	CPUGPU inline Matrix3T operator -(const Matrix3T& rhs) const;
	CPUGPU inline Matrix3T operator *(const real rhs) const;

	CPUGPU inline void bilinearForm(const Matrix3T& lhs, const Matrix3T& rhs, Matrix3T& result) const;
	CPUGPU inline real bilinearForm(const Vector3T<real>& lhs, const Vector3T<real>& rhs) const;

	CPUGPU inline void quadraticForm(const Matrix3T& parameter, Matrix3T& result) const;
	CPUGPU inline void quadraticForm(const Vector3T<real>& parameter, real& result) const;
	CPUGPU inline void transposed(Matrix3T& result) const;
	CPUGPU inline void inverse(Matrix3T & result)const;
	CPUGPU inline Matrix3T inverse()const;
	real contraction( Matrix3T& rhs ) const;
	CPUGPU inline void transpose();
	CPUGPU inline real determinant() const;
	CPUGPU inline real trace() const;
	CPUGPU inline real mises() const;
	CPUGPU inline real rowSumNorm() const;
	CPUGPU inline real colSumNorm() const;
	CPUGPU inline real frobeniusNorm() const;
	CPUGPU inline Matrix3T abs() const;

	CPUGPU inline Vector3T<real> getSkewPart() const;

	//real maxAbsElement() const;
	CPUGPU inline Vector3T<real>& operator[](int row);
	CPUGPU inline const Vector3T<real>& operator[](int row) const;

	Vector3T<real> getColumnVector(unsigned int index) const;
	const Vector3T<real>& getRowVector(unsigned int index) const;

	CPUGPU inline friend Matrix3T operator *(real a, Matrix3T const & b) { return b * a; }

	CPUGPU inline friend bool operator ==(const Matrix3T<real>& lhs, const Matrix3T<real>& rhs)
	{
		for (int row = 0; row < 3; ++row) {
			for (int col = 0; col < 3; ++col) {
				// without epsilon: desired!
				if (lhs[row][col] != rhs[row][col]) { return false; }
			}
		}
		return true;
	}

	CPUGPU inline friend bool operator !=(const Matrix3T<real>& lhs, const Matrix3T<real>& rhs)
	{
		return !(lhs == rhs);
	}

	static CPUGPU Matrix3T identity()
	{
		return Matrix3T{
			Vector3T<real>(1.0f, 0.0f, 0.0f),
			Vector3T<real>(0.0f, 1.0f, 0.0f),
			Vector3T<real>(0.0f, 0.0f, 1.0f)
		};
	}
	
	static Matrix3T zero, ones;
	static Matrix3T scale(real a,real b,real c);
	static Matrix3T rotationX(real angle);
	static Matrix3T rotationY(real angle);
	static Matrix3T rotationZ(real angle);
	static Matrix3T rotationAxisAngle(const Vector3T<real>& axisIn, real angle);
	static Matrix3T crossProductMatrix(const Vector3T<real>& vec);
	static Matrix3T rotationFromTo(const Vector3T<real>& from, const Vector3T<real>& to);
	CPUGPU inline Matrix3T transposeRet() const;
};
template<typename real>
std::ostream & operator <<(std::ostream & ioStream, const Matrix3T<real>& bm);
template<typename real>
void outerProduct(const Vector3T<real>& lhs, const Vector3T<real>& rhs, Matrix3T<real>& result);
template<typename real>
CPUGPU Matrix3T<real> outerProduct( const Vector3T<real>& lhs, const Vector3T<real>& rhs);

#include "Matrix3.inl"
typedef Matrix3T<float> Matrix3;

