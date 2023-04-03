// ========================================================================= //
//                                                                           //
// Filename: Matrix3.cpp                                                     
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

#include "Matrix3.h"
#include "Vector3.h"
//#include "DatatypesTraits.h"
#include "Real.h"
#include <cmath>
#include <stdio.h>

// bei manchen Compilern ist z.b. Vector3T<real>::e1 noch nicht initialisiert, da die Reihenfolge der statischen Variableninitialisierung nicht deterministisch ist
// -> identity ist komplett 0
//Matrix3T Matrix3T<real>::identity()(Vector3T<real>::e1,Vector3T<real>::e2,Vector3T<real>::e3);

template<typename real>
Matrix3T<real> Matrix3T<real>::zero(
	Vector3T<real>(0.0f, 0.0f, 0.0f),
	Vector3T<real>(0.0f, 0.0f, 0.0f),
	Vector3T<real>(0.0f, 0.0f, 0.0f) );

template<typename real>
Matrix3T<real> Matrix3T<real>::ones(
	Vector3T<real>(1.0f, 1.0f, 1.0f),
	Vector3T<real>(1.0f, 1.0f, 1.0f),
	Vector3T<real>(1.0f, 1.0f, 1.0f) );

template<typename real>
std::ostream & operator <<(std::ostream & os, const Matrix3T<real>& bm)
{
	char str[256];
	sprintf(str, "%12.6g %12.6g %12.6g\n"
		"%12.6g %12.6g %12.6g\n"
		"%12.6g %12.6g %12.6g",
		unsafe_Convert_ToFloat(bm[0][0]), unsafe_Convert_ToFloat(bm[0][1]), unsafe_Convert_ToFloat(bm[0][2]),
		unsafe_Convert_ToFloat(bm[1][0]), unsafe_Convert_ToFloat(bm[1][1]), unsafe_Convert_ToFloat(bm[1][2]),
		unsafe_Convert_ToFloat(bm[2][0]), unsafe_Convert_ToFloat(bm[2][1]), unsafe_Convert_ToFloat(bm[2][2]));
	os << str;
	return os;
}


template<typename real>
real Matrix3T<real>::contraction( Matrix3T& rhs ) const
{
	real res = 0.0f;
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			res += m_rows[i][j] * rhs.m_rows[i][j];
		}
	}
	return res;
}

//Matrix3T<real>::~Matrix3T()
//{
//}


template<typename real>
Matrix3T<real> Matrix3T<real>::rotationX(real angle)
{
	real c = cos(angle);
	real s = sin(angle);
	return Matrix3T(Vector3T<real>(1.0f,0.0f,0.0f), 
		Vector3T<real>(0.0f, c, -s), 
		Vector3T<real>(0.0f, s, c)
		);
}

template<typename real>
Matrix3T<real> Matrix3T<real>::rotationY(real angle)
{
	real c = cos(angle);
	real s = sin(angle);
	return Matrix3T(Vector3T<real>(c,0.0f,-s), 
		Vector3T<real>(0.0f, 1.0f, 0.0f), 
		Vector3T<real>(s, 0.0f, c)
		);

}

template<typename real>
Matrix3T<real> Matrix3T<real>::rotationZ(real angle)
{
	real c = cos(angle);
	real s = sin(angle);
	return Matrix3T(Vector3T<real>(c,-s,0.0f), 
		Vector3T<real>(s, c, 0.0f), 
		Vector3T<real>(0.0f, 0.0f, 1.0f)
		);

}

template<typename real>
Matrix3T<real> Matrix3T<real>::rotationAxisAngle(const Vector3T<real>& axisIn, real angle)
{
	//public void matrixFromAxisAngle(AxisAngle4d a1) {

	//	double c = Math.cos(a1.angle);
	//	double s = Math.sin(a1.angle);
	//	double t = 1.0 - c;
	//	//  if axis is not already normalised then uncomment this
	//	// double magnitude = Math.sqrt(a1.x*a1.x + a1.y*a1.y + a1.z*a1.z);
	//	// if (magnitude==0) throw error;
	//	// a1.x /= magnitude;
	//	// a1.y /= magnitude;
	//	// a1.z /= magnitude;

	//	m00 = c + a1.x*a1.x*t;
	//	m11 = c + a1.y*a1.y*t;
	//	m22 = c + a1.z*a1.z*t;


	//	double tmp1 = a1.x*a1.y*t;
	//	double tmp2 = a1.z*s;
	//	m10 = tmp1 + tmp2;
	//	m01 = tmp1 - tmp2;
	//	tmp1 = a1.x*a1.z*t;
	//	tmp2 = a1.y*s;
	//	m20 = tmp1 - tmp2;
	//	m02 = tmp1 + tmp2;    tmp1 = a1.y*a1.z*t;
	//	tmp2 = a1.x*s;
	//	m21 = tmp1 + tmp2;
	//	m12 = tmp1 - tmp2;
	//}

	real c = cos(angle);
	real s = sin(angle);
	real t = 1.0f - c;
	Vector3T<real> axis(axisIn);
	axis.normalize();


	real tmp1 = axis[0] * axis[1] *t;
	real tmp2 = axis[2] *s;
	
	real tmp3 = axis[0]*axis[2]*t;
	real tmp4 = axis[1]*s;


	real tmp5 = axis[1]*axis[2]*t;
	real tmp6 = axis[0]*s;

	return Matrix3T(
		Vector3T<real>(c + axis[0] * axis[0] * t, tmp1 - tmp2, tmp3 - tmp4), 
		Vector3T<real>(tmp1 + tmp2, c + axis[1] * axis[1] * t, tmp5 - tmp6), 
		Vector3T<real>(tmp3 + tmp4, tmp5 + tmp6, c + axis[2] * axis[2] * t)
		);

}


template<typename real>
Matrix3T<real> Matrix3T<real>::rotationFromTo( const Vector3T<real>& from, const Vector3T<real>& to )
{
	// vec a, vec b
	// vec v = a cross b
	// s = || v || (sinus winkel)
	// c = a dot b (cos winkel)
	// R = I +

	Vector3T<real> fromN = from.normalized();
	Vector3T<real> toN = to.normalized();

	Vector3T<real> v = fromN.cross(toN);
	real s = v.length();
	real c = fromN * toN; 

	if(fabs(s) < 1.e-10f)
		return Matrix3T<real>::identity();
	else
	{
		Matrix3T V = crossProductMatrix(v);
		Matrix3T R = Matrix3T<real>::identity() + V + (V * V) * ((1.0f - c) / (s*s));
		return R;
	}
}

template<typename real>
Matrix3T<real> Matrix3T<real>::crossProductMatrix(const Vector3T<real>& vec)
{
	return Matrix3T(
		Vector3T<real>(0.0f, -vec[2], vec[1]), 
		Vector3T<real>(vec[2], 0.0f, -vec[0]), 
		Vector3T<real>(-vec[1], vec[0], 0.0f)
		);
}



template<typename real>
Matrix3T<real> Matrix3T<real>::scale(real a,real b,real c)
{
	return Matrix3T(Vector3T<real>(a,0.0f,0.0f),
				   Vector3T<real>(0.0f,b,0.0f),
				   Vector3T<real>(0.0f,0.0f,c));
}

template<typename real>
Vector3T<real> Matrix3T<real>::getColumnVector(unsigned int col) const
{
	return Vector3T<real>(m_rows[0][col], m_rows[1][col], m_rows[2][col]);
}

template<typename real>
const Vector3T<real>& Matrix3T<real>::getRowVector( unsigned int index ) const
{
	return m_rows[index];
}



//real Matrix3T<real>::maxAbsElement()
//{
//	Vector3T<real> temp(m_rows[0].maxAbsElement(), m_rows[1].maxAbsElement(), m_rows[2].maxAbsElement() );
//	return temp.maxAbsElement();
//}

#include "Real.h"

TEMPLATE_INSTANTIATE_FOR_SCALARS(class Matrix3T)
#define LOCAL_TEMPLATE_INSTANTIATION_CLASS_FOR_SCALARS(REALTYPE) \
template std::ostream & operator <<(std::ostream & os, const Matrix3T<REALTYPE>& bm);
EXECUTE_MACRO_FOR_ALL_SCALARS(LOCAL_TEMPLATE_INSTANTIATION_CLASS_FOR_SCALARS)
#undef LOCAL_TEMPLATE_INSTANTIATION_CLASS_FOR_SCALARS
