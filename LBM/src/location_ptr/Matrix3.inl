// ========================================================================= //
//                                                                           //
// Filename: Matrix3T.inl                                                     
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

#include "DatatypesTraits.h"

template<typename real>
Matrix3T<real>::Matrix3T(const Vector3T<real>& row1, const Vector3T<real>& row2, const Vector3T<real>& row3)
{
	m_rows[0] = row1;
	m_rows[1] = row2;
	m_rows[2] = row3;
}

template<typename real>
Matrix3T<real> Matrix3T<real>::transposeRet() const
{
	Matrix3T result;
	transposed(result);
	return result;
}

template<typename real> template<typename real2>
Vector3T<real2> Matrix3T<real>::operator *(const Vector3T<real2>& rhs) const
{	
	Vector3T<real2> ret;
	for(int i = 0;i < 3;++i)
		for(int j = 0;j < 3;++j)
			ret[i] += m_rows[i][j] * rhs[j];

	return ret;
}

template<typename real>
Vector3T<real> Matrix3T<real>::leftMult( const Vector3T<real>& rhs ) const
{
	Vector3T<real> res;
	for(unsigned int j = 0; j < 3; ++j)
	{
		for(unsigned int i = 0; i < 3; ++i)
		{
			res[j] += m_rows[i][j] * rhs[i];
		}
	}
	return res;
}

template<typename real>
Vector3T<real> Matrix3T<real>::transposedMult(const Vector3T<real>& rhs) const
{
	return Vector3T<real>(m_rows[0][0] * rhs[0] + m_rows[1][0] * rhs[1] + m_rows[2][0] * rhs[2],
				   m_rows[0][1] * rhs[0] + m_rows[1][1] * rhs[1] + m_rows[2][1] * rhs[2],
				   m_rows[0][2] * rhs[0] + m_rows[1][2] * rhs[1] + m_rows[2][2] * rhs[2]);
}

template<typename real>
Matrix3T<real> Matrix3T<real>::diagMultRight(const Vector3T<real>& diag) const
{
	Matrix3T res;
	for(unsigned int i = 0; i < 3; ++i)
	{
		for(unsigned int j = 0; j < 3; ++j)
		{
			res[i][j] = m_rows[i][j] * diag[j];
		}
	}
	return res;
}

template<typename real>
Vector3T<real>& Matrix3T<real>::operator[](int index)
{
	return m_rows[index];
}

template<typename real>
const Vector3T<real>& Matrix3T<real>::operator[](int index) const
{
	return m_rows[index];
}

template<typename real>
Matrix3T<real>& Matrix3T<real>::operator *=(const real rhs)
{
	m_rows[0] *= rhs;
	m_rows[1] *= rhs;
	m_rows[2] *= rhs;
	return *this;
}


template<typename real>
Matrix3T<real>& Matrix3T<real>::operator +=(const Matrix3T& rhs)
{
	m_rows[0] += rhs.m_rows[0];
	m_rows[1] += rhs.m_rows[1];
	m_rows[2] += rhs.m_rows[2];
	return *this;
}

template<typename real>
void Matrix3T<real>::transposedAdd(const Matrix3T& rhs )
{
	m_rows[0][0] += rhs.m_rows[0][0];
	m_rows[0][1] += rhs.m_rows[1][0];
	m_rows[0][2] += rhs.m_rows[2][0];

	m_rows[1][0] += rhs.m_rows[0][1];
	m_rows[1][1] += rhs.m_rows[1][1];
	m_rows[1][2] += rhs.m_rows[2][1];

	m_rows[2][0] += rhs.m_rows[0][2];
	m_rows[2][1] += rhs.m_rows[1][2];
	m_rows[2][2] += rhs.m_rows[2][2];

}

template<typename real>
Matrix3T<real> Matrix3T<real>::transposedMult() const
{
	Matrix3T mat_t;
	this->transposed(mat_t);
	Matrix3T res = mat_t * (*this);
	return res;
}

template<typename real>
Matrix3T<real> Matrix3T<real>::abs() const
{
	Matrix3T result;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			result.m_rows[i][j] = Trait<real>::abs(m_rows[i][j]);
	return result;
}

template<typename real>
inline Vector3T<real> Matrix3T<real>::getSkewPart() const
{
	return Vector3T<real>(0.5f * (m_rows[2][1] - m_rows[1][2]), 0.5f * (m_rows[0][2] - m_rows[2][0]), 0.5f * (m_rows[1][0] - m_rows[0][1]));
}

template<typename real>
Matrix3T<real>& Matrix3T<real>::operator -=(const Matrix3T& rhs)
{
	m_rows[0] -= rhs.m_rows[0];
	m_rows[1] -= rhs.m_rows[1];
	m_rows[2] -= rhs.m_rows[2];
	return *this;
}


template<typename real>
Matrix3T<real>& Matrix3T<real>::operator *=(const Matrix3T& rhs)
{
	Matrix3T thisCopy(*this);
	/** row 0 */
	m_rows[0][0] =   thisCopy[0][0] * rhs[0][0]
				   + thisCopy[0][1] * rhs[1][0]
				   + thisCopy[0][2] * rhs[2][0];

	m_rows[0][1] =   thisCopy[0][0] * rhs[0][1]
				   + thisCopy[0][1] * rhs[1][1]
				   + thisCopy[0][2] * rhs[2][1];

	m_rows[0][2] =   thisCopy[0][0] * rhs[0][2]
				   + thisCopy[0][1] * rhs[1][2]
				   + thisCopy[0][2] * rhs[2][2];

	/** row 1 */
	m_rows[1][0] =   thisCopy[1][0] * rhs[0][0]
				   + thisCopy[1][1] * rhs[1][0]
				   + thisCopy[1][2] * rhs[2][0];

	m_rows[1][1] =   thisCopy[1][0] * rhs[0][1]
				   + thisCopy[1][1] * rhs[1][1]
				   + thisCopy[1][2] * rhs[2][1];

	m_rows[1][2] =   thisCopy[1][0] * rhs[0][2]
				   + thisCopy[1][1] * rhs[1][2]
				   + thisCopy[1][2] * rhs[2][2];

	/** row 2 */
	m_rows[2][0] =   thisCopy[2][0] * rhs[0][0]
				   + thisCopy[2][1] * rhs[1][0]
				   + thisCopy[2][2] * rhs[2][0];

	m_rows[2][1] =   thisCopy[2][0] * rhs[0][1]
				   + thisCopy[2][1] * rhs[1][1]
				   + thisCopy[2][2] * rhs[2][1];

	m_rows[2][2] =   thisCopy[2][0] * rhs[0][2]
				   + thisCopy[2][1] * rhs[1][2]
				   + thisCopy[2][2] * rhs[2][2];

	return (*this);
}


template<typename real>
Matrix3T<real>& Matrix3T<real>::operator /=(const real rhs)
{
	m_rows[0] /= rhs;
	m_rows[1] /= rhs;
	m_rows[2] /= rhs;
	return (*this);
}


template<typename real>
Matrix3T<real> Matrix3T<real>::operator *(const Matrix3T& rhs) const
{
	return Matrix3T(
	/** row 0 */
	Vector3T<real>(         m_rows[0][0] * rhs[0][0]
				   + m_rows[0][1] * rhs[1][0]
				   + m_rows[0][2] * rhs[2][0],

					 m_rows[0][0] * rhs[0][1]
				   + m_rows[0][1] * rhs[1][1]
				   + m_rows[0][2] * rhs[2][1],

					 m_rows[0][0] * rhs[0][2]
				   + m_rows[0][1] * rhs[1][2]
				   + m_rows[0][2] * rhs[2][2]),
	/** row 1 */
	Vector3T<real>(         m_rows[1][0] * rhs[0][0]
				   + m_rows[1][1] * rhs[1][0]
				   + m_rows[1][2] * rhs[2][0],

					 m_rows[1][0] * rhs[0][1]
				   + m_rows[1][1] * rhs[1][1]
				   + m_rows[1][2] * rhs[2][1],

					 m_rows[1][0] * rhs[0][2]
				   + m_rows[1][1] * rhs[1][2]
				   + m_rows[1][2] * rhs[2][2]),
	/** row 2 */
	Vector3T<real>(         m_rows[2][0] * rhs[0][0]
				   + m_rows[2][1] * rhs[1][0]
				   + m_rows[2][2] * rhs[2][0],

					 m_rows[2][0] * rhs[0][1]
				   + m_rows[2][1] * rhs[1][1]
				   + m_rows[2][2] * rhs[2][1],

					 m_rows[2][0] * rhs[0][2]
				   + m_rows[2][1] * rhs[1][2]
				   + m_rows[2][2] * rhs[2][2]));
}


template<typename real>
Matrix3T<real> Matrix3T<real>::operator +(const Matrix3T& rhs) const
{
	return Matrix3T(Vector3T<real>(m_rows[0][0] + rhs[0][0],
						   m_rows[0][1] + rhs[0][1],
						   m_rows[0][2] + rhs[0][2]),
				   Vector3T<real>(m_rows[1][0] + rhs[1][0],
						   m_rows[1][1] + rhs[1][1],
						   m_rows[1][2] + rhs[1][2]),
				   Vector3T<real>(m_rows[2][0] + rhs[2][0],
						   m_rows[2][1] + rhs[2][1],
						   m_rows[2][2] + rhs[2][2]));
}

template<typename real>
Matrix3T<real> Matrix3T<real>::operator -(const Matrix3T& rhs) const
{
	return Matrix3T(Vector3T<real>(m_rows[0][0] - rhs[0][0],
						   m_rows[0][1] - rhs[0][1],
						   m_rows[0][2] - rhs[0][2]),
				   Vector3T<real>(m_rows[1][0] - rhs[1][0],
						   m_rows[1][1] - rhs[1][1],
						   m_rows[1][2] - rhs[1][2]),
				   Vector3T<real>(m_rows[2][0] - rhs[2][0],
						   m_rows[2][1] - rhs[2][1],
						   m_rows[2][2] - rhs[2][2]));
}

template<typename real>
Matrix3T<real> Matrix3T<real>::operator *(const real rhs) const
{
	return Matrix3T(m_rows[0] * rhs, m_rows[1] * rhs, m_rows[2] * rhs);
}

template<typename real>
real Matrix3T<real>::bilinearForm(const Vector3T<real>& lhs, const Vector3T<real>& rhs) const
{
	real result = 0.0f;
	for(int k = 0; k < 3; ++k)
	{
		for(int l = 0; l < 3; ++l)
		{
			result += lhs[k] * (*this)[k][l] * rhs[l];
		}
	}
	return result;
}

template<typename real>
void Matrix3T<real>::bilinearForm(const Matrix3T& lhs, const Matrix3T& rhs, Matrix3T& result) const
{
	result = lhs;
	result *= *this;
	result *= rhs;
}

template<typename real>
void Matrix3T<real>::quadraticForm(const Matrix3T& parameter, Matrix3T& result) const
{
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			result[i][j] = 0.0f; 
			for(int k = 0; k < 3; ++k)
			{
				for(int l = 0; l < 3; ++l)
				{
					result[i][j] += parameter[i][k] * (*this)[k][l] * parameter[j][l];
				}
			}
		}
	}
	//Matrix3T transposed;
	//parameter.transposed(transposed);
	//result = parameter;
	//result *= (*this);
	//result *= transposed;
}

template<typename real>
void Matrix3T<real>::quadraticForm(const Vector3T<real>& parameter, real& result) const 
{
	Vector3T<real> res = this->operator *(parameter);
	result = res * parameter;
}

template<typename real>
void Matrix3T<real>::transposed(Matrix3T& result) const 
{
	result[0][0] = m_rows[0][0];
	result[0][1] = m_rows[1][0];
	result[0][2] = m_rows[2][0];
	result[1][0] = m_rows[0][1];
	result[1][1] = m_rows[1][1];
	result[1][2] = m_rows[2][1];
	result[2][0] = m_rows[0][2];
	result[2][1] = m_rows[1][2];
	result[2][2] = m_rows[2][2];
}

template<typename real>
void Matrix3T<real>::inverse(Matrix3T& result) const
{
	real d = determinant();
	if (d != 0) 
	{
		result[0][0] = m_rows[1][1] * m_rows[2][2] - m_rows[1][2] * m_rows[2][1]; 
		result[0][1] = m_rows[0][2] * m_rows[2][1] - m_rows[0][1] * m_rows[2][2];
		result[0][2] = m_rows[0][1] * m_rows[1][2] - m_rows[0][2] * m_rows[1][1];

		result[1][0] = m_rows[1][2] * m_rows[2][0] - m_rows[1][0] * m_rows[2][2];
		result[1][1] = m_rows[0][0] * m_rows[2][2] - m_rows[0][2] * m_rows[2][0];
		result[1][2] = m_rows[0][2] * m_rows[1][0] - m_rows[0][0] * m_rows[1][2];

		result[2][0] = m_rows[1][0] * m_rows[2][1] - m_rows[1][1] * m_rows[2][0];
		result[2][1] = m_rows[0][1] * m_rows[2][0] - m_rows[0][0] * m_rows[2][1];
		result[2][2] = m_rows[0][0] * m_rows[1][1] - m_rows[0][1] * m_rows[1][0];

		result /= d;
	}
}

template<typename real>
inline Matrix3T<real> Matrix3T<real>::inverse()const
{
	Matrix3T result;
	inverse(result);
	return result;
}

template<typename real>
void Matrix3T<real>::transpose()
{
	for(int i = 0; i < 3; ++i) {
		for(int j = i+1; j < 3; ++j) {
			real temp = m_rows[i][j];
			m_rows[i][j] = m_rows[j][i];
			m_rows[j][i] = temp;
		}
	}
}



template<typename real>
real Matrix3T<real>::determinant() const 
{
	real result =   m_rows[0][0] * m_rows[1][1] * m_rows[2][2]
				   + m_rows[0][1] * m_rows[1][2] * m_rows[2][0]
				   + m_rows[0][2] * m_rows[1][0] * m_rows[2][1];
	result -=   m_rows[0][0] * m_rows[1][2] * m_rows[2][1]
			  + m_rows[0][2] * m_rows[1][1] * m_rows[2][0]
			  + m_rows[0][1] * m_rows[1][0] * m_rows[2][2];
	return result;
}

template<typename real>
real Matrix3T<real>::trace() const
{
	return m_rows[0][0] + m_rows[1][1] + m_rows[2][2];
}

template<typename real>
real Matrix3T<real>::mises() const
{
	real res = m_rows[0][0] * m_rows[0][0] + m_rows[1][1] * m_rows[1][1] + m_rows[2][2] * m_rows[2][2] 
			- m_rows[0][0] * m_rows[1][1] - m_rows[0][0] * m_rows[2][2] - m_rows[1][1] * m_rows[2][2] 
			+ 3*(m_rows[1][0] * m_rows[1][0] + m_rows[2][0] * m_rows[2][0] + m_rows[1][2] * m_rows[1][2]);
	return sqrt(res);
}

template<typename real>
real Matrix3T<real>::rowSumNorm() const
{
	real val1 = fabs(m_rows[0][0]) + fabs(m_rows[0][1]) + fabs(m_rows[0][2]);
	real val2 = fabs(m_rows[1][0]) + fabs(m_rows[1][1]) + fabs(m_rows[1][2]);
	real val3 = fabs(m_rows[2][0]) + fabs(m_rows[2][1]) + fabs(m_rows[2][2]);
	if (val3 > val2) { val2 = val3; }
	if (val2 > val1) { val1 = val2; }
	return val1;
}

template<typename real>
real Matrix3T<real>::colSumNorm() const
{
	real val1 = fabs(m_rows[0][0]) + fabs(m_rows[1][0]) + fabs(m_rows[2][0]);
	real val2 = fabs(m_rows[0][1]) + fabs(m_rows[1][1]) + fabs(m_rows[2][1]);
	real val3 = fabs(m_rows[0][2]) + fabs(m_rows[1][2]) + fabs(m_rows[2][2]);
	if (val3 > val2) { val2 = val3; }
	if (val2 > val1) { val1 = val2; }
	return val1;
}

template<typename real>
real Matrix3T<real>::frobeniusNorm() const 
{
	real res = m_rows[0][0] * m_rows[0][0] + m_rows[0][1] * m_rows[0][1] + m_rows[0][2] * m_rows[0][2]
		+ m_rows[1][0] * m_rows[1][0] + m_rows[1][1] * m_rows[1][1] + m_rows[1][2] * m_rows[1][2]
		+ m_rows[2][0] * m_rows[2][0] + m_rows[2][1] * m_rows[2][1] + m_rows[2][2] * m_rows[2][2];
	return sqrt(res);
}


template<typename real>
void outerProduct(const Vector3T<real>& lhs, const Vector3T<real>& rhs, Matrix3T<real>& result)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			result[i][j] = lhs[i] * rhs[j];
		}
	}
}


template<typename real>
Matrix3T<real> outerProduct(const Vector3T<real>& lhs, const Vector3T<real>& rhs)
{
	Matrix3T<real> result;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			result[i][j] = lhs[i] * rhs[j];
		}
	}
	return result;
}