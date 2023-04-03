#pragma once

#include "array.h"
#include "DatatypesTraits.h"

#include "cpugpu_macros_push.inl"

#ifdef WITH_CPPAD
#include <cppad/cppad.hpp>
#endif

template<typename Real, typename Real2, int N>
CPUGPU FORCEINLINE auto dot(array<Real, N> a, array<Real2, N> b) -> typename std::common_type<Real, Real2>::type
{
	using RealC = typename std::common_type<Real, Real2>::type;
	return sum(array<RealC, N>(a) * array<RealC, N>(b));
}

template<typename Real>
CPUGPU FORCEINLINE array<Real, 3> cross(array<Real, 3> a, array<Real, 3> b)
{
	array<Real, 3> r;
	r[0] = a[1] * b[2] - a[2] * b[1];
	r[1] = a[2] * b[0] - a[0] * b[2];
	r[2] = a[0] * b[1] - a[1] * b[0];
	return r;
}

template<typename Real, int N, int M>
struct matrix
	: array<array<Real, M>, N>
{
	matrix() = default;
	explicit CPUGPU FORCEINLINE matrix(array<array<Real, M>, N> o) : array<array<Real, M>, N>(std::move(o)) {}
	explicit CPUGPU FORCEINLINE matrix(Real c) : array<array<Real, N>, N>(array<Real, N>(std::move(c))) {}
};

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> & operator*=(matrix<Real, N, M> & a, Real b)
{
	a *= array<Real, M>(b);
	return a;
}

template<typename Real, typename EleReal, int N, int M>
CPUGPU FORCEINLINE auto operator*=(matrix<Real, N, M> & a, EleReal b)
	-> meta::enable_if<std::is_same<typename ScalarTrait<Real>::BuiltInType, EleReal>::value && !std::is_same<Real, EleReal>::value, matrix<Real, N, M> &>
{
	a *= array<Real, M>(b);
	return a;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> operator*(matrix<Real, N, M> a, Real b)
{
	a *= b;
	return a;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> operator*(Real a, matrix<Real, N, M> b)
{
	b *= a;
	return b;
}

template<typename Real, typename EleReal, int N, int M>
CPUGPU FORCEINLINE auto operator*(matrix<Real, N, M> a, EleReal b)
	-> meta::enable_if<std::is_same<typename ScalarTrait<Real>::BuiltInType, EleReal>::value && !std::is_same<Real, EleReal>::value, matrix<Real, N, M>>
{
	a *= b;
	return a;
}

template<typename Real, typename EleReal, int N, int M>
CPUGPU FORCEINLINE auto operator*(EleReal a, matrix<Real, N, M> b)
	-> meta::enable_if<std::is_same<typename ScalarTrait<Real>::BuiltInType, EleReal>::value && !std::is_same<Real, EleReal>::value, matrix<Real, N, M>>
{
	b *= a;
	return b;
}

template<typename Real, int N, int M, int P>
CPUGPU FORCEINLINE matrix<Real, N, P> operator*(matrix<Real, N, M> a, matrix<Real, M, P> b)
{
	matrix<Real, N, P> result;
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for(int n = 0; n < N; ++n)
	{
		auto const arow = a[n];
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int p = 0; p < P; ++p)
		{
			array<Real, M> bcol;
			#ifdef __CUDA_ARCH__
			#pragma unroll
			#endif
			for(int m = 0; m < M; ++m)
				bcol[m] = b[m][p];
			result[n][p] = dot(arow, bcol);
		}
	}
	return result;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> & operator+=(matrix<Real, N, M> & a, matrix<Real, N, M> b)
{
	a += static_cast<array<array<Real, M>, N> const &>(b);
	return a;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> operator+(matrix<Real, N, M> a, matrix<Real, N, M> b)
{
	a += b;
	return a;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> & operator-=(matrix<Real, N, M> & a, matrix<Real, N, M> b)
{
	a -= static_cast<array<array<Real, M>, N> const &>(b);
	return a;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> operator-(matrix<Real, N, M> a, matrix<Real, N, M> b)
{
	a -= b;
	return a;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE array<Real, N> operator*(matrix<Real, N, M> a, array<Real, M> b)
{
	array<Real, N> r;
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for(int k = 0; k < N; ++k)
		r[k] = dot(a[k], b);
	return r;
}

template<typename Real, int N>
CPUGPU FORCEINLINE array<Real, N> multiplyWithoutDiagonal(matrix<Real, N, N> a, array<Real, N> b)
{
	array<Real, N> r;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
	for (int k = 0; k < N; ++k)
	{
		auto row = a[k];
		row[k] = 0;
		r[k] = dot(row, b);
	}
	return r;
}

template<typename Real, int N>
CPUGPU FORCEINLINE array<Real, N> multiplyWithDiagonalInverses(matrix<Real, N, N> a, array<Real, N> b)
{
	array<Real, N> r;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
	for (int k = 0; k < N; ++k)
	{
		r[k] = b[k] / a[k][k];
	}
	return r;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, N, M> zeroMatrix()
{
	return matrix<Real, N, M>(array<array<Real, M>, N>(array<Real, M>(static_cast<Real>(-0.0))));
}

template<typename Real, int N>
CPUGPU FORCEINLINE matrix<Real, N, N> identityMatrix()
{
	matrix<Real, N, N> z = zeroMatrix<Real, N, N>();
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for (int k = 0; k < N; ++k)
	{
		z[k][k] = static_cast<Real>(1.0);
	}
	return z;
}

template<typename Real, int N>
CPUGPU FORCEINLINE matrix<Real, N, N> outer(array<Real, N> a, array<Real, N> b)
{
	matrix<Real, N, N> r;
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for(int k = 0; k < N; ++k)
		r[k] = a[k] * b;
	return r;
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE matrix<Real, M, N> transpose(matrix<Real, N, M> A)
{
	matrix<Real, M, N> T;

#ifdef __CUDA_ARCH__
#pragma unroll
#endif
	for(int n = 0; n < N; ++n)
	{
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for(int m = 0; m < M; ++m)
		{
			T[m][n] = A[n][m];
		}
	}

	return T;
}

template<typename Real>
CPUGPU FORCEINLINE Real det(matrix<Real, 1, 1> A)
{
	return A[0][0];
}

template<typename Real>
CPUGPU FORCEINLINE Real det(matrix<Real, 2, 2> A)
{
	return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

template<typename Real>
CPUGPU FORCEINLINE Real det(matrix<Real, 3, 3> A)
{
	return
		+ A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2])
		- A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
		+ A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

template<typename Real, int N>
CPUGPU FORCEINLINE Real det(matrix<Real, N, N> A)
{
	Real res = 0.0;
#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
	for (int j = 0; j < N; ++j)
	{
		matrix<Real, N - 1, N - 1> subMatrix;
		const int i = 0; // Entwicklung nach der ersten Zeile
#ifdef __CUDA_ARCH__
#pragma	unroll
#endif
		for (int k = 0; k < N - 1; ++k)
		{
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
			for (int l = 0; l < N - 1; ++l)
			{
				subMatrix[k][l] = A[k + (k >= i)][l + (l >= j)];
			}
		}
		auto sgn = (i + j) % 2 ? -1 : 1;
		res += sgn * A[i][j] * det(subMatrix);
	}
	return res;
}

template<typename Real>
CPUGPU FORCEINLINE matrix<Real, 1, 1> inverse(matrix<Real, 1, 1> A)
{
	matrix<Real, 1, 1> res;
	res[0][0] = 1 / A[0][0];
	return res;
}

template<typename Real>
CPUGPU FORCEINLINE matrix<Real, 2, 2> inverse(matrix<Real, 2, 2> A)
{
	auto invdet = 1 / det(A);
	matrix<Real, 2, 2> res;
	res[0][0] =  A[1][1] * invdet;
	res[0][1] = -A[0][1] * invdet;
	res[1][0] = -A[1][0] * invdet;
	res[1][1] =  A[0][0] * invdet;
	return res;
}

template<typename Real>
CPUGPU FORCEINLINE matrix<Real, 3, 3> inverse(matrix<Real, 3, 3> A)
{
	auto invdet = 1 / det(A);
	matrix<Real, 3, 3> res;
	res[0][0] =  (A[1][1] * A[2][2] - A[2][1] * A[1][2]) * invdet;
	res[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * invdet;
	res[0][2] =  (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invdet;
	res[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * invdet;
	res[1][1] =  (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invdet;
	res[1][2] = -(A[0][0] * A[1][2] - A[1][0] * A[0][2]) * invdet;
	res[2][0] =  (A[1][0] * A[2][1] - A[2][0] * A[1][1]) * invdet;
	res[2][1] = -(A[0][0] * A[2][1] - A[2][0] * A[0][1]) * invdet;
	res[2][2] =  (A[0][0] * A[1][1] - A[1][0] * A[0][1]) * invdet;
	return res;
}

template<typename Real, int N>
CPUGPU FORCEINLINE matrix<Real, N, N> inverse(matrix<Real, N, N> A)
{
	auto invdet = 1 / det(A);
	matrix<Real, N, N> res;

#ifdef __CUDA_ARCH__
	#pragma	unroll
#endif
	for (int i = 0; i < N; ++i)
	{
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (int j = 0; j < N; ++j)
		{
			matrix<Real, N - 1, N - 1> submatrix;
#ifdef __CUDA_ARCH__
			#pragma	unroll
#endif
			for (int k = 0; k < N - 1; ++k)
			{
#ifdef __CUDA_ARCH__
				#pragma unroll
#endif
				for (int l = 0; l < N - 1; ++l)
				{
					submatrix[k][l] = A[k + (k >= j)][l + (l >= i)];
				}
			}
			auto sgn = (i + j) % 2 ? -1 : 1;
			res[i][j] = sgn * det(submatrix) * invdet;
		}
	}

	return res;
}

template<typename Real>
CPUGPU FORCEINLINE Real trace(matrix<Real, 0, 0>) { return Real(0); }

template<typename Real, int N>
CPUGPU FORCEINLINE Real trace(matrix<Real, N, N> M)
{
	auto r = M[0][0];
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for(int i = 1; i < N; ++i)
		r += M[i][i];
	return r;
}

// TODO DASPITZE unfortunately, this overload is not local at all - as we're inside a header, the anonymous namespace is not file local...
// As the sqrt function resides in different namespaces for different types, we define a overloaded wrapper function and call this wrapper instead
namespace {
	#ifdef WITH_CPPAD
	FORCEINLINE CppAD::AD<double> localOverloadSqrt(const CppAD::AD<double> & x)
	{
		return CppAD::sqrt(x);
	}
	#endif

	template<typename Real>
	CPUGPU FORCEINLINE Real localOverloadSqrt(const Real & x)
	{
		return std::sqrt(x);
	}
}

template<typename Real, int N>
CPUGPU FORCEINLINE Real euclideanNorm(array<Real, N> a)
{
#ifdef __CUDA_ARCH__
	return sqrt(sum(a * a));
#else
	return localOverloadSqrt(sum(a * a));
#endif
}

template<typename Real, int N>
CPUGPU FORCEINLINE Real euclideanDistance(array<Real, N> a, array<Real, N> b)
{
	auto ab = a - b;

	return euclideanNorm(ab);
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE Real frobeniusNorm(matrix<Real, N, M> const & mat)
{
#ifndef __CUDA_ARCH__
	using std::sqrt;
#endif
	return sqrt(sum(sum(static_cast<array<array<Real, M>, N> const &>(mat) * static_cast<array<array<Real, M>, N> const &>(mat))));
}

template<typename Real>
CPUGPU FORCEINLINE Real orientedParallelotopeVolume(array<Real, 3> const (&p)[4])
{
	return dot(p[1] - p[0], cross(p[2] - p[0], p[3] - p[0]));
}

template<typename Real>
CPUGPU FORCEINLINE Real orientedSimplexVolume(array<Real, 3> const (&p)[4])
{
	return (1.f / 6.f) * orientedParallelotopeVolume(p);
}

template<typename Real>
CPUGPU FORCEINLINE void barycentricGradients(array<Real, 3> (&g)[4], array<Real, 3> const (&p)[4])
{
	auto iV6 = 1.f / orientedParallelotopeVolume(p);

#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
	for(int i = 0; i < 4; ++i)
		g[i] = iV6 * cross(p[(4 - i - 1) % 4] - p[(4 - i + 1) % 4], p[(i + 2) % 4] - p[(i + 1) % 4]);
}

template<typename Real>
CPUGPU FORCEINLINE Real scalarTripleProduct(array<Real, 3> a, array<Real, 3> b, array<Real, 3> c)
{
	return dot(c, cross(a, b));
}

template<typename Real, int N, int M>
CPUGPU FORCEINLINE Real doubleContraction(matrix<Real, N, M> a, matrix<Real, N, M> b)
{
	array<Real, N> temp;
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for (int n = 0; n < N; ++n)
	{
		temp[n] = dot(a[n], b[n]);
	}
	return sum(temp);
}

template<typename Real, int N>
CPUGPU FORCEINLINE matrix<Real, N, N> inverseLUP(matrix<Real, N, N> A)
{
	auto P = identityMatrix<Real, N>();
	//decomposition PA = LU
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for (int i = 0; i < N; ++i) {
		int ind = i;
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for (int j = i; j < N; ++j) {
			ind = abs(A[j][i]) > abs(A[ind][i]) ? j : ind;
		}

		auto tmpP = P[i];
		P[i] = P[ind];
		P[ind] = tmpP;

		auto tmpA = A[i];
		A[i] = A[ind];
		A[ind] = tmpA;

		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for (int j = i + 1; j < N; ++j) {
			A[j][i] /= A[i][i];
			#ifdef __CUDA_ARCH__
			#pragma unroll
			#endif
			for (int k = i + 1; k < N; ++k) {
				A[j][k] -= A[j][i] * A[i][k];
			}
		}
	}
	//solve  Ly = Pb, Ux = y
	#ifdef __CUDA_ARCH__
	#pragma unroll
	#endif
	for (int i = 0; i < N; i++) {
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for (int j = 0; j < N; j++) {
			#ifdef __CUDA_ARCH__
			#pragma unroll
			#endif
			for (int k = 0; k < j; k++) {
				P[j][i] -= A[j][k] * P[k][i];
			}
		}
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for (int j = N - 1; j >= 0; --j) {
			#ifdef __CUDA_ARCH__
			#pragma unroll
			#endif
			for (int k = j + 1; k < N; ++k) {
				P[j][i] -= A[j][k] * P[k][i];
			}
			P[j][i] /= A[j][j];
		}
	}
	return P;
}

#include "cpugpu_macros_pop.inl"
