// ========================================================================= //
//                                                                           //
// Filename: array.h
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut für Graphische Datenverarbeitung (IGD)        //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2018 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 10.2013 jsroemer
//                                                                           //
// ========================================================================= //
#define WITH_CUDA
#pragma once

#include "arrayFwd.h"
#include "meta.h"

#include <ostream>
#include <type_traits>

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include "cpugpu_macros_push.inl"

#pragma push_macro("DEFINE_BINARY_ARRAY_OPERATOR")
#pragma push_macro("DEFINE_COMPARISON_ARRAY_OPERATOR")
#pragma push_macro("DEFINE_UNARY_ARRAY_OPERATOR")
#pragma push_macro("DEFINE_INCDEC_ARRAY_OPERATOR")

#ifdef DEFINE_BINARY_ARRAY_OPERATOR
#undef DEFINE_BINARY_ARRAY_OPERATOR
#endif
#ifdef DEFINE_COMPARISON_ARRAY_OPERATOR
#undef DEFINE_COMPARISON_ARRAY_OPERATOR
#endif
#ifdef DEFINE_UNARY_ARRAY_OPERATOR
#undef DEFINE_UNARY_ARRAY_OPERATOR
#endif
#ifdef DEFINE_INCDEC_ARRAY_OPERATOR
#undef DEFINE_INCDEC_ARRAY_OPERATOR
#endif

#define DEFINE_BINARY_ARRAY_OPERATOR(op) \
	CPUGPU array & operator op ## = (array const & o) \
	{ \
		for(int i = 0; i < N; ++i) \
			(*this)[i] op ## = o[i]; \
		return *this; \
	} \
	friend CPUGPU array operator op (array a, array const & b) \
	{ \
		a op ## = b; \
		return a; \
	} \
	CPUGPU array & operator op ## = (T const & o) \
	{ \
		for(int i = 0; i < N; ++i) \
			(*this)[i] op ## = o; \
		return *this; \
	} \
	friend CPUGPU array operator op (array a, T const & b) \
	{ \
		a op ## = b; \
		return a; \
	} \
	friend CPUGPU array operator op (T const & a, array const & b) \
	{ \
		array ret; \
		for(int i = 0; i < N; ++i) \
			ret[i] = a op b[i]; \
		return ret; \
	}

#define DEFINE_COMPARISON_ARRAY_OPERATOR(op) \
	friend CPUGPU array<bool, N> operator op (array const & a, array const & b) \
	{ \
		array<bool, N> ret; \
		for(int i = 0; i < N; ++i) \
			ret[i] = a[i] op b[i]; \
		return ret; \
	} \
	friend CPUGPU array<bool, N> operator op (array const & a, T const & b) \
	{ \
		array<bool, N> ret; \
		for(int i = 0; i < N; ++i) \
			ret[i] = a[i] op b; \
		return ret; \
	} \
	friend CPUGPU array<bool, N> operator op (T const & a, array const & b) \
	{ \
		array<bool, N> ret; \
		for(int i = 0; i < N; ++i) \
			ret[i] = a op b[i]; \
		return ret; \
	}

#define DEFINE_UNARY_ARRAY_OPERATOR(op) \
	CPUGPU array operator op () const \
	{ \
		array ret; \
		for(int i = 0; i < N; ++i) \
			ret[i] = op (*this)[i]; \
		return ret; \
	}

#define DEFINE_INCDEC_ARRAY_OPERATOR(op) \
	CPUGPU array& operator op () \
	{ \
		for(int i = 0; i < N; ++i) \
			op (*this)[i]; \
		return *this; \
	} \
	CPUGPU array operator op (int) \
	{ \
		array ret(*this); \
		op *this; \
		return ret; \
	}

#pragma push_macro("FUNC_ARRAY_NARY_CONSTRUCTOR_BODY")
#pragma push_macro("FUNC_ARRAY_NARY_CONSTRUCTOR")

#ifdef FUNC_ARRAY_NARY_CONSTRUCTOR_BODY
#undef FUNC_ARRAY_NARY_CONSTRUCTOR_BODY
#endif
#ifdef FUNC_ARRAY_NARY_CONSTRUCTOR
#undef FUNC_ARRAY_NARY_CONSTRUCTOR
#endif

#define FUNC_ARRAY_NARY_CONSTRUCTOR_BODY(z, k, _) \
	(*this)[k] = p ## k;
#define FUNC_ARRAY_NARY_CONSTRUCTOR(z, Nm2, _) \
	constexpr CPUGPU array(BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(BOOST_PP_INC(Nm2)), T const& p)) \
		: Storage{{BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(BOOST_PP_INC(Nm2)), p)}} \
	{}

namespace detail
{
	template<typename T, int N, bool align>
	struct array_storage
	{
		T data_[N];

		CPUGPU T (&data())[N] { return this->data_; }
		constexpr CPUGPU T const (&data() const)[N] { return this->data_; }
	};

	template<typename T, int N>
	struct ALIGN(N * sizeof(T)) array_storage<T, N, true>
	{
		T data_[N];

		CPUGPU T (&data())[N] { return this->data_; }
		constexpr CPUGPU T const (&data() const)[N] { return this->data_; }
	};

	template<typename T, bool align>
	struct array_storage<T, 0, align>
	{
		CPUGPU T * data() { return static_cast<T *>(nullptr); }
		constexpr CPUGPU T const * data() const { return static_cast<T *>(nullptr); }
	};
}

// do not use 64 bit size_t with cuda!!!
template<typename T, int N>
struct array : protected detail::array_storage<T, N, sizeof(T) * N == 4 || sizeof(T) * N == 8 || sizeof(T) * N == 16>
{
	using Storage = detail::array_storage<T, N, sizeof(T) * N == 4 || sizeof(T) * N == 8 || sizeof(T) * N == 16>;
	using Storage::data;

	constexpr /* CPUGPU */ array() = default;

	template<typename S>
	constexpr CPUGPU explicit array(
		array<S, N> const & o,
		typename std::enable_if<!std::is_same<S, T>::value>::type * = nullptr
	)
		: array(o, meta::make_integer_sequence<int, N>())
	{}

	explicit constexpr CPUGPU array(T const & o)
		: array(o, meta::make_integer_sequence<int, N>())
	{}

	CPUGPU array& operator=(T const & o)
	{
		for(int i = 0; i < N; ++i)
			(*this)[i] = o;
		return *this;
	}

	BOOST_PP_REPEAT(80, FUNC_ARRAY_NARY_CONSTRUCTOR, ~) // initializers with 2 to 81 elements... I wish I could use C++11

	CPUGPU T & operator[](int n) { return this->data()[n]; }
	constexpr CPUGPU T const & operator[](int n) const { return this->data()[n]; }

	CPUGPU T * begin() { return this->data() + 0; }
	CPUGPU T * end() { return this->data() + N; }
	constexpr CPUGPU T const * begin() const { return this->data() + 0; }
	constexpr CPUGPU T const * end() const { return this->data() + N; }

	constexpr CPUGPU int size() const { return N; }

	template<typename S>
	constexpr CPUGPU array<S, N> cast() const
	{
		return array<S, N>(*this);
	}

	DEFINE_BINARY_ARRAY_OPERATOR(+)
	DEFINE_BINARY_ARRAY_OPERATOR(*)
	DEFINE_BINARY_ARRAY_OPERATOR(-)
	DEFINE_BINARY_ARRAY_OPERATOR(/)
	DEFINE_BINARY_ARRAY_OPERATOR(%)
	DEFINE_BINARY_ARRAY_OPERATOR(<<)
	DEFINE_BINARY_ARRAY_OPERATOR(>>)
	DEFINE_BINARY_ARRAY_OPERATOR(|)
	DEFINE_BINARY_ARRAY_OPERATOR(&)
	DEFINE_BINARY_ARRAY_OPERATOR(^)
	DEFINE_COMPARISON_ARRAY_OPERATOR(==)
	DEFINE_COMPARISON_ARRAY_OPERATOR(!=)
	DEFINE_COMPARISON_ARRAY_OPERATOR(<)
	DEFINE_COMPARISON_ARRAY_OPERATOR(<=)
	DEFINE_COMPARISON_ARRAY_OPERATOR(>)
	DEFINE_COMPARISON_ARRAY_OPERATOR(>=)
	DEFINE_UNARY_ARRAY_OPERATOR(~)
	DEFINE_UNARY_ARRAY_OPERATOR(+)
	DEFINE_UNARY_ARRAY_OPERATOR(-)
	DEFINE_INCDEC_ARRAY_OPERATOR(++)
	DEFINE_INCDEC_ARRAY_OPERATOR(--)

	friend std::ostream& operator << (std::ostream& os, array const & a)
	{
		os << "[";
		for(int i = 0; i < N; ++i)
		{
			if(i != 0) os << ", ";
			os << a[i];
		}
		os << "]";
		return os;
	}

private:
	template<typename S, int... Indices>
	constexpr CPUGPU explicit array(
		array<S, N> const & o,
		meta::integer_sequence<int, Indices...>,
		typename std::enable_if<!std::is_same<S, T>::value && N != 0>::type * = nullptr
	)
		: Storage{{static_cast<T>(o[Indices])...}}
	{}

	template<typename S, int... Indices>
	constexpr CPUGPU explicit array(
		array<S, N> const &,
		meta::integer_sequence<int, Indices...>,
		typename std::enable_if<!std::is_same<S, T>::value && N == 0>::type * = nullptr
	)
	{}

	template<int... Indices>
	constexpr CPUGPU array(T const & o, meta::integer_sequence<int, Indices...>)
		: Storage{{((void)Indices, o)...}}
	{}
};

template<typename T, int N>
constexpr CPUGPU T prod(array<T, N> const & a, int i = 0)
{
	return i >= N ? T(1) : a[i] * prod(a, i + 1);
}

template<typename T, int N>
CPUGPU T sum(array<T, N> const & a)
{
	T ret(0);
	for(int i = 0; i < N; ++i)
		ret += a[i];
	return ret;
}

template<int N>
CPUGPU bool all(array<bool, N> const & a)
{
#ifndef __CUDA_ARCH__
	for(int i = 0; i < N; ++i)
		if(!a[i])
			return false;
	return true;
#else
	bool ret = true;
	for(int i = 0; i < N; ++i)
		ret &= a[i]; // bitwise instead of short circuiting (short circuiting prevents unrolling)
	return ret;
#endif
}

template<int N>
CPUGPU bool any(array<bool, N> const & a)
{
#ifndef __CUDA_ARCH__
	for(int i = 0; i < N; ++i)
		if(a[i])
			return true;
	return false;
#else
	bool ret = false;
	for(int i = 0; i < N; ++i)
		ret |= a[i]; // bitwise instead of short circuiting (short circuiting prevents unrolling)
	return ret;
#endif
}

namespace detail
{
	template<typename T, int N, int K>
	struct lexical_less_impl
	{
		static CPUGPU bool apply(array<T, N> const & a, array<T, N> const & b)
		{
			return a[K] < b[K] || (a[K] == b[K] && lexical_less_impl<T, N, K + 1>::apply(a, b));
		}
	};

	template<typename T, int N>
	struct lexical_less_impl<T, N, N>
	{
		static CPUGPU bool apply(array<T, N> const &, array<T, N> const &) { return false; }
	};
}

template<typename T, int N>
CPUGPU bool lexical_less(array<T, N> const & a, array<T, N> const & b)
{
	return detail::lexical_less_impl<T, N, 0>::apply(a, b);
}

template<typename T, int N, typename S>
CPUGPU array<T, N> add(array<T,N> ret, int ax, S val)
{
	ret[ax] += val;
	return ret;
}

template<typename T, int N>
CPUGPU array<T, N> min(array<T,N> ar1, array<T,N> ar2)
{
	for(int i = 0; i < N; ++i)
		ar1[i] = ar1[i] < ar2[i] ? ar1[i] : ar2[i];
	return ar1;
}

template<typename T, int N>
CPUGPU T min(array<T, N> ar1)
{
	T ret = ar1[0];
	for(int i = 1; i < N; ++i)
		ret = ar1[i] < ret ? ar1[i] : ret;
	return ret;
}

template<typename T, int N>
CPUGPU array<T, N> max(array<T,N> ar1, array<T,N> ar2)
{
	for(int i = 0; i < N; ++i)
		ar1[i] = ar1[i] > ar2[i] ? ar1[i] : ar2[i];
	return ar1;
}

template<typename T, int N, typename S>
CPUGPU array<T, N> sub(array<T,N> ret, int ax, S val)
{
	ret[ax] -= val;
	return ret;
}

template<int N, template<typename, typename> class Container, typename T, typename Allocator>
array<T, N> make_array_from_container(Container<T, Allocator> const & o)
{
	array<T, N> ret;
	for(int n = N; n--;)
		ret[n] = o[n];
	return ret;
}

namespace detail
{
	template<typename E, typename F, int N, typename R = typename std::decay<decltype(std::declval<F>()(std::declval<E>()))>::type, int... Indices>
	CPUGPU constexpr array<R, N> map_helper(F const & f, array<E, N> const & i, meta::integer_sequence<int, Indices...>)
	{
		return array<R, N>{f(i[Indices])...};
	}
}

template<typename E, typename F, int N, typename R = typename std::decay<decltype(std::declval<F>()(std::declval<E>()))>::type>
CPUGPU constexpr array<R, N> map(F f, array<E, N> const & i)
{
	return ::detail::map_helper(f, i, meta::make_integer_sequence<int, N>());
}

#pragma pop_macro("FUNC_ARRAY_NARY_CONSTRUCTOR")
#pragma pop_macro("FUNC_ARRAY_NARY_CONSTRUCTOR_BODY")
#pragma pop_macro("DEFINE_INCDEC_ARRAY_OPERATOR")
#pragma pop_macro("DEFINE_UNARY_ARRAY_OPERATOR")
#pragma pop_macro("DEFINE_COMPARISON_ARRAY_OPERATOR")
#pragma pop_macro("DEFINE_BINARY_ARRAY_OPERATOR")

#include "cpugpu_macros_pop.inl"
