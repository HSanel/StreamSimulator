#pragma once

#include "binom.h"
#include "cpugpu_macros_push.inl"

template<typename T, int N>
struct array;

namespace meta
{
	template<int... Is> struct multi_index {};

	namespace detail
	{
		template<int I, int I0, int... Is>
		struct get_impl
		{
			static CPUGPU FORCEINLINE constexpr int value() { return get_impl<I - 1, Is...>::value(); }
		};

		template<int I0, int... Is>
		struct get_impl<0, I0, Is...>
		{
			static CPUGPU FORCEINLINE constexpr int value() { return I0; }
		};

		template<int I0, int... Is>
		struct sum_impl
		{
			static CPUGPU FORCEINLINE constexpr int value() { return I0 + sum_impl<Is...>::value(); }
		};

		template<int I0>
		struct sum_impl<I0>
		{
			static CPUGPU FORCEINLINE constexpr int value() { return I0; }
		};

		template<int I0, int... Is> struct last_nonzero_impl
		{
			static CPUGPU FORCEINLINE constexpr int value() { return sizeof...(Is); }
		};

		template<int... Is>
		struct last_nonzero_impl<0, Is...>
		{
			static CPUGPU FORCEINLINE constexpr int value() { return last_nonzero_impl<Is...>::value(); }
		};

		template<>
		struct last_nonzero_impl<0>
		{
			static CPUGPU FORCEINLINE constexpr int value() { return -1; }
		};

		template<int... Is>
		CPUGPU FORCEINLINE constexpr int last_nonzero_helper(multi_index<Is...>)
		{
			return last_nonzero_impl<Is...>::value();
		}
	}

	template<int... Is>
	CPUGPU FORCEINLINE constexpr int size(multi_index<Is...>)
	{
		return sizeof...(Is);
	}

	template<int I, int... Is>
	CPUGPU FORCEINLINE constexpr int get(multi_index<Is...>)
	{
		static_assert(0 <= I && I < sizeof...(Is), "I must be in [0, sizeof...(Is))");
		return detail::get_impl<I, Is...>::value();
	}

	template<int... Is>
	CPUGPU FORCEINLINE constexpr int sum(multi_index<Is...>)
	{
		return detail::sum_impl<Is...>::value();
	}

	template<int... Is0, int... Is1>
	CPUGPU FORCEINLINE constexpr auto concat(multi_index<Is0...>, multi_index<Is1...>)
		-> multi_index<Is0..., Is1...>
	{
		return {};
	}

	namespace detail
	{
		template<typename I>
		struct reverse_impl;

		template<>
		struct reverse_impl<multi_index<>>
		{
			static CPUGPU FORCEINLINE constexpr multi_index<> apply() { return {}; }
		};

		template<int I0, int... Is>
		struct reverse_impl<multi_index<I0, Is...>>
		{
			static CPUGPU FORCEINLINE constexpr auto apply()
				-> decltype(concat(reverse_impl<multi_index<Is...>>::apply(), multi_index<I0>()))
			{
				return {};
			}
		};
	}

	template<typename I>
	CPUGPU FORCEINLINE constexpr auto reverse(I)
		-> decltype(detail::reverse_impl<I>::apply())
	{
		return {};
	}

	template<int... Is>
	CPUGPU FORCEINLINE constexpr int last_nonzero(multi_index<Is...> i)
	{
		return detail::last_nonzero_helper(reverse(i));
	}

	namespace detail
	{
		template<int Pos, int Val, typename I, typename R = multi_index<>>
		struct set_impl;

		template<int Val, int I0, int... Is, int... Rs>
		struct set_impl<0, Val, multi_index<I0, Is...>, multi_index<Rs...>>
		{
			static CPUGPU FORCEINLINE constexpr multi_index<Rs..., Val, Is...> value() { return {}; }
		};

		template<int Pos, int Val, int I0, int... Is, int... Rs>
		struct set_impl<Pos, Val, multi_index<I0, Is...>, multi_index<Rs...>>
			: set_impl<Pos - 1, Val, multi_index<Is...>, multi_index<Rs..., I0>>
		{};

		template<int Pos, int Val, typename I, typename R = multi_index<>>
		struct add_impl;

		template<int Val, int I0, int... Is, int... Rs>
		struct add_impl<0, Val, multi_index<I0, Is...>, multi_index<Rs...>>
		{
			static CPUGPU FORCEINLINE constexpr multi_index<Rs..., I0 + Val, Is...> value() { return{}; }
		};

		template<int Pos, int Val, int I0, int... Is, int... Rs>
		struct add_impl<Pos, Val, multi_index<I0, Is...>, multi_index<Rs...>>
			: add_impl<Pos - 1, Val, multi_index<Is...>, multi_index<Rs..., I0>>
		{};
	}

	template<int Pos, int Val, typename I>
	CPUGPU FORCEINLINE constexpr auto set(I)
		-> decltype(detail::set_impl<Pos, Val, I>::value())
	{
		return {};
	}

	template<int Pos, int Val, typename I>
	CPUGPU FORCEINLINE constexpr auto add(I)
		-> decltype(detail::add_impl<Pos, Val, I>::value())
	{
		return{};
	}

	namespace detail
	{
		template<bool IsLast, int Last, typename I>
		struct next_impl
		{
			static CPUGPU FORCEINLINE constexpr auto value()
				-> decltype(add<Last + 1, 1>(add<Last, -1>(I())))
			{
				return {};
			}
		};

		template<int Last, typename I>
		struct next_impl<true, Last, I>
		{
			using Itemp = decltype(set<Last, 0>(I()));
			static CPUGPU FORCEINLINE constexpr int Prev()
			{
				return last_nonzero(Itemp());
			}
			static CPUGPU FORCEINLINE constexpr int V()
			{
				return get<Last>(I());
			}
			static CPUGPU FORCEINLINE constexpr auto value()
				-> decltype(add<Prev() + 1, V() + 1>(add<Prev(), -1>(Itemp())))
			{
				return {};
			}
		};

		template<int D, int P>
		struct make_multi_index_impl
		{
			static CPUGPU FORCEINLINE constexpr auto value()
				-> decltype(set<0, P>(make_multi_index_impl<D, 0>::value()))
			{
				return {};
			}
		};

		template<int D>
		struct make_multi_index_impl<D, 0>
		{
			static CPUGPU FORCEINLINE constexpr auto value()
				-> decltype(concat(multi_index<0>(), make_multi_index_impl<D - 1, 0>::value()))
			{
				return {};
			}
		};

		template<>
		struct make_multi_index_impl<0, 0>
		{
			static CPUGPU FORCEINLINE constexpr multi_index<> value() { return {}; }
		};
	}

	template<typename I, int Last = last_nonzero(I())>
	CPUGPU FORCEINLINE constexpr auto next(I)
		-> decltype(detail::next_impl<Last == size(I()) - 1, Last, I>::value())
	{
		return {};
	}

	template<typename I>
	CPUGPU FORCEINLINE constexpr bool have_next(I)
	{
		return sum(I()) != get<size(I()) - 1>(I());
	}

	template<int D, int P>
	CPUGPU FORCEINLINE constexpr auto make_multi_index()
		-> decltype(detail::make_multi_index_impl<D, P>::value())
	{
		return {};
	}

	namespace detail
	{
		template<int K, bool HaveNext, typename I>
		struct for_each_multi_index_impl
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f(I());
				using Inext = decltype(next(I()));
				for_each_multi_index_impl<K + 1, have_next(Inext()), Inext>::apply(f);
			}

			template<typename Functor>
			static CPUGPU FORCEINLINE void apply_enumerate(Functor && f)
			{
				f.template apply<K>(I());
				using Inext = decltype(next(I()));
				for_each_multi_index_impl<K + 1, have_next(Inext()), Inext>::apply_enumerate(f);
			}
		};

		template<int K, typename I>
		struct for_each_multi_index_impl<K, false, I>
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f(I());
			}

			template<typename Functor>
			static CPUGPU FORCEINLINE void apply_enumerate(Functor && f)
			{
				f.template apply<K>(I());
			}
		};
	}

	template<int D, int P, typename Functor>
	CPUGPU FORCEINLINE void for_each_multi_index(Functor && f)
	{
		using I = decltype(make_multi_index<D, P>());
		detail::for_each_multi_index_impl<0, have_next(I()), I>::apply(f);
	}

	template<int D, int P, typename Functor>
	CPUGPU FORCEINLINE void enumerate_each_multi_index(Functor && f)
	{
		using I = decltype(make_multi_index<D, P>());
		detail::for_each_multi_index_impl<0, have_next(I()), I>::apply_enumerate(f);
	}

	template<int... Is>
	CPUGPU FORCEINLINE constexpr auto to_array(multi_index<Is...>)
		-> array<int, sizeof...(Is)>
	{
		return array<int, sizeof...(Is)>(Is...);
	}

	namespace detail
	{
		template<bool B, typename T = void> struct enable_if_impl {};
		template<typename T> struct enable_if_impl<true, T> { using type = T; };
	}

	template<bool B, typename T = void>
	using enable_if = typename detail::enable_if_impl<B, T>::type;

	namespace detail
	{
		template<int K, typename I, bool HaveNext = (K + 1 < size(I())), bool IsNonZero = (get<K>(I()) != 0)>
		struct for_each_nonzero_impl
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f.template apply<K>();
				for_each_nonzero_impl<K + 1, I>::apply(f);
			}
		};

		template<int K, typename I>
		struct for_each_nonzero_impl<K, I, true, false>
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				for_each_nonzero_impl<K + 1, I>::apply(f);
			}
		};

		template<int K, typename I>
		struct for_each_nonzero_impl<K, I, false, true>
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f.template apply<K>();
			}
		};

		template<int K, typename I>
		struct for_each_nonzero_impl<K, I, false, false>
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor &&) {}
		};
	}

	template<typename I, typename Functor>
	CPUGPU FORCEINLINE void for_each_nonzero(I, Functor&& f)
	{
		detail::for_each_nonzero_impl<0, I>::apply(f);
	}

	namespace detail
	{
		template<typename in, typename out = meta::multi_index<>>
		struct nonzero_values_impl;

		template<int... Is, int... Js>
		struct nonzero_values_impl<meta::multi_index<0, Is...>, meta::multi_index<Js...>>
			: nonzero_values_impl<meta::multi_index<Is...>, meta::multi_index<Js...>>
		{};

		template<int I0, int... Is, int... Js>
		struct nonzero_values_impl<meta::multi_index<I0, Is...>, meta::multi_index<Js...>>
			: nonzero_values_impl<meta::multi_index<Is...>, meta::multi_index<Js..., I0>>
		{};

		template<int... Js>
		struct nonzero_values_impl<meta::multi_index<>, meta::multi_index<Js...>>
		{
			using type = meta::multi_index<Js...>;
		};

		template<typename in, typename out = meta::multi_index<>, int K = 0>
		struct nonzero_indices_impl;

		template<int... Is, int... Js, int K>
		struct nonzero_indices_impl<meta::multi_index<0, Is...>, meta::multi_index<Js...>, K>
			: nonzero_indices_impl<meta::multi_index<Is...>, meta::multi_index<Js...>, K + 1>
		{};

		template<int I0, int... Is, int... Js, int K>
		struct nonzero_indices_impl<meta::multi_index<I0, Is...>, meta::multi_index<Js...>, K>
			: nonzero_indices_impl<meta::multi_index<Is...>, meta::multi_index<Js..., K>, K + 1>
		{};

		template<int... Js, int K>
		struct nonzero_indices_impl<meta::multi_index<>, meta::multi_index<Js...>, K>
		{
			using type = meta::multi_index<Js...>;
		};
	}

	template<typename MI>
	using nonzero_values = typename detail::nonzero_values_impl<MI>::type;

	template<typename MI>
	using nonzero_indices = typename detail::nonzero_indices_impl<MI>::type;

	namespace detail
	{
		template<int Num, typename I, typename J>
		struct bernstein_num_impl
		{
			static CPUGPU FORCEINLINE constexpr int value() { return Num; }
		};

		template<int Num, int I0, int... Is, int J0, int... Js>
		struct bernstein_num_impl<Num, multi_index<I0, Is...>, multi_index<J0, Js...>>
		{
			static CPUGPU FORCEINLINE constexpr int value()
			{
				return bernstein_num_impl<Num * binom(I0 + J0, I0), multi_index<Is...>, multi_index<Js...>>::value();
			}
		};
	}

	template<typename I, typename J>
	CPUGPU FORCEINLINE constexpr float bernstein_coeff(I, J)
	{
		static_assert(size(I()) == size(J()), "Bernstein-Bézier dimensions must match");
		return float(detail::bernstein_num_impl<1, I, J>::value()) / binom(sum(I()) + sum(J()), sum(I()));
	}

	namespace detail
	{
		template<int N, int M, int K, int I, int J>
		struct for_each_combination_impl
		{
			template<int... Is, typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				for_each_combination_impl<N, M, K, I + 1, J + 1>::template apply<Is..., J>(static_cast<decltype(f) &&>(f));
				for_each_combination_impl<N, M, K + binom(N - J - 1, M - I - 1), I, J + 1>::template apply<Is...>(static_cast<decltype(f) &&>(f));
			}
		};

		template<int N, int M, int K, int J>
		struct for_each_combination_impl<N, M, K, M, J>
		{
			template<int... Is, typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f.template apply<K, Is...>();
			}
		};

		template<int N, int M, int K>
		struct for_each_combination_impl<N, M, K, M, N>
		{
			template<int... Is, typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f.template apply<K, Is...>();
			}
		};

		template<int N, int M, int K, int I>
		struct for_each_combination_impl<N, M, K, I, N>
		{
			template<int..., typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor &&) {}
		};
	}

	template<int N, int M, typename Functor>
	CPUGPU FORCEINLINE void for_each_combination(Functor && f)
	{
		detail::for_each_combination_impl<N, M, 0, 0, 0>::apply(static_cast<decltype(f) &&>(f));
	}

	namespace detail
	{
		template <int N, typename I>
		struct for_each_index_of_multi_index_impl
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f.template apply<N, I>(); 
				for_each_index_of_multi_index_impl<N - 1, I>::apply(f);
			}
		};

		template <typename I>
		struct for_each_index_of_multi_index_impl<0, I>
		{
			template<typename Functor>
			static CPUGPU FORCEINLINE void apply(Functor && f)
			{
				f.template apply<0, I>();
			}
		};
	}

	template<typename I, typename Functor>
	CPUGPU FORCEINLINE void for_each_index_of_multi_index(I, Functor&& f)
	{
		detail::for_each_index_of_multi_index_impl<meta::size(I()) - 1, I>::apply(f);
	}



	template<typename Int, Int... Is>
	struct integer_sequence {};

	namespace detail
	{
		template<typename Int, Int Sp, typename IntPack, Int Ep>
		struct make_integer_sequence_impl;

		template<typename Int, Int Sp, Int Ep, Int... Indices>
		struct make_integer_sequence_impl<Int, Sp, integer_sequence<Int, Indices...>, Ep>
		{
			using type = typename make_integer_sequence_impl<Int, Sp + 1, integer_sequence<Int, Indices..., Sp>, Ep>::type;
		};

		template<typename Int, Int Ep, Int... Indices>
		struct make_integer_sequence_impl<Int, Ep, integer_sequence<Int, Indices...>, Ep>
		{
			using type = integer_sequence<Int, Indices...>;
		};
	}

	template<typename Int, Int Ep, Int Sp = Int(0)>
	using make_integer_sequence = typename detail::make_integer_sequence_impl<Int, Sp, integer_sequence<Int>, Ep>::type;

	namespace detail
	{
		template<int P, int N, int CP, int CN, int bc = int(binom(P + int(N - CN) - 1 - CP - 1, int(N - CN) - 1))>
		struct multi_index_to_linear_impl
		{
			static CPUGPU FORCEINLINE constexpr int apply(array<int, N> const & mi)
			{
				return mi[CN] == CP ?
					bc + multi_index_to_linear_impl<P - CP, N, P - CP, CN + 1>::apply(mi) :
					multi_index_to_linear_impl<P, N, CP - 1, CN>::apply(mi);
			}
		};

		template<int P, int N, int CN, int bc>
		struct multi_index_to_linear_impl<P, N, 0, CN, bc>
		{
			static CPUGPU FORCEINLINE constexpr int apply(array<int, N> const & mi)
			{
				return bc + multi_index_to_linear_impl<P, N, P, CN + 1>::apply(mi);
			}
		};

		template<int P, int N, int CP, int bc>
		struct multi_index_to_linear_impl<P, N, CP, N, bc>
		{
			static CPUGPU FORCEINLINE constexpr int apply(array<int, N> const &)
			{
				return 0;
			}
		};

		template<int P, int N, int bc>
		struct multi_index_to_linear_impl<P, N, 0, N, bc>
		{
			static CPUGPU FORCEINLINE constexpr int apply(array<int, N> const &)
			{
				return 0;
			}
		};

		template<int N, int CN, int bc>
		struct multi_index_to_linear_impl<0, N, 0, CN, bc>
		{
			static CPUGPU FORCEINLINE constexpr int apply(array<int, N> const &)
			{
				return 0;
			}
		};

		template<int N, int bc>
		struct multi_index_to_linear_impl<0, N, 0, N, bc>
		{
			static CPUGPU FORCEINLINE constexpr int apply(array<int, N> const &)
			{
				return 0;
			}
		};
	}

	template<int P, int N>
	constexpr CPUGPU int multi_index_to_linear(array<int, N> const & mi)
	{
		return detail::multi_index_to_linear_impl<P, N, P, 0>::apply(mi);
	}

	template<int>
	constexpr CPUGPU int combination_to_linear(meta::multi_index<>)
	{
		return 0;
	}

	template<
		int N,
		int I0,
		int... Is,
		int M = 1 + int(sizeof...(Is)),
		int bc1 = int(binom(N, M)),
		int bc2 = int(binom(N - I0, M))
	>
	constexpr CPUGPU int combination_to_linear(meta::multi_index<I0, Is...>)
	{
		static_assert(0 <= I0 && I0 < N, "index out of range!");
		return (bc1 - bc2) + combination_to_linear<N - I0 - 1>(meta::multi_index<(Is - I0 - 1)...>());
	}

	template<typename I>
	constexpr CPUGPU I factorial(I i)
	{
		return i <= I(1) ? I(1) : factorial(i - 1) * i;
	}

	namespace detail
	{
		template<unsigned i, typename Real>
		constexpr CPUGPU FORCEINLINE auto ipow_recursion(Real r, Real)
			-> meta::enable_if<(i == 0), Real>
		{
			return r;
		}

		template<unsigned i, typename Real>
		constexpr CPUGPU FORCEINLINE auto ipow_recursion(Real r, Real v)
			-> meta::enable_if<(i > 0 && (i & 1) != 0), Real>;

		template<unsigned i, typename Real>
		constexpr CPUGPU FORCEINLINE auto ipow_recursion(Real r, Real v)
			-> meta::enable_if<(i > 0 && (i & 1) == 0), Real>
		{
			return ipow_recursion<(i >> 1)>(r, v * v);
		}

		template<unsigned i, typename Real>
		constexpr CPUGPU FORCEINLINE auto ipow_recursion(Real r, Real v)
			-> meta::enable_if<(i > 0 && (i & 1) != 0), Real>
		{
			return ipow_recursion<(i >> 1)>(v * r, v * v);
		}
	}

	template<int i, typename Real>
	constexpr CPUGPU FORCEINLINE auto ipow(Real v)
		-> meta::enable_if<(i >= 0), Real>
	{
		return detail::ipow_recursion<i>(Real(1), v);
	}

	template<int i, typename Real>
	constexpr CPUGPU FORCEINLINE auto ipow(Real v)
		-> meta::enable_if<(i < 0), Real>
	{
		return Real(1) / detail::ipow_recursion<-unsigned(i)>(Real(1), v);
	}
}

#include "cpugpu_macros_pop.inl"
