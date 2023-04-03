#pragma once

#include "cpugpu_macros_push.inl"

namespace detail
{
	constexpr CPUGPU unsigned binom_impl(unsigned n, unsigned k)
	{
		return k == 0 ? 1 : n * binom_impl(n - 1, k - 1) / k;
	}
}

constexpr CPUGPU unsigned binom(unsigned n, unsigned k)
{
	return k > n ? 0 : detail::binom_impl(n, k < n - k ? k : n - k);
}

constexpr CPUGPU unsigned binom(int n, int k)
{
	return n < 0 || k < 0 ? 0 : binom(unsigned(n), unsigned(k));
}

#include "cpugpu_macros_pop.inl"
