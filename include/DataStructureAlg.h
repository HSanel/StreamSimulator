#ifndef LBM_TYPES
#define LBM_TYPES

#include <array>

constexpr unsigned int packsize = 32;

#define packCount(nodeCount)((packsize + nodeCount)/packSize)

#if defined(AoS) && ! defined(SoA)
#define dataLayout(Q, alpha, pos, i)(Q * pos + i)
#endif

#if defined(SoA) && ! defined(AoS)
#define dataLayout(Q, alpha, pos, i)(alpha * i + pos)
#endif

#if ! defined(SoA) && ! defined(AoS) || defined(SoA) && defined(AoS)
#define dataLayout(Q, alpha, pos, i)(Q * alpha * (pos / alpha) + alpha * i + pos % alpha)
#endif

#define arrayLayout(pos, axis)(3 * packsize * (pos / packsize) + packsize * axis + pos % packsize)

#define packCount(nodeCount)((nodeCount + packsize - 1) / packsize )

void pack_CPU(std::array<float, 3>* array_out, float* array_in, unsigned int count);

void pack_GPU(float* array_out, std::array<float, 3>* array_in, unsigned int count);
#endif // !LBM_TYPES
