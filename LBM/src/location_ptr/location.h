#pragma once
#define WITH_CUDA
struct location_cpu {};
struct location_gpu {};

inline char const * to_string(location_cpu) noexcept { return "CPU";}
inline char const * to_string(location_gpu) noexcept { return "GPU";}

// locationT for SimplexMesh in ElasticityModels: Always passing a gpu-tetmesh except when cuda is deactivated
#ifdef WITH_CUDA
using locationT = location_gpu;
#else
using locationT = location_cpu;
#endif // WITH_CUDA
