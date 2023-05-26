#include "DataStructureAlg.h"


void pack_CPU(std::array<float, 3>* array_out, float* array_in, unsigned int count)
{
	for (int i = 0; i < count; ++i)
		array_out[i] = { array_in[arrayLayout(i, 0)], array_in[arrayLayout(i, 1)] , array_in[arrayLayout(i, 2)] };
}

void pack_GPU(float* array_out, std::array<float, 3>* array_in, unsigned int count)
{
	for (int i = 0; i < count; ++i)
	{
		array_out[arrayLayout(i, 0)] = array_in[i][0];
		array_out[arrayLayout(i, 1)] = array_in[i][1];
		array_out[arrayLayout(i, 2)] = array_in[i][2];
	} 
}

