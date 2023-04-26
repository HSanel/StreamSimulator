#include "LBMKernels.h"
#include <iostream>


__global__ void test_kernel()
{
	printf("Hallo Welt\n");
}

void test_call()
{
	test_kernel<<<2,8>>>();
}
