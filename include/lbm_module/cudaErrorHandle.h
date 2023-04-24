#ifndef __ERROR_H__
#define __ERROR_H__
#include <stdio.h>
#include <cuda_runtime_api.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

static void printGPUInfo(int dev)
{
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));

    printf(" --- General Information for device %d ---\n", dev);
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Device copy overlap: ");
    if (prop.deviceOverlap)
        printf("Enabled\n");
    else
        printf("Disabled\n");
    printf("Kernel execition timeout : ");
    if (prop.kernelExecTimeoutEnabled)
        printf("Enabled\n");
    else
        printf("Disabled\n");
    printf("Concurrent Kernel Execution : ");
    if (prop.concurrentKernels)
        printf("Enabled\n");
    else
        printf("Disabled\n");

    printf(" --- Memory Information for device %d ---\n", dev);
    printf("Total global mem: %lld\n", prop.totalGlobalMem);
    printf("Total constant Mem: %lld\n", prop.totalConstMem);
    printf("Max mem pitch: %lld\n", prop.memPitch);
    printf("Texture Alignment: %lld\n", prop.textureAlignment);

    printf(" --- MP Information for device %d ---\n", dev);
    printf("Multiprocessor count: %d\n",
        prop.multiProcessorCount);
    printf("Shared mem per mp: %zd\n", prop.sharedMemPerBlock);
    printf("Registers per mp: %d\n", prop.regsPerBlock);
    printf("Threads in warp: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n",
        prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1],
        prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1],
        prop.maxGridSize[2]);
    printf("\n\n");
}







#endif  // __ERROR_H__
