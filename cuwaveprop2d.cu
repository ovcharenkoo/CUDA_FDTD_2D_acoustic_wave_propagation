#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"

/*
Add this to c_cpp_properties.json if linting isn't working for cuda libraries
"includePath": [
                "/usr/local/cuda-9.0/targets/x86_64-linux/include",
                "${workspaceFolder}/**"
            ],
*/          
#include "cuda.h"
#include "cuda_runtime.h"


// Check error codes for CUDA functions
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}





int main( int argc, char *argv[])
{
    // Get device count
    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    // Print device properties
    cudaDeviceProp deviceProp;
    for(int dev = 0; dev < ngpus; dev++)
    {
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("%d: %s\n", dev, deviceProp.name);
    }

    // For now only, use only one device
    CHECK(cudaSetDevice(ngpus-1))

    return 0;
}
