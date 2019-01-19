/*
Hello world of wave propagation in CUDA. FDTD acoustic wave propagation in homogeneous medium. Second order in space and time 
*/

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
    cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define PI      3.14159265359
#define PAD     4
#define a0     -3.0124472f
#define a1      1.7383092f
#define a2     -0.2796695f
#define a3      0.0547837f
#define a4     -0.0073118f

// Allocate stencil coefficients in the constant device memory
__device__ __constant__ float coef[5];



int main( int argc, char *argv[])
{
    // Print out name of the main GPU
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("%d: %s\n", 0, deviceProp.name);
    CHECK(cudaSetDevice(0));

    // Model dimensions
    int nx    = 1024;                       /* x dim */
    int nz    = 1024;                       /* z dim */

    size_t nxz = nx * nz;   
    size_t nbytes = nxz * sizeof(float);    /* bytes to store nx * nz */
    
    float dx = 1;                           /* m */
    float dz = dx;
    
    // Allocate memory for velocity model
    float _vp = 3300;                       /* m/s, p-wave velocity */
    float *h_vp;
    h_vp = (float *)malloc(nbytes);
    memset(h_vp, _vp, nbytes);              /* initiate h_vp with _vp */

    // Time stepping
    float t_total = 0.55;                   /* sec, total time of wave propagation */
    float dt = 0.7 * fmin(dx, dz) / _vp;    /* sec, time step assuming constant vp */
    float nt = round(t_total / dt);         /* number of time steps */

    // Source
    float f0 = 10.0;                        /* Hz, source dominant frequency */
    float t0 = 1.2 / f0;                    /* source padding to move wavelet from left of zero */
    int jsrc = round((float) nz / 2);       /* source location, oz */
    int isrc = round((float) nx / 2);       /* source location, ox */

    float *h_wavelet, *h_time;
    h_time = (float *) malloc(nt * sizeof(float));
    h_wavelet = (float *) malloc(nt * sizeof(float));

    // Fill source waveform vecror
    float a = PI * PI * f0 * f0;    /* const for wavelet */
    for(size_t it = 0; it < nt; it++)
    {
        h_time[it] = it * dt;
        h_wavelet[it] = 1e10 * (1.0 - 2.0*a*(h_time[it] - t0));
    }
    

    // Allocate memory on GPU
    float *d_u1, *d_u2;
    CHECK(cudaMalloc((void **) &d_u1, nbytes))
    CHECK(cudaMalloc((void **) &d_u2, nbytes))
    // Fill those memory with zeros
    CHECK(cudaMemset(d_u1, 0, nbytes))
    CHECK(cudaMemset(d_u2, 0, nbytes))

    // Put stencil coefficients into GPU's constant memory
    float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(coef, h_coef, 5 * sizeof(float)));

    
    for(int istep = 0; istep < nt; istep++)
    {
    //    kernel_2dfd<<<grid, block>>>(d_u1, d_u2)
    }
    


    return 0;
}
