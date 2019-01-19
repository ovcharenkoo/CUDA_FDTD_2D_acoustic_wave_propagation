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

// Allocate the constant device memory
__constant__ float c_coef[5];       /* coefficients for 8th order fd */
__constant__ int c_isrc;            /* source location, ox */
__constant__ int c_jsrc;            /* source location, oz */
__constant__ int c_nx;              /* x dim */
__constant__ int c_ny;              /* y dim */

// Add source wavelet
__global__ void kernel_add_wavelet(float *d_u1, float *d_wavelet)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * c_nx + ix;

    if (ix == c_isrc) {
        printf("!!!%i",ix);
    }
    
    printf("GPU kernel add wavelet %i \n", idx);
}

// FD kernel
__global__ void kernel_2dfd(float *d_u1, float *d_u2, float *d_vp)
{

}

int main( int argc, char *argv[])
{
    // Print out name of the main GPU
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("%d: %s\n", 0, deviceProp.name);
    CHECK(cudaSetDevice(0));

    // Model dimensions
    int nx    = 1024;                       /* x dim */
    int ny    = 1024;                       /* z dim */

    // Add padding for derivatives
    nx += 2 * PAD;
    ny += 2 * PAD;

    size_t nxy = nx * ny;   
    size_t nbytes = nxy * sizeof(float);    /* bytes to store nx * ny */
    
    float dx = 1;                           /* m */
    float dy = dx;
    
    // Allocate memory for velocity model
    float _vp = 3300;                       /* m/s, p-wave velocity */
    float *h_vp;
    h_vp = (float *)malloc(nbytes);
    memset(h_vp, _vp, nbytes);              /* initiate h_vp with _vp */

    // Time stepping
    float t_total = 0.05;                   /* sec, total time of wave propagation */
    float dt = 0.7 * fmin(dx, dy) / _vp;    /* sec, time step assuming constant vp */
    int nt = round(t_total / dt);         /* number of time steps */

    // Source
    float f0 = 10.0;                        /* Hz, source dominant frequency */
    float t0 = 1.2 / f0;                    /* source padding to move wavelet from left of zero */

    float *h_wavelet, *h_time;
    h_time = (float *) malloc(nt * sizeof(float));
    h_wavelet = (float *) malloc(nt * sizeof(float));

    // Fill source waveform vecror
    float a = PI * PI * f0 * f0;            /* const for wavelet */
    for(size_t it = 0; it < nt; it++)
    {
        h_time[it] = it * dt;
        h_wavelet[it] = 1e10 * (1.0 - 2.0*a*pow(h_time[it] - t0, 2))*exp(-a*pow(h_time[it] - t0, 2));
        h_wavelet[it] *= dt * dt / (dx * dy);
    }

    // Allocate memory on device
    float *d_u1, *d_u2, *d_vp, *d_wavelet;
    CHECK(cudaMalloc((void **) &d_u1, nbytes))          /* wavefield at t-1 */
    CHECK(cudaMalloc((void **) &d_u2, nbytes))          /* wavefield at t-2 */
    CHECK(cudaMalloc((void **) &d_vp, nbytes))          /* velocity model */
    CHECK(cudaMalloc((void **) &d_wavelet, nbytes));    /* source term for each time step */
    
    // Transfer data to device
    CHECK(cudaMemset(d_u1, 0, nbytes))
    CHECK(cudaMemset(d_u2, 0, nbytes))
    CHECK(cudaMemcpy(d_vp, h_vp, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wavelet, h_wavelet, nbytes, cudaMemcpyHostToDevice));

    float coef[] = {a0, a1, a2, a3, a4};
    int isrc = round((float) nx / 2);                 /* source location, ox */
    int jsrc = round((float) ny / 2);                 /* source location, oz */

    CHECK(cudaMemcpyToSymbol(c_coef, coef, 5 * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_isrc, &isrc, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_jsrc, &jsrc, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nx, &nx, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_ny, &ny, sizeof(int)));
    
    printf("%i\tnt\n",nt);
    for(int istep = 0; istep < nt; istep++)
    {
    //    kernel_2dfd<<<grid, block>>>(d_u1, d_u2)
        kernel_add_wavelet<<<32,1>>>(d_u1, d_wavelet);
    }
    
    CHECK(cudaFree(d_u1));
    CHECK(cudaFree(d_u2));
    CHECK(cudaFree(d_vp));
    CHECK(cudaFree(d_wavelet));

    CHECK(cudaDeviceReset());



    return 0;
}
