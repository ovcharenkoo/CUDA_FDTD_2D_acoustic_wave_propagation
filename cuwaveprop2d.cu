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
                "/usr/local/cuda-10.0/targets/x86_64-linux/include",
                "${workspaceFolder}/**"
            ],
*/

#include "cuda.h"
#include "cuda_runtime.h"

// Check error codes for CUDA functions
#define CHECK(call)                                                \
    {                                                              \
        cudaError_t error = call;                                  \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

#define PI 3.14159265359
#define HALO 4
#define HALO2 8
#define a0 -3.0124472f
#define a1 1.7383092f
#define a2 -0.2796695f
#define a3 0.0547837f
#define a4 -0.0073118f

#define BDIMX 16
#define BDIMY 16

#define SDIMX BDIMX + HALO2
#define SDIMY BDIMY + HALO2

// Allocate the constant device memory
__constant__ float c_coef[5]; /* coefficients for 8th order fd */
__constant__ int c_isrc;      /* source location, ox */
__constant__ int c_jsrc;      /* source location, oz */
__constant__ int c_nx;        /* x dim */
__constant__ int c_ny;        /* y dim */
__constant__ int c_nt;        /* time steps */


// Save snapshot as a binary
void saveSnapshotIstep(int istep, float *data, int ny, int nx)
{
    float *iwave = (float *)malloc(nx * ny * sizeof(float));

    unsigned int isize = nx * ny;
    CHECK(cudaMemcpy(iwave, data, isize * sizeof(float), cudaMemcpyDeviceToHost));

    char fname[20];
    sprintf(fname, "snap_%i_%i_%i", istep, ny, nx);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    printf("%s: nx = %i ny = %i istep = %i\n", fname, nx, ny, istep);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}

// Add source wavelet
__global__ void kernel_add_wavelet(float *d_u1, float *d_wavelet, int it)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * c_nx + ix;

    if ((ix == c_isrc) && (iy == c_jsrc))
    {
        d_u1[idx] += d_wavelet[it];
        // printf("%d %f\n",it, d_wavelet[it]);
    }
}

__device__ void set_halo(float *global, float shared[][SDIMX], int tx, int ty, int gx, int gy, int nx, int ny)
{
    const int sx = tx + HALO;
    const int sy = ty + HALO;

    // fill inner smem
    shared[sy][sx] = global[gy * nx + gx];

    // LEFT
    if (tx < HALO)
    {
        if (gx < HALO)
        {
            // if global left
            shared[sy][sx - HALO] = 0.0;
        }
        else
        {
            // if block left
            shared[sy][sx - HALO] = global[gy * nx + gx - HALO];
        }
    }
    // RIGHT
    if ((tx > (BDIMX - HALO)) || ((gx + HALO) > nx))
    {
        if ((gx + HALO) > nx)
        {
            // if global right
            shared[sy][sx + HALO] = 0.0;
        }
        else
        {
            // if block right
            shared[sy][sx + HALO] = global[gy * nx + gx + HALO];
        }
    }
    
    // BOTTOM
    if (ty < HALO)
    {
        if (gy < HALO)
        {
            // if global bottom
            shared[sy - HALO][sx] = 0.0;
        }
        else
        {
            // if block bottom
            shared[sy - HALO][sx] = global[(gy - HALO) * nx + gx];
        }
    }
    
    // TOP
    if ((ty > (BDIMY - HALO)) || ((gy + HALO) > ny))
    {
        if ((gy + HALO) > ny)
        {
            // if global top
            shared[sy + HALO][sx] = 0.0;
        }
        else
        {
            // if block top
            shared[sy + HALO][sx] = global[(gy + HALO) * nx + gx];
        }
    }
}

// FD kernel
__global__ void kernel_2dfd(float *d_u1, float *d_u2, float *d_vp)
{
    const int nx = c_nx;
    const int ny = c_ny;

    // thread addres (ty, tx) within a block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // thread address (gy, gx) in global memory space
    const unsigned int gx = blockIdx.x * blockDim.x + tx;
    const unsigned int gy = blockIdx.y * blockDim.y + ty;

    // Allocate shared memory(smem)
    __shared__ float s_u1[BDIMX + HALO2][BDIMX + HALO2];
    __shared__ float s_u2[BDIMX + HALO2][BDIMX + HALO2];
    __shared__ float s_vp[BDIMX + HALO2][BDIMX + HALO2];


    // if thread points into the model
    if ((tx < nx) && (ty < ny))
    {
        set_halo(d_u1, s_u1, tx, ty, gx, gy, nx, ny);
        set_halo(d_u2, s_u1, tx, ty, gx, gy, nx, ny);
        set_halo(d_vp, s_vp, tx, ty, gx, gy, nx, ny);
    }


    __syncthreads();
}

int main(int argc, char *argv[])
{
    // Print out name of the main GPU
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("%s\t%d.%d:\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    printf("%lu GB:\t total Global memory (gmem)\n", deviceProp.totalGlobalMem / 1024 / 1024 / 1000);
    printf("%lu MB:\t total Constant memory (cmem)\n", deviceProp.totalConstMem / 1024);
    printf("%lu MB:\t total Shared memory per block (smem)\n", deviceProp.sharedMemPerBlock / 1024);
    printf("%d:\t total threads per block\n", deviceProp.maxThreadsPerBlock);
    printf("%d:\t total registers per block\n", deviceProp.regsPerBlock);
    printf("%d:\t warp size\n", deviceProp.warpSize);
    printf("%d x %d x %d:\t max dims of block\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("%d x %d x %d:\t max dims of grid\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    CHECK(cudaSetDevice(0));

    // Model dimensions
    int nx = 1024; /* x dim */
    int ny = 1024; /* z dim */

    size_t nxy = nx * ny;
    size_t nbytes = nxy * sizeof(float); /* bytes to store nx * ny */

    float dx = 1; /* m */
    float dy = dx;

    // Allocate memory for velocity model
    float _vp = 3300; /* m/s, p-wave velocity */
    float *h_vp;
    h_vp = (float *)malloc(nbytes);
    memset(h_vp, _vp, nbytes); /* initiate h_vp with _vp */

    // Time stepping
    float t_total = 0.05;                /* sec, total time of wave propagation */
    float dt = 0.7 * fmin(dx, dy) / _vp; /* sec, time step assuming constant vp */
    int nt = round(t_total / dt);        /* number of time steps */

    // Source
    float f0 = 100.0;    /* Hz, source dominant frequency */
    float t0 = 1.2 / f0; /* source HALOding to move wavelet from left of zero */

    float *h_wavelet, *h_time;
    h_time = (float *)malloc(nt * sizeof(float));
    h_wavelet = (float *)malloc(nt * sizeof(float));

    // Fill source waveform vecror
    float a = PI * PI * f0 * f0; /* const for wavelet */
    for (size_t it = 0; it < nt; it++)
    {
        h_time[it] = it * dt;
        h_wavelet[it] = 1e10 * (1.0 - 2.0 * a * pow(h_time[it] - t0, 2)) * exp(-a * pow(h_time[it] - t0, 2));
        h_wavelet[it] *= dt * dt / (dx * dy);
    }

    // Allocate memory on device
    printf("Allocate and copy memory on device...");
    float *d_u1, *d_u2, *d_vp, *d_wavelet;
    CHECK(cudaMalloc((void **)&d_u1, nbytes))       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_u2, nbytes))       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_vp, nbytes))       /* velocity model */
    CHECK(cudaMalloc((void **)&d_wavelet, nbytes)); /* source term for each time step */

    CHECK(cudaMemset(d_u1, 0, nbytes))
    CHECK(cudaMemset(d_u2, 0, nbytes))
    CHECK(cudaMemcpy(d_vp, h_vp, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wavelet, h_wavelet, nbytes, cudaMemcpyHostToDevice));

    float coef[] = {a0, a1, a2, a3, a4};
    int isrc = round((float)nx / 2); /* source location, ox */
    int jsrc = round((float)ny / 2); /* source location, oz */

    CHECK(cudaMemcpyToSymbol(c_coef, coef, 5 * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_isrc, &isrc, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_jsrc, &jsrc, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nx, &nx, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_ny, &ny, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nt, &nt, sizeof(int)));
    printf("OK\n");

    // Setup kernel run
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // MAIN LOOP 
    for (int it = 0; it < nt; it++)
    {
        printf("Step %i/%i\n", it+1, nt);
        kernel_add_wavelet<<<grid, block>>>(d_u1, d_wavelet, it);
        kernel_2dfd<<<grid, block>>>(d_u1, d_u2, d_vp);

        if (it == 100)
        {
            saveSnapshotIstep(it, d_u2, ny, nx);
        }
    }

    printf("Clean memory...");
    free(h_vp);
    free(h_time);
    free(h_wavelet);

    CHECK(cudaFree(d_u1));
    CHECK(cudaFree(d_u2));
    CHECK(cudaFree(d_vp));
    CHECK(cudaFree(d_wavelet));
    printf("OK\n");

    CHECK(cudaDeviceReset());

    return 0;
}
