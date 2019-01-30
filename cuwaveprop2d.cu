/*
Hello world of wave propagation in CUDA. FDTD acoustic wave propagation in homogeneous medium. Second order accurate in time and eigth in space.

Oleg Ovcharenko
Vladimir Kazei, 2019

oleg.ovcharenko@kaust.edu.sa
vladimir.kazei@kaust.edu.sa
*/

#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
/*
Add this to c_cpp_properties.json if linting isn't working for CUDA libraries
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

// Padding for FD scheme
#define HALO 4
#define HALO2 8

// FD stencil coefficients
#define a0  -2.8472222f
#define a1   1.6000000f
#define a2  -0.2000000f
#define a3   0.0253968f
#define a4  -0.0017857f

// Block dimensions
#define BDIMX 32
#define BDIMY 32

// Shared memory tile dimenstions
#define SDIMX BDIMX + HALO2
#define SDIMY BDIMY + HALO2

// Constant device memory
__constant__ float c_coef[5]; /* coefficients for 8th order fd */
__constant__ int c_isrc;      /* source location, ox */
__constant__ int c_jsrc;      /* source location, oz */
__constant__ int c_nx;        /* x dim */
__constant__ int c_ny;        /* y dim */
__constant__ int c_nt;        /* time steps */
__constant__ float c_dt2dx2;  /* dt2 / dx2 for fd*/

// Save snapshot as a binary, filename snap/snap_tag_it_ny_nx
void saveSnapshotIstep(int it, float *data, int nx, int ny, const char *tag)
{
    /*
    it      :timestep id
    data    :pointer to an array in device memory
    nx, ny  :model dimensions
    tag     :user-defined file identifier
    */

    // Array to store wavefield
    unsigned int isize = nx * ny * sizeof(float);
    float *iwave = (float *)malloc(isize);
    CHECK(cudaMemcpy(iwave, data, isize, cudaMemcpyDeviceToHost));

    char fname[32];
    sprintf(fname, "snap/snap_%s_%i_%i_%i", tag, it, ny, nx);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    printf("\tSave...%s: nx = %i ny = %i it = %i tag = %s\n", fname, nx, ny, it, tag);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}

// Add source wavelet
__global__ void kernel_add_wavelet(float *d_u, float *d_wavelet, int it)
{
    /*
    d_u             :pointer to an array on device where to add source term
    d_wavelet       :pointer to an array on device with source signature
    it              :time step id
    */
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gy * c_nx + gx;

    if ((gx == c_isrc) && (gy == c_jsrc))
    {
        d_u[idx] += d_wavelet[it];
    }
}

__device__ void set_halo(float *global, float shared[][SDIMX], int tx, int ty, int sx, int sy, int gx, int gy, int nx, int ny)
{
    /*
    global      :pointer to an array in global memory (gmem)
    shared      :2D array in shared device memory
    tx, ty      :thread id's in a block
    sx, sy      :thread id's in a shared memory tile
    gx, gy      :thread id's in the entire computational domain
    */

    // Each thread copies one value from gmem into smem
    shared[sy][sx] = global[gy * nx + gx];

    // Populate halo regions in smem for left, right, top and bottom boundaries of a block
    // if thread near LEFT border of a block
    if (tx < HALO)
    {
        // if global left
        if (gx < HALO)
        {
            // reflective boundary
            shared[sy][sx - HALO] = 0.0;
        }
        else
        {
            // if block left
            shared[sy][sx - HALO] = global[gy * nx + gx - HALO];
        }
    }
    // if thread near RIGHT border of a block
    if ((tx >= (BDIMX - HALO)) || ((gx + HALO) >= nx))
    {
        // if global right
        if ((gx + HALO) >= nx)
        {
            // reflective boundary
            shared[sy][sx + HALO] = 0.0;
        }
        else
        {
            // if block right
            shared[sy][sx + HALO] = global[gy * nx + gx + HALO];
        }
    }

    // if thread near BOTTOM border of a block
    if (ty < HALO)
    {
        // if global bottom
        if (gy < HALO)
        {
            // reflective boundary
            shared[sy - HALO][sx] = 0.0;
        }
        else
        {
            // if block bottom
            shared[sy - HALO][sx] = global[(gy - HALO) * nx + gx];
        }
    }

    // if thread near TOP border of a block    
    if ((ty >= (BDIMY - HALO)) || ((gy + HALO) >= ny))
    {
        // if global top
        if ((gy + HALO) >= ny)
        {
            // reflective boundary
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
    // save model dims in registers as they are much faster
    const int nx = c_nx;
    const int ny = c_ny;

    // FD coefficient dt2 / dx2
    const float dt2dx2 = c_dt2dx2;

    // Thread address (ty, tx) in a block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // Thread address (sy, sx) in shared memory
    const unsigned int sx = threadIdx.x + HALO;
    const unsigned int sy = threadIdx.y + HALO;

    // Thread address (gy, gx) in global memory
    const unsigned int gx = blockIdx.x * blockDim.x + tx;
    const unsigned int gy = blockIdx.y * blockDim.y + ty;

    // Global linear index
    const unsigned int idx = gy * nx + gx;

    // Allocate shared memory for a block (smem)
    __shared__ float s_u1[SDIMY][SDIMX];
    __shared__ float s_u2[SDIMY][SDIMX];
    __shared__ float s_vp[SDIMY][SDIMX];

    // If thread points into the physical domain
    if ((gx < nx) && (gy < ny))
    {
        // Copy regions from gmem into smem
        //       gmem, smem,  block, shared, global, dims
        set_halo(d_u1, s_u1, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_u2, s_u2, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_vp, s_vp, tx, ty, sx, sy, gx, gy, nx, ny);
        __syncthreads();

        // Central point of fd stencil, o o o o x o o o o
        float du2_xx = c_coef[0] * s_u2[sy][sx];
        float du2_yy = c_coef[0] * s_u2[sy][sx];

#pragma unroll
        for (int d = 1; d <= 4; d++)
        {
            du2_xx += c_coef[d] * (s_u2[sy][sx - d] + s_u2[sy][sx + d]);
            du2_yy += c_coef[d] * (s_u2[sy - d][sx] + s_u2[sy + d][sx]);
        }
        // Second order wave equation
        d_u1[idx] = 2.0 * s_u2[sy][sx] - s_u1[sy][sx] + s_vp[sy][sx] * s_vp[sy][sx] * (du2_xx + du2_yy) * dt2dx2;

        __syncthreads();
    }
}



/*
===================================================================================
MAIN
===================================================================================
*/
int main(int argc, char *argv[])
{
    // Model dimensions
    int nx = 512;                      /* x dim */
    int ny = 512;                      /* z dim */

    size_t nxy = nx * ny;
    size_t nbytes = nxy * sizeof(float);/* bytes to store nx * ny */

    float dx = 10.0;                    /* grid step, assume dy = dx, m */

    // Allocate memory for velocity model
    float _vp = 3300.0;                 /* p-wave velocity, m/s */
    float *h_vp;
    h_vp = (float *)malloc(nbytes);
    for (int i = 0; i < nxy; i++)
    {
        h_vp[i] = _vp;                  /* assume homogeneous velocity model */
    }

    printf("MODEL:\n");
    printf("\t%i x %i\t:ny x nx\n", ny, nx);
    printf("\t%f\t:dx\n", dx);
    printf("\t%f\t:h_vp[0]\n", h_vp[0]);

    // Time stepping
    float t_total = 1.5;               /* total time of wave propagation, sec */
    float dt = 0.5 * dx / _vp;          /* time step assuming constant vp, sec */
    int nt = round(t_total / dt);       /* number of time steps */
    int snap_step = round(0.1 * nt);   /* save snapshot every ... steps */

    printf("TIME STEPPING:\n");
    printf("\t%e\t:t_total\n", t_total);
    printf("\t%e\t:dt\n", dt);
    printf("\t%i\t:nt\n", nt);

    // Source
    float f0 = 10.0;                    /* source dominant frequency, Hz */
    float t0 = 1.2 / f0;                /* source padding to move wavelet from left of zero */
    int isrc = round((float)nx / 2);    /* source location, ox */
    int jsrc = round((float)ny / 2);    /* source location, oz */

    float *h_wavelet, *h_time;
    float tbytes = nt * sizeof(float);
    h_time = (float *)malloc(tbytes);
    h_wavelet = (float *)malloc(tbytes);

    // Fill source waveform vecror
    float a = PI * PI * f0 * f0;            /* const for wavelet */
    float dt2dx2 = (dt * dt) / (dx * dx);   /* const for fd stencil */
    for (int it = 0; it < nt; it++)
    {
        h_time[it] = it * dt;
        // Ricker wavelet (Mexican hat), second derivative of Gaussian
        h_wavelet[it] = 1e10 * (1.0 - 2.0 * a * pow(h_time[it] - t0, 2)) * exp(-a * pow(h_time[it] - t0, 2));
        h_wavelet[it] *= dt2dx2;
    }

    printf("SOURCE:\n");
    printf("\t%f\t:f0\n", f0);
    printf("\t%f\t:t0\n", t0);
    printf("\t%i\t:isrc - ox\n", isrc);
    printf("\t%i\t:jsrc - oy\n", jsrc);
    printf("\t%e\t:dt2dx2\n", dt2dx2);
    printf("\t%f\t:min wavelength [m]\n",(float)_vp / (2*f0));
    printf("\t%f\t:ppw\n",(float)_vp / (2*f0) / dx);

    // Allocate memory on device
    printf("Allocate and copy memory on the device...\n");
    float *d_u1, *d_u2, *d_vp, *d_wavelet;
    CHECK(cudaMalloc((void **)&d_u1, nbytes))       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_u2, nbytes))       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_vp, nbytes))       /* velocity model */
    CHECK(cudaMalloc((void **)&d_wavelet, tbytes)); /* source term for each time step */
    // Fill allocated memory with a value
    CHECK(cudaMemset(d_u1, 0, nbytes))
    CHECK(cudaMemset(d_u2, 0, nbytes))
    // Copy arrays from host to device
    CHECK(cudaMemcpy(d_vp, h_vp, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wavelet, h_wavelet, tbytes, cudaMemcpyHostToDevice));

    // Copy constants to device constant memory
    float coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(c_coef, coef, 5 * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_isrc, &isrc, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_jsrc, &jsrc, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nx, &nx, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_ny, &ny, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nt, &nt, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_dt2dx2, &dt2dx2, sizeof(float)));
    printf("\t%f MB\n", (4 * nbytes + tbytes)/1024/1024);
    printf("OK\n");

    // Print out specs of the main GPU
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("GPU0:\t%s\t%d.%d:\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    printf("\t%lu GB:\t total Global memory (gmem)\n", deviceProp.totalGlobalMem / 1024 / 1024 / 1000);
    printf("\t%lu MB:\t total Constant memory (cmem)\n", deviceProp.totalConstMem / 1024);
    printf("\t%lu MB:\t total Shared memory per block (smem)\n", deviceProp.sharedMemPerBlock / 1024);
    printf("\t%d:\t total threads per block\n", deviceProp.maxThreadsPerBlock);
    printf("\t%d:\t total registers per block\n", deviceProp.regsPerBlock);
    printf("\t%d:\t warp size\n", deviceProp.warpSize);
    printf("\t%d x %d x %d:\t max dims of block\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("\t%d x %d x %d:\t max dims of grid\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    CHECK(cudaSetDevice(0));

    // Print out CUDA domain partitioning info
    printf("CUDA:\n");
    printf("\t%i x %i\t:block dim\n", BDIMY, BDIMX);
    printf("\t%i x %i\t:shared dim\n", SDIMY, SDIMX);
    printf("CFL:\n");
    printf("\t%f\n", _vp * dt / dx);

    // Setup CUDA run
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    

    // MAIN LOOP
    printf("Time loop...\n");
    for (int it = 0; it < nt; it++)
    {
        // These kernels are in the same stream so they will be executed one by one
        kernel_add_wavelet<<<grid, block>>>(d_u2, d_wavelet, it);
        kernel_2dfd<<<grid, block>>>(d_u1, d_u2, d_vp);
        CHECK(cudaDeviceSynchronize());

        // Exchange time steps
        float *d_u3 = d_u1;
        d_u1 = d_u2;
        d_u2 = d_u3;

        // Save snapshot every snap_step iterations
        if ((it % snap_step == 0))
        {   
            printf("%i/%i\n", it+1, nt);
            saveSnapshotIstep(it, d_u3, nx, ny,"u3");
        }
    }
    printf("OK\n");

    CHECK(cudaGetLastError());

    printf("Clean memory...");
    delete[] h_vp;
    delete[] h_time;
    delete[] h_wavelet;

    CHECK(cudaFree(d_u1));
    CHECK(cudaFree(d_u2));
    CHECK(cudaFree(d_vp));
    CHECK(cudaFree(d_wavelet));
    printf("OK\n");

    CHECK(cudaDeviceReset());

    return 0;
}
