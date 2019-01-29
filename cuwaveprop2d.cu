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

#define a0  -2.8472222f
#define a1   1.6000000f
#define a2  -0.2000000f
#define a3   0.0253968f
#define a4  -0.0017857f

#define BDIMX 32
#define BDIMY 32

#define SDIMX BDIMX + HALO2
#define SDIMY BDIMY + HALO2

// Allocate the constant device memory
__constant__ float c_coef[5]; /* coefficients for 8th order fd */
__constant__ int c_isrc;      /* source location, ox */
__constant__ int c_jsrc;      /* source location, oz */
__constant__ int c_nx;        /* x dim */
__constant__ int c_ny;        /* y dim */
__constant__ int c_nt;        /* time steps */
__constant__ int c_dt2dx2;    /* dt2 / dx2 for fd*/

// Save snapshot as a binary
void saveSnapshotIstep(int istep, float *data, int nx, int ny, const char *tag)
{
    // Array to store wavefield
    unsigned int isize = nx * ny * sizeof(float);
    float *iwave = (float *)malloc(isize);
    CHECK(cudaMemcpy(iwave, data, isize, cudaMemcpyDeviceToHost));

    char fname[32];
    sprintf(fname, "snap/snap_%s_%i_%i_%i", tag, istep, ny, nx);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    printf("%s: nx = %i ny = %i istep = %i tag = %s\n", fname, nx, ny, istep, tag);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}

// Add source wavelet
__global__ void kernel_add_wavelet(float *d_u, float *d_wavelet, int it)
{
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
    if ((tx >= (BDIMX - HALO)) || ((gx + HALO) >= nx))
    {
        if ((gx + HALO) >= nx)
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
    if ((ty >= (BDIMY - HALO)) || ((gy + HALO) >= ny))
    {
        if ((gy + HALO) >= ny)
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
__global__ void kernel_2dfd(float *d_u1, float *d_u2, float *d_vp, float dt2dx2)
{
    // save model dims in registers as they are much faster
    const int nx = c_nx;
    const int ny = c_ny;

    // fd coef dt2 / dx2
    // const float dt2dx2 = c_dt2dx2;

    // thread addres (ty, tx) in a block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // thread addres (sy, sx) in shared memory
    const unsigned int sx = threadIdx.x + HALO;
    const unsigned int sy = threadIdx.y + HALO;

    // thread address (gy, gx) in global memory
    const unsigned int gx = blockIdx.x * blockDim.x + tx;
    const unsigned int gy = blockIdx.y * blockDim.y + ty;

    // global linear index
    const unsigned int idx = gy * nx + gx;

    // Allocate shared memory(smem)
    __shared__ float s_u1[SDIMY][SDIMX];
    __shared__ float s_u2[SDIMY][SDIMX];
    __shared__ float s_vp[SDIMY][SDIMX];

    // if thread points into the model
    if ((gx < nx) && (gy < ny))
    {
        // Copy regions from gmem into smem
        //       gmem, smem,  block, shared, global, dims
        set_halo(d_u1, s_u1, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_u2, s_u2, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_vp, s_vp, tx, ty, sx, sy, gx, gy, nx, ny);
        __syncthreads();

        float du2_xx = c_coef[0] * s_u2[sy][sx];
        float du2_yy = c_coef[0] * s_u2[sy][sx];

#pragma unroll
        for (int d = 1; d <= 4; d++)
        {
            du2_xx += c_coef[d] * (s_u2[sy][sx - d] + s_u2[sy][sx + d]);
            du2_yy += c_coef[d] * (s_u2[sy - d][sx] + s_u2[sy + d][sx]);
        }

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
    int nx = 1024; /* x dim */
    int ny = 1024; /* z dim */

    size_t nxy = nx * ny;
    size_t nbytes = nxy * sizeof(float); /* bytes to store nx * ny */

    float dx = 10.0; /* m */

    // Allocate memory for velocity model
    float _vp = 3300.0; /* m/s, p-wave velocity */
    float *h_vp;
    h_vp = (float *)malloc(nbytes);
    for (int i = 0; i < nxy; i++)
    {
        h_vp[i] = _vp;
    }

    printf("MODEL:\n");
    printf("\t%i x %i\t:ny x nx\n", ny, nx);
    printf("\t%f\t:dx\n", dx);
    printf("\t%f\t:h_vp[0]\n", h_vp[0]);

    // Time stepping
    float t_total = 1.55;         /* sec, total time of wave propagation */
    float dt = 0.5 * dx / _vp;    /* sec, time step assuming constant vp */
    int nt = round(t_total / dt); /* number of time steps */
    int snap_step = round(0.05 * nt);

    printf("TIME STEPPING:\n");
    printf("\t%e\t:t_total\n", t_total);
    printf("\t%e\t:dt\n", dt);
    printf("\t%i\t:nt\n", nt);

    // Source
    float f0 = 10.0;                 /* Hz, source dominant frequency */
    float t0 = 1.2 / f0;             /* source HALOding to move wavelet from left of zero */
    int isrc = round((float)nx / 2); /* source location, ox */
    int jsrc = round((float)ny / 2); /* source location, oz */

    float *h_wavelet, *h_time;
    float tbytes = nt * sizeof(float);
    h_time = (float *)malloc(tbytes);
    h_wavelet = (float *)malloc(tbytes);

    // Fill source waveform vecror
    float a = PI * PI * f0 * f0;        /* const for wavelet */
    float dt2dx2 = (dt * dt) / (dx * dx); /* const for fd stencil */
    for (size_t it = 0; it < nt; it++)
    {
        h_time[it] = it * dt;
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

    CHECK(cudaMemset(d_u1, 0, nbytes))
    CHECK(cudaMemset(d_u2, 0, nbytes))
    CHECK(cudaMemcpy(d_vp, h_vp, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wavelet, h_wavelet, tbytes, cudaMemcpyHostToDevice));

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

    printf("CUDA:\n");
    printf("\t%i x %i\t:block dim\n", BDIMY, BDIMX);
    printf("\t%i x %i\t:shared dim\n", SDIMY, SDIMX);
    printf("CFL:\n");
    printf("\t%f\n", _vp * dt / dx);

    // Setup kernel run
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    // MAIN LOOP
    printf("Time loop...\n");
    for (int it = 0; it < nt; it++)
    {
        // These kernels are in the same stream so they will be executes successively
        kernel_add_wavelet<<<grid, block>>>(d_u2, d_wavelet, it);
        kernel_2dfd<<<grid, block>>>(d_u1, d_u2, d_vp, dt2dx2);
        CHECK(cudaDeviceSynchronize());

        // Exchange time steps
        float *d_u3 = d_u1;
        d_u1 = d_u2;
        d_u2 = d_u3;

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
