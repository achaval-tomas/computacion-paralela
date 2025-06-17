/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

        This code is a simple prototype that demonstrates how to use the
        code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
        for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <stdio.h>
#include <stdlib.h>

#include "indices.h"
#include "wtime.h"
#include "solver.h"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef N_VALUE
#define N_VALUE 128
#endif

/* macros */

#define IX(x,y) (((x) % 2) ^ ((y) % 2)) * (N_VALUE+2) * ((N_VALUE+2) / 2) + ((x) / 2) + (y) * ((N_VALUE+2) / 2)


/* global variables */

static int N = N_VALUE;
static float dt, diff, visc;
static float force, source;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

static float *u_d, *v_d, *u_prev_d, *v_prev_d;
static float *dens_d, *dens_prev_d;


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data(void)
{
    if (u) {
        free(u);
    }
    if (v) {
        free(v);
    }
    if (u_prev) {
        free(u_prev);
    }
    if (v_prev) {
        free(v_prev);
    }
    if (dens) {
        free(dens);
    }
    if (dens_prev) {
        free(dens_prev);
    }

    if (u_d) {
        cudaFree(u_d);
    }
    if (v_d) {
        cudaFree(v_d);
    }
    if (dens_d) {
        cudaFree(dens_d);
    }
    if (u_prev_d) {
        cudaFree(u_prev_d);
    }
    if (v_prev_d) {
        cudaFree(v_prev_d);
    }
    if (dens_prev_d) {
        cudaFree(dens_prev_d);
    }
}

static void clear_data(void)
{
    int i, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
    }

    cudaMemset(u_d, 0, size * sizeof(float));
    cudaMemset(v_d, 0, size * sizeof(float));
    cudaMemset(u_prev_d, 0, size * sizeof(float));
    cudaMemset(v_prev_d, 0, size * sizeof(float));
    cudaMemset(dens_d, 0, size * sizeof(float));
    cudaMemset(dens_prev_d, 0, size * sizeof(float));
}

static int tryCudaMalloc(float** a, int size)
{
    cudaError_t err = cudaMalloc((void**)a, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return (0);
    }
    return (1);
}

static int allocate_data(void)
{
    int size = (N + 2) * (N + 2);

    u = (float*)malloc(size * sizeof(float));
    v = (float*)malloc(size * sizeof(float));
    u_prev = (float*)malloc(size * sizeof(float));
    v_prev = (float*)malloc(size * sizeof(float));
    dens = (float*)malloc(size * sizeof(float));
    dens_prev = (float*)malloc(size * sizeof(float));

    if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev) {
        fprintf(stderr, "cannot allocate data\n");
        return (0);
    }

    int err = tryCudaMalloc(&u_d, size);
    err &= tryCudaMalloc(&v_d, size);
    err &= tryCudaMalloc(&u_prev_d, size);
    err &= tryCudaMalloc(&v_prev_d, size);
    err &= tryCudaMalloc(&dens_d, size);
    err &= tryCudaMalloc(&dens_prev_d, size);

    return err;
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void get_array_max(float *a, size_t size, float *result)
{
    __shared__ float partial_max;

    size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;	//global id
    size_t tid  = threadIdx.x;		// thread id, dentro del bloque
    size_t lid  = tid%warpSize;		// lane id, dentro del warp

    // Fase 1, inicialización
    if (tid==0)
    	partial_max = 0.0f;
    __syncthreads();

    // Fase 2, cómputo dentro del bloque
    float warp_reduce = a[gtid];

    // Fase 2.1, max en warp
    #define FULL_MASK 0xffffffff
    warp_reduce = max(__shfl_down_sync(FULL_MASK, warp_reduce, 16), warp_reduce);
    warp_reduce = max(__shfl_down_sync(FULL_MASK, warp_reduce, 8), warp_reduce);
    warp_reduce = max(__shfl_down_sync(FULL_MASK, warp_reduce, 4), warp_reduce);
    warp_reduce = max(__shfl_down_sync(FULL_MASK, warp_reduce, 2), warp_reduce);
    warp_reduce = max(__shfl_down_sync(FULL_MASK, warp_reduce, 1), warp_reduce);

    // Fase 2.2, acumulacion a shared
    if (lid==0) {
        atomicMax(&partial_max, warp_reduce);
    }
    __syncthreads();

    // Fase 3, acumulación del resultado local del bloque en la global
    if (tid==0) {
        atomicMax(result, partial_max);
    }
}

__global__ void calculate_velocity2(float* u, float*v, unsigned int size){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        u[tid] = u[tid]*u[tid] + v[tid]*v[tid];
    }
}

__global__ void set_react(
    float *u, float *v, float *d,
    float* max_velocity2, float* max_density,
    int N, float force, float source
){
    if (*max_velocity2 < 0.0000005f) {
        u[IX(N / 2, N / 2)] = force * 10.0f;
        v[IX(N / 2, N / 2)] = force * 10.0f;
    }
    if (*max_density < 1.0f) {
        d[IX(N / 2, N / 2)] = source * 10.0f;
    }
}

static void react(float* d, float* u, float* v)
{
    int size = (N + 2) * (N + 2);

    int numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    calculate_velocity2<<<numBlocks, THREADS_PER_BLOCK>>>(u, v, size);

    float *max_velocity2, *max_density;
    tryCudaMalloc(&max_velocity2, 1);
    tryCudaMalloc(&max_density, 1);

    cudaMemset(max_velocity2, 0, sizeof(float));
    cudaMemset(max_density, 0, sizeof(float));

    get_array_max<<<numBlocks, THREADS_PER_BLOCK>>>(d, size, max_density);
    get_array_max<<<numBlocks, THREADS_PER_BLOCK>>>(u, size, max_velocity2);

    cudaMemset(u, 0, size * sizeof(float));
    cudaMemset(v, 0, size * sizeof(float));
    cudaMemset(d, 0, size * sizeof(float));

    set_react<<<1, 1>>>(u, v, d, max_velocity2, max_density, N, force, source);

    cudaFree(max_velocity2); cudaFree(max_density);

    return;
}


static void one_step()
{
    int size = (N + 2) * (N + 2);

    react(dens_prev_d, u_prev_d, v_prev_d);
    vel_step(N, u_d, v_d, u_prev_d, v_prev_d, visc, dt);
    dens_step(N, dens_d, dens_prev_d, u_d, v_d, diff, dt);

    cudaDeviceSynchronize();

    // device -> host
    cudaMemcpy(u, u_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, v_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens, dens_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_prev, u_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_prev, v_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens_prev, dens_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char** argv)
{
    int i = 0;

    if (argc != 1 && argc != 6) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
        fprintf(stderr, "where:\n");
        fprintf(stderr, "\t N      : grid resolution\n");
        fprintf(stderr, "\t dt     : time step\n");
        fprintf(stderr, "\t diff   : diffusion rate of the density\n");
        fprintf(stderr, "\t visc   : viscosity of the fluid\n");
        fprintf(stderr, "\t force  : scales the mouse movement that generate a force\n");
        fprintf(stderr, "\t source : amount of density that will be deposited\n");
        exit(1);
    }

    if (argc == 1) {
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
                N, dt, diff, visc, force, source);
    } else {
        dt = atof(argv[1]);
        diff = atof(argv[2]);
        visc = atof(argv[3]);
        force = atof(argv[4]);
        source = atof(argv[5]);
    }

    if (!allocate_data()) {
        exit(1);
    }
    clear_data();

    double start_t_program = wtime();
    double start_t;
    double total = 0;
    i = 0;
    while (i < 2048) {
        i++;
        start_t = wtime();
        one_step();
        total += (double)(3 * N * N) / (1.0e6 * (wtime() - start_t));
        if (wtime() - start_t_program > 15)
            break;
    }

    printf("\ntotal_cells_per_us: %lf\n", total / (double)i);

    free_data();

    exit(0);
}
