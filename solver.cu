#include <stddef.h>
#include <stdio.h>

#include "solver.h"
#include "indices.h"

#define THREADS_PER_BLOCK 256

#define IX(x,y) ((x % 2) ^ (y % 2)) * (n+2) * ((n+2) / 2) + (x / 2) + y * ((n+2) / 2)
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

__global__ void add_source(unsigned int n, float * x, const float * s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        x[tid] += dt * s[tid];
    }
}

__global__ void set_bnd(unsigned int n, boundary b, float * x)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
        x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
        x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
        x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
    } else if (tid <= n) {
        x[IX(0, tid)]     = b == VERTICAL ? -x[IX(1, tid)] : x[IX(1, tid)];
        x[IX(n + 1, tid)] = b == VERTICAL ? -x[IX(n, tid)] : x[IX(n, tid)];
        x[IX(tid, 0)]     = b == HORIZONTAL ? -x[IX(tid, 1)] : x[IX(tid, 1)];
        x[IX(tid, n + 1)] = b == HORIZONTAL ? -x[IX(tid, n)] : x[IX(tid, n)];
    }

}


__global__ void lin_solve_red_step(
    unsigned int n,
    float a,
    float c,
    const float * __restrict__ same0,
    const float * __restrict__ neigh,
    float * __restrict__ same
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n * (n / 2)) {
        unsigned int y = (tid / (n / 2)) + 1;
        unsigned int x = tid % (n / 2);


        unsigned int width = (n + 2) / 2;

        int index = x + ((y + 1) % 2) + y * width;
        int shift = 1 - 2 * ((y + 1) % 2);
        same[index] = (same0[index] + a * (neigh[index - width] +
                                           neigh[index] +
                                           neigh[index + shift] +
                                           neigh[index + width])) / c;
    }
}

__global__ void lin_solve_black_step(
    unsigned int n,
    float a,
    float c,
    const float * __restrict__ same0,
    const float * __restrict__ neigh,
    float * __restrict__ same
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n * (n / 2)) {
        unsigned int y = (tid / (n / 2)) + 1;
        unsigned int x = tid % (n / 2);


        unsigned int width = (n + 2) / 2;

        int index = x + (y % 2) + y * width;
        int shift = 1 - 2 * (y % 2);
        same[index] = (same0[index] + a * (neigh[index - width] +
                                           neigh[index] +
                                           neigh[index + shift] +
                                           neigh[index + width])) / c;

    }
}

static void lin_solve(unsigned int n, boundary b,
                      float * __restrict__ x,
                      const float * __restrict__ x0,
                      float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    const float * red0 = x0;
    const float * blk0 = x0 + color_size;
    float * red = x;
    float * blk = x + color_size;

    int numBlocks = (color_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksSetBnd = 1;

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_red_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, a, c, red0, blk, red);
        cudaDeviceSynchronize();
        lin_solve_black_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, a, c, blk0, red, blk);
        cudaDeviceSynchronize();
        set_bnd<<<numBlocksSetBnd, THREADS_PER_BLOCK>>>(n, b, x);
        cudaDeviceSynchronize();
    }

}

static void diffuse(unsigned int n, boundary b, float * x, const float * x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

__global__ void advect_red_step(
        unsigned int n, 
        float * __restrict__ d, 
        const float * __restrict__ d0, 
        const float * __restrict__ u, 
        const float * __restrict__ v, 
        float dt
    )
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float dt0 = dt * n;

    if (tid < n * (n / 2)) {
        unsigned int j = (tid / (n / 2)) + 1;
        unsigned int i = tid % (n / 2);

        unsigned int index = i + ((j + 1) % 2) + j * ((n+2)/2);

        float x = 2*i + 1 + ((j + 1) % 2) - dt0 * u[index];
        float y = j - dt0 * v[index];
        if (x < 0.5f) {
            x = 0.5f;
        } else if (x > n + 0.5f) {
            x = n + 0.5f;
        }
        int i0 = (int) x;
        int i1 = i0 + 1;
        if (y < 0.5f) {
            y = 0.5f;
        } else if (y > n + 0.5f) {
            y = n + 0.5f;
        }
        int j0 = (int) y;
        int j1 = j0 + 1;
        float s1 = x - i0;
        float s0 = 1 - s1;
        float t1 = y - j0;
        float t0 = 1 - t1;
        d[index] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                        s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

__global__ void advect_black_step(
        unsigned int n, 
        float * __restrict__ d, 
        const float * __restrict__ d0, 
        const float * __restrict__ u, 
        const float * __restrict__ v, 
        float dt
    )
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float dt0 = dt * n;

    if (tid < n * (n / 2)) {
        unsigned int j = (tid / (n / 2)) + 1;
        unsigned int i = tid % (n / 2);

        unsigned int index = i + (j % 2) + j * ((n+2)/2);

        float x = 2*i + 1 + (j % 2) - dt0 * u[index];
        float y = j - dt0 * v[index];
        if (x < 0.5f) {
            x = 0.5f;
        } else if (x > n + 0.5f) {
            x = n + 0.5f;
        }
        int i0 = (int) x;
        int i1 = i0 + 1;
        if (y < 0.5f) {
            y = 0.5f;
        } else if (y > n + 0.5f) {
            y = n + 0.5f;
        }
        int j0 = (int) y;
        int j1 = j0 + 1;
        float s1 = x - i0;
        float s0 = 1 - s1;
        float t1 = y - j0;
        float t0 = 1 - t1;
        d[index] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                        s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
}


static void advect(
        unsigned int n, 
        boundary b, 
        float * __restrict__ d, 
        const float * __restrict__ d0, 
        const float * __restrict__ u, 
        const float * __restrict__ v, 
        float dt
    )
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;

    unsigned int width = (n+2)/2;

    const float * __restrict__ u_black = u + width * (n+2);
    const float * __restrict__ v_black = v + width * (n+2);
    float * __restrict__ d_black = d + width * (n+2);

    int numBlocks = (width * (n+2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksSetBnd = 1;


    advect_red_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, d, d0, u, v, dt);
    cudaDeviceSynchronize();
    advect_black_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, d_black, d0, u_black, v_black, dt);
    cudaDeviceSynchronize();
    set_bnd<<<numBlocksSetBnd, THREADS_PER_BLOCK>>>(n, b, d);
    cudaDeviceSynchronize();

}

__global__ void project_first_step(
        unsigned int n, 
        float * __restrict__ u,
        float * __restrict__ v,
        float * __restrict__ p, 
        float * __restrict__ div,
        int color_shift
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n * (n / 2)) {
        unsigned int y = (tid / (n / 2)) + 1;
        unsigned int x = tid % (n / 2);

        unsigned int width = (n + 2) / 2;
    
        int index = x + ((y + color_shift) % 2) + y * width;
        int shift = 1 - 2 * ((y + color_shift) % 2);
        
        div[index] = -0.5f * ((u[index] - u[index + shift]) * (-shift) +
                                v[index + width] - v[index - width]) / n;
        p[index] = 0;
    }

}

__global__ void project_second_step(
        unsigned int n, 
        float * __restrict__ u,
        float * __restrict__ v,
        float * __restrict__ p,
        int color_shift
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n * (n / 2)) {
        unsigned int y = (tid / (n / 2)) + 1;
        unsigned int x = tid % (n / 2);

        unsigned int width = (n + 2) / 2;
    
        int index = x + ((y + color_shift) % 2) + y * width;
        int shift = 1 - 2 * ((y + color_shift) % 2);

        u[index] -= 0.5f * n * (p[index] - p[index + shift]) * (-shift);
        v[index] -= 0.5f * n * (p[index + width] - p[index - width]);
    }
}

static void project(
        unsigned int n, 
        float * __restrict__ u, 
        float * __restrict__ v, 
        float * __restrict__ p, 
        float * __restrict__ div
    )
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    float * u_red = u;
    float * u_black = u + color_size;
    float * v_red = v;
    float * v_black = v + color_size;
    float * div_red = div;
    float * div_black = div + color_size;
    float * p_red = p;
    float * p_black = p + color_size;

    unsigned int width = (n + 2) / 2;

    int numBlocks = (color_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksSetBnd = 1;

    project_first_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, u_black, v_black, p_red, div_red, 1);
    project_first_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, u_red, v_red, p_black, div_black, 0);
    cudaDeviceSynchronize();

    set_bnd<<<numBlocksSetBnd, THREADS_PER_BLOCK>>>(n, NONE, div);
    set_bnd<<<numBlocksSetBnd, THREADS_PER_BLOCK>>>(n, NONE, p);
    cudaDeviceSynchronize();

    
    lin_solve(n, NONE, p, div, 1, 4);
    cudaDeviceSynchronize();

    project_second_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, u_red, v_red, p_black, 1);
    project_second_step<<<numBlocks, THREADS_PER_BLOCK>>>(n, u_black, v_black, p_red, 0);
    cudaDeviceSynchronize();

    set_bnd<<<numBlocksSetBnd, THREADS_PER_BLOCK>>>(n, VERTICAL, u);
    set_bnd<<<numBlocksSetBnd, THREADS_PER_BLOCK>>>(n, HORIZONTAL, v);
    cudaDeviceSynchronize();

}

void dens_step(unsigned int n, float *x, float *x0, float *u, float *v, float diff, float dt)
{
    int numBlocks = ((n+2)*(n+2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    add_source<<<numBlocks, THREADS_PER_BLOCK>>>(n, x, x0, dt);
    cudaDeviceSynchronize();

    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
    cudaDeviceSynchronize();
}

void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0, float visc, float dt)
{   
    int numBlocks = ((n+2)*(n+2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    add_source<<<numBlocks, THREADS_PER_BLOCK>>>(n, u, u0, dt);
    add_source<<<numBlocks, THREADS_PER_BLOCK>>>(n, v, v0, dt);
    cudaDeviceSynchronize();
    
    SWAP(u0, u);
    diffuse(n, VERTICAL, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, HORIZONTAL, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, VERTICAL, u, u0, u0, v0, dt);
    advect(n, HORIZONTAL, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
    cudaDeviceSynchronize();
}
