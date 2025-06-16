#include <stddef.h>
#include <stdio.h>

#include "solver.h"
#include "indices.h"

#define IX(x,y) (rb_idx((x),(y),(n+2)))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

static void add_source(unsigned int n, float * x, const float * s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)]     = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)]     = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
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

    float * red_d;
    cudaError_t err = cudaMalloc((void **)&red_d, color_size * sizeof(float));

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }


    float * blk_d;
    err = cudaMalloc((void **)&blk_d, color_size * sizeof(float));

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    float * red0_d;
    err = cudaMalloc((void **)&red0_d, color_size * sizeof(float));

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    float * blk0_d;
    err = cudaMalloc((void **)&blk0_d, color_size * sizeof(float));

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(red_d, red, color_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blk_d, blk, color_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(red0_d, red0, color_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blk0_d, blk0, color_size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (color_size + threadsPerBlock - 1) / threadsPerBlock;


    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_red_step<<<numBlocks, threadsPerBlock>>>(n, a, c, red0_d, blk_d, red_d);
        lin_solve_black_step<<<numBlocks, threadsPerBlock>>>(n, a, c, blk0_d, red_d, blk_d);
        cudaDeviceSynchronize();
        cudaMemcpy(red, red_d, color_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(blk, blk_d, color_size * sizeof(float), cudaMemcpyDeviceToHost);
        set_bnd(n, b, x);
    }


    cudaFree(red_d);
    cudaFree(blk_d);
    cudaFree(red0_d);
    cudaFree(blk0_d);
}

static void diffuse(unsigned int n, boundary b, float * x, const float * x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
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

    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 0; i < n/2; i++) {

            unsigned int index = idx(i + ((j + 1) % 2), j ,width);

            x = 2*i + 1 + ((j + 1) % 2) - dt0 * u[index];
            y = j - dt0 * v[index];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[index] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }

    const float * __restrict__ u_black = u + width * (n+2);
    const float * __restrict__ v_black = v + width * (n+2);
    float * __restrict__ d_black = d + width * (n+2);

    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 0; i < n/2; i++) {

            unsigned int index = idx(i + (j % 2), j ,width);

            x = 2*i + 1 + (j % 2) - dt0 * u_black[index];
            y = j - dt0 * v_black[index];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d_black[index] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }

    set_bnd(n, b, d);
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

    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + ((y + 1) % 2), y, width);
            int shift = 1 - 2 * ((y + 1) % 2);

            div_red[index] = -0.5f * ((u_black[index] - u_black[index + shift]) * (-shift) +
                                       v_black[index + width] - v_black[index - width]) / n;
            p_red[index] = 0;
        }
    }

    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + (y % 2), y, width);
            int shift = 1 - 2 * (y % 2);

            div_black[index] = -0.5f * ((u_red[index] - u_red[index + shift]) * (-shift) +
                                         v_red[index + width] - v_red[index - width]) / n;
            p_black[index] = 0;
        }
    }


    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);



    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + ((y + 1) % 2), y, width);
            int shift = 1 - 2 * ((y + 1) % 2);
            u_red[index] -= 0.5f * n * (p_black[index] - p_black[index + shift]) * (-shift);
            v_red[index] -= 0.5f * n * (p_black[index + width] - p_black[index - width]);
        }
    }

    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + (y % 2), y, width);
            int shift = 1 - 2 * (y % 2);
            u_black[index] -= 0.5f * n * (p_red[index] - p_red[index + shift]) * (-shift);
            v_black[index] -= 0.5f * n * (p_red[index + width] - p_red[index - width]);
        }
    }

    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
}

extern "C" void dens_step(unsigned int n, float *x, float *x0, float *u, float *v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

extern "C" void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
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
}
