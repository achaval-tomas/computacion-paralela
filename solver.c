#include <stddef.h>

#include "solver.h"
#include "indices.h"
#include <omp.h>

#define IX(x,y) (rb_idx((x),(y),(n+2)))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

static void add_source(unsigned int n, float * x, const float * s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    #pragma omp parallel for
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

static void lin_solve_rb_step(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float * restrict same0,
                              const float * restrict neigh,
                              float * restrict same)
{
    unsigned int width = (n + 2) / 2;

    #pragma omp parallel for
    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + ((y + 1 + (color == BLACK)) % 2), y, width);
            int shift = 1 - 2 * ((y + 1 + (color == BLACK)) % 2);
            same[index] = (same0[index] + a * (neigh[index - width] +
                                               neigh[index] +
                                               neigh[index + shift] +
                                               neigh[index + width])) / c;
        }
    }
}


static void lin_solve(unsigned int n, boundary b,
                      float * restrict x,
                      const float * restrict x0,
                      float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    const float * red0 = x0;
    const float * blk0 = x0 + color_size;
    float * red = x;
    float * blk = x + color_size;

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step(RED,   n, a, c, red0, blk, red);
        lin_solve_rb_step(BLACK, n, a, c, blk0, red, blk);
        set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float * x, const float * x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect(unsigned int n, boundary b, float * restrict d, const float * restrict d0, const float * restrict u, const float * restrict v, float dt)
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;

    unsigned int width = (n+2)/2;

    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 0; i < n/2; i++) {

            unsigned int index = idx(i + ((j + 1) % 2), j, width);

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

    const float * restrict u_black = u + width * (n+2);
    const float * restrict v_black = v + width * (n+2);
    float * restrict d_black = d + width * (n+2);

    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 0; i < n/2; i++) {

            unsigned int index = idx(i + (j % 2), j, width);

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

static void project(unsigned int n, float * restrict u, float * restrict v, float * restrict p, float * restrict div)
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
    /*
	for (unsigned int j = 1; j <= n; j++) {
		for (unsigned int i = 1; i <= n; i++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                     v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
            p[IX(i, j)] = 0;
        }
    }
    */
    #pragma omp parallel for
    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + ((y + 1) % 2), y, width);
            int shift = 1 - 2 * ((y + 1) % 2);

            div_red[index] = -0.5f * ((u_black[index] - u_black[index + shift]) * (-shift) +
                                       v_black[index + width] - v_black[index - width]) / n;
            p_red[index] = 0;
        }
    }

    #pragma omp parallel for
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

    #pragma omp parallel for
    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + ((y + 1) % 2), y, width);
            int shift = 1 - 2 * ((y + 1) % 2);
            u_red[index] -= 0.5f * n * (p_black[index] - p_black[index + shift]) * (-shift);
            v_red[index] -= 0.5f * n * (p_black[index + width] - p_black[index - width]);
        }
    }

    #pragma omp parallel for
    for (unsigned int y = 1; y <= n; ++y) {
        for (unsigned int x = 0; x < n/2; ++x) {
            int index = idx(x + (y % 2), y, width);
            int shift = 1 - 2 * (y % 2);
            u_black[index] -= 0.5f * n * (p_red[index] - p_red[index + shift]) * (-shift);
            v_black[index] -= 0.5f * n * (p_red[index + width] - p_red[index - width]);
        }
    }


    /*
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
    */
    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
}

void dens_step(unsigned int n, float *x, float *x0, float *u, float *v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0, float visc, float dt)
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
