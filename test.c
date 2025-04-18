#include <stdlib.h>

#define N 512

#define IX(i, j) ((i) + (n + 2) * (j))

typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;



static void set_bnd(unsigned int n, boundary b, float* x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void lin_solve(unsigned int n, boundary b, float* x, const float* x0, float a, float c)
{
    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int i = 1; i <= n; i++) {
            for (unsigned int j = 1; j <= n; j++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
            }
        }
        set_bnd(n, b, x);
    }
}

int main() {
    static float *x, *x0;

    int size = (N + 2) * (N + 2);

    x = (float*)malloc(size * sizeof(float));
    x0 = (float*)malloc(size * sizeof(float));

    for (unsigned int i = 1; i < size - 1; ++i) {
        x[i] = (float)i;
        x0[i] = (float)i + 1.0;
    }

    lin_solve(N, NONE, x, x0, 3.0, 2.0);

    return (int)(x[3] + x0[10]);
}
