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

#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

#include "indices.h"
#include "wtime.h"
#include "solver.h"

/* macros */
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef N_VALUE
#define N_VALUE 128
#endif

#define IX(x,y) (((x) % 2) ^ ((y) % 2)) * (N_VALUE+2) * ((N_VALUE+2) / 2) + ((x) / 2) + (y) * ((N_VALUE+2) / 2)


/* global variables */

static int N = N_VALUE;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

static float *u_d, *v_d, *u_prev_d, *v_prev_d;
static float *dens_d, *dens_prev_d;


static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;

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

    return (err);
}


/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display(void)
{
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void post_display(void)
{
    glutSwapBuffers();
}

static void draw_velocity(void)
{
    int i, j;
    float x, y, h;

    h = 1.0f / N;

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.0f);

    glBegin(GL_LINES);

    for (i = 1; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 1; j <= N; j++) {
            y = (j - 0.5f) * h;

            glVertex2f(x, y);
            glVertex2f(x + u[IX(i, j)], y + v[IX(i, j)]);
        }
    }

    glEnd();
}

static void draw_density(void)
{
    int i, j;
    float x, y, h, d00, d01, d10, d11;

    h = 1.0f / N;

    glBegin(GL_QUADS);

    for (i = 0; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 0; j <= N; j++) {
            y = (j - 0.5f) * h;

            d00 = dens[IX(i, j)];
            d01 = dens[IX(i, j + 1)];
            d10 = dens[IX(i + 1, j)];
            d11 = dens[IX(i + 1, j + 1)];

            glColor3f(d00, d00, d00);
            glVertex2f(x, y);
            glColor3f(d10, d10, d10);
            glVertex2f(x + h, y);
            glColor3f(d11, d11, d11);
            glVertex2f(x + h, y + h);
            glColor3f(d01, d01, d01);
            glVertex2f(x, y + h);
        }
    }

    glEnd();
}

/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

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
    int i, j, size = (N + 2) * (N + 2);
    
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

    if (!mouse_down[0] && !mouse_down[2]) {
        return;
    }

    i = (int)((mx / (float)win_x) * N + 1);
    j = (int)(((win_y - my) / (float)win_y) * N + 1);

    if (i < 1 || i > N || j < 1 || j > N) {
        return;
    }

    if (mouse_down[0]) {
        float val = force * (mx - omx);
        cudaMemcpy(&u[IX(i, j)], &val, sizeof(float), cudaMemcpyHostToDevice);
        val = force * (omy - my);
        cudaMemcpy(&v[IX(i, j)], &val, sizeof(float), cudaMemcpyHostToDevice);
    }

    if (mouse_down[2]) {
        cudaMemcpy(&d[IX(i, j)], &source, sizeof(float), cudaMemcpyHostToDevice);
    }

    omx = mx;
    omy = my;

    return;
}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func(unsigned char key, int x, int y)
{
    switch (key) {
    case 'c':
    case 'C':
        clear_data();
        break;

    case 'q':
    case 'Q':
        free_data();
        exit(0);
        break;

    case 'v':
    case 'V':
        dvel = !dvel;
        break;
    }
}

static void mouse_func(int button, int state, int x, int y)
{
    omx = mx = x;
    omx = my = y;

    mouse_down[button] = state == GLUT_DOWN;
}

static void motion_func(int x, int y)
{
    mx = x;
    my = y;
}

static void reshape_func(int width, int height)
{
    glutSetWindow(win_id);
    glutReshapeWindow(width, height);

    win_x = width;
    win_y = height;
}

static void idle_func(void)
{
    static int times = 1;
    static double start_t = 0.0;
    static double one_second = 0.0;
    static double react_ns_p_cell = 0.0;
    static double vel_ns_p_cell = 0.0;
    static double dens_ns_p_cell = 0.0;
    int size = (N + 2) * (N + 2);


    start_t = wtime();
    react(dens_prev_d, u_prev_d, v_prev_d);
    react_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    vel_step(N, u_d, v_d, u_prev_d, v_prev_d, visc, dt);
    vel_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    dens_step(N, dens_d, dens_prev_d, u_d, v_d, diff, dt);
    dens_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

    cudaDeviceSynchronize();

     // device -> host
    cudaMemcpy(u, u_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, v_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens, dens_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_prev, u_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_prev, v_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens_prev, dens_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    if (1.0 < wtime() - one_second) { /* at least 1s between stats */
        printf("%lf, %lf, %lf, %lf: ns per cell total, react, vel_step, dens_step\n",
               (react_ns_p_cell + vel_ns_p_cell + dens_ns_p_cell) / times,
               react_ns_p_cell / times, vel_ns_p_cell / times, dens_ns_p_cell / times);
        one_second = wtime();
        react_ns_p_cell = 0.0;
        vel_ns_p_cell = 0.0;
        dens_ns_p_cell = 0.0;
        times = 1;
    } else {
        times++;
    }

    glutSetWindow(win_id);
    glutPostRedisplay();
}

static void display_func(void)
{
    pre_display();

    if (dvel) {
        draw_velocity();
    } else {
        draw_density();
    }

    post_display();
}


/*
  ----------------------------------------------------------------------
   open_glut_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glut_window(void)
{
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    win_id = glutCreateWindow("Alias | wavefront");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();

    pre_display();

    glutKeyboardFunc(key_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char** argv)
{
    glutInit(&argc, argv);

    if (argc != 1 && argc != 7) {
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
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        diff = atof(argv[3]);
        visc = atof(argv[4]);
        force = atof(argv[5]);
        source = atof(argv[6]);
    }

    printf("\n\nHow to use this demo:\n\n");
    printf("\t Add densities with the right mouse button\n");
    printf("\t Add velocities with the left mouse button and dragging the mouse\n");
    printf("\t Toggle density/velocity display with the 'v' key\n");
    printf("\t Clear the simulation by pressing the 'c' key\n");
    printf("\t Quit by pressing the 'q' key\n");

    dvel = 0;

    if (!allocate_data()) {
        exit(1);
    }
    clear_data();

    win_x = 512;
    win_y = 512;
    open_glut_window();

    glutMainLoop();

    exit(0);
}
