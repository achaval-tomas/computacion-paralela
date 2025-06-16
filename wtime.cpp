// wtime.c
#ifdef _WIN32
#include <windows.h>

double wtime(void)
{
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER now;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / frequency.QuadPart;
}

#else
#define _POSIX_C_SOURCE 199309L
#include <time.h>

double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#endif