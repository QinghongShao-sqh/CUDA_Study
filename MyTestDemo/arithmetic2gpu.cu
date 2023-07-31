#include <math.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include<stdint.h>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

const int NUM_REPEATS = 10;
const real x0 = 100.0;
void __global__ arithmetic(real* x, const real x0, const int N);

int main(void)
{
    const int N = 10000;
   // const int N = atoi(argv[1]);
    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    const int M = sizeof(real) * N;
    real* h_x = (real*)malloc(M);
    real* d_x;
    CHECK(cudaMalloc((void**)&d_x, M));

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        for (int n = 0; n < N; ++n)
        {
            h_x[n] = 0.0;
        }
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        arithmetic << <grid_size, block_size >> > (d_x, x0, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ arithmetic(real* d_x, const real x0, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real x_tmp = d_x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        d_x[n] = x_tmp;
    }
}


