#include<stdint.h>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
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

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real* h_x, real* d_x, const int method);

int main(void)
{
    real* h_x = (real*)malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real* d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(real* d_x, real* d_y)
{
    const int tid = threadIdx.x;
 //定义指针X，右边表示 d_x 数组第  blockDim.x * blockIdx.x个元素的地址
 //该情况下x 在不同线程块，指向全局内存不同的地址---》使用不同的线程块对dx数组不同部分分别进行处理   
    real* x = d_x + blockDim.x * blockIdx.x;

    //blockDim.x >> 1  等价于 blockDim.x /2，核函数中，位操作比 对应的整数操作高效
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        //同步语句，作用：同一个线程块内的线程按照代码先后执行指令（块内同步，块外不用同步）
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}

void __global__ reduce_shared(real* d_x, real* d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    //定义了共享内存数组 s_y[128]，注意关键词  __shared__
    __shared__ real s_y[128];
    //将全局内存中的数据复制到共享内存中
    //共享内存的特 征：每个线程块都有一个共享内存变量的副本
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    //调用函数 __syncthreads 进行线程块内的同步
    __syncthreads();
    //归约计算用共享内存变量替换了原来的全局内存变量。这里也要记住： 每个线程块都对其中的共享内存变量副本进行操作。在归约过程结束后，每一个线程
    //块中的 s_y[0] 副本就保存了若干数组元素的和
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

void __global__ reduce_dynamic(real* d_x, real* d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    //声明 动态共享内存 s_y[]  限定词 extern，不能指定数组大小
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {//将每一个线程块中归约的结果从共享内存 s_y[0] 复制到全局内 存d_y[bid]
        d_y[bid] = s_y[0];
    }
}

real reduce(real* d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real* d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real* h_y = (real*)malloc(ymem);

    switch (method)
    {
    case 0:
        reduce_global << <grid_size, BLOCK_SIZE >> > (d_x, d_y);
        break;
    case 1:
        reduce_shared << <grid_size, BLOCK_SIZE >> > (d_x, d_y);
        break;
    case 2:
        reduce_dynamic << <grid_size, BLOCK_SIZE, smem >> > (d_x, d_y);
        break;
    default:
        printf("Error: wrong method\n");
        exit(1);
        break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int n = 0; n < grid_size; ++n)
    {
        result += h_y[n];
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(real* h_x, real* d_x, const int method)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}


