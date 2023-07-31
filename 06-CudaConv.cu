#include<stdint.h>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

const int NUM_REPEATS = 10;

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



static void HandleError(cudaError_t err,
    const char* file,

    int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n",
            cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int getThreadNum()
{
    cudaDeviceProp prop;
    int count;

    CHECK(cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

__global__ void conv(float* img, float* kernel, float* result,
    int width, int height, int kernelSize)
{
    int ti = threadIdx.x;
    int bi = blockIdx.x;
    int id = (bi * blockDim.x + ti);
    if (id >= width * height)
    {
        return;
    }
    int row = id / width;
    int col = id % width;
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            float imgValue = 0;
            int curRow = row - kernelSize / 2 + i;
            int curCol = col - kernelSize / 2 + j;
            if (curRow < 0 || curCol < 0 || curRow >= height || curCol >= width)
            {
            }
            else
            {
                imgValue = img[curRow * width + curCol];
            }
            result[id] += kernel[i * kernelSize + j] * imgValue;
        }

    }
}

int main()
{
    int width = 1000;
    int height = 1000;
    float* img = new float[width * height];
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            img[col + row * width] = (col + row) % 256;
        }
    }

    int kernelSize = 3;
    float* kernel = new float[kernelSize * kernelSize];
    for (int i = 0; i < kernelSize * kernelSize; ++i)
    {
        kernel[i] = i % kernelSize - 1;
    }

    float* imgGpu;
    float* kernelGpu;
    float* resultGpu;

    CHECK(cudaMalloc((void**)&imgGpu, width * height * sizeof(float)));
    CHECK(cudaMalloc((void**)&kernelGpu, kernelSize * kernelSize * sizeof(float)));
    CHECK(cudaMalloc((void**)&resultGpu, width * height * sizeof(float)));

    CHECK(cudaMemcpy(imgGpu, img,
        width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(kernelGpu, kernel,
        kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    int threadNum = getThreadNum();
    int blockNum = (width * height - 0.5) / threadNum + 1;

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        conv << <blockNum, threadNum >> >
            (imgGpu, kernelGpu, resultGpu, width, height, kernelSize);

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


    float* result = new float[width * height];
    CHECK(cudaMemcpy(result, resultGpu,
        width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // visualization
    printf("img\n");
    for (int row = 0; row < 10; ++row)
    {
        for (int col = 0; col < 10; ++col)
        {
            printf("%2.0f ", img[col + row * width]);
        }
        printf("\n");
    }
    printf("kernel\n");
    for (int row = 0; row < kernelSize; ++row)
    {
        for (int col = 0; col < kernelSize; ++col)
        {
            printf("%2.0f ", kernel[col + row * kernelSize]);
        }
        printf("\n");
    }

    printf("result\n");
    for (int row = 0; row < 10; ++row)
    {
        for (int col = 0; col < 10; ++col)
        {
            printf("%2.0f ", result[col + row * width]);
        }
        printf("\n");
    }


    return 0;
}
