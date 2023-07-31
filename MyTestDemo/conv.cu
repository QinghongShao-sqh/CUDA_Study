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
    int ti = threadIdx.x;//一个线程在一个线程块中的线程指标，其取值范围是从 0 到 blockDim.x - 1
    int bi = blockIdx.x;//一个线程在一个网格中的线程块指标，其取值范围是从 0 到 gridDim.x - 1
    int id = (bi * blockDim.x + ti);//grid划分成1维，block划分为1维
    if (id >= width * height)
    {
        return;
    }
    int row = id / width;//判断在第几行
    int col = id % width;//判断在第几列
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            float imgValue = 0;//默认为0，处理一些padding无效位置数据。比如（0,0）点卷积，其左边和上边是没有的，设为0
            //判断当前卷积操作的中心坐标
            int curRow = row - kernelSize / 2 + i;// 
            int curCol = col - kernelSize / 2 + j;
            //判断卷积操作中心是否超界
            if (curRow < 0 || curCol < 0 || curRow >= height || curCol >= width)
            {
            }
            else
            {   //取出图片上对应位置的点用来计算
                imgValue = img[curRow * width + curCol];
            }
            //卷积计算
            result[id] += kernel[i * kernelSize + j] * imgValue; //每个线程负责计算1个值 并存在result里面
        }
    }
}

int main()
{
    int width = 1000;//表示有 1000*1000个数值需要计算
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

    //网格大小：线程块数量  线程块大小：线程数量
    int threadNum = getThreadNum();
    int blockNum = (width * height - 0.5) / threadNum + 1;
    

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat < 1; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        //for (int i = 0; i <= 50; i++)
        //{
        //    conv << <blockNum, threadNum >> >  //网格大小：线程块数量  线程块大小：线程数量
        //        (imgGpu, kernelGpu, resultGpu, width, height, kernelSize);
        //}
        conv << <blockNum, threadNum >> >  //网格大小：线程块数量  线程块大小：线程数量
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