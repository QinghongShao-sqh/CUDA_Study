//#include<stdint.h>
//#include<cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
////定义核函数, 在host调用， device上计算, 该函数作用为把空间幅值为1
//__global__ void kernel(float* a)
//{
//	a[threadIdx.x] = 1;
//
//}
//int main(int argc, char** argv)
//{
//	// 设置使用的显卡设备  cpu用 host表示  gpu用device表示
//	cudaSetDevice(0);
//	//分配显存空间  dx  表示 device 上的空间x
//	float* dx;
//	cudaMalloc((void**)&dx, 16 * sizeof(float));
//	//分配cpu内存空间
//	float hx[16] = { 0 };
//	//把cpu上的数据拷贝到gpu device 上
//	cudaMemcpy(dx, hx, 16 * sizeof(float), cudaMemcpyHostToDevice);
//	kernel << <1, 16 >> > (dx);
//	//把gpu上的数据，计算结果，拷贝到cpu host 上
//	cudaMemcpy(hx, dx, 16 * sizeof(float), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 16; i++)
//	{
//		printf("%f \n", hx[i]);
//	}
//	//释放资源 分别释放 显存和内存空间
//	cudaFree(dx);
//	free(hx);
////	cudaDeviceReset();
//	return 0;
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//


//#include <math.h>
//#include <stdio.h>
//#include<stdint.h>
//#include<cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
//
//#define CHECK(call)                                   \
//do                                                    \
//{                                                     \
//    const cudaError_t error_code = call;              \
//    if (error_code != cudaSuccess)                    \
//    {                                                 \
//        printf("CUDA Error:\n");                      \
//        printf("    File:       %s\n", __FILE__);     \
//        printf("    Line:       %d\n", __LINE__);     \
//        printf("    Error code: %d\n", error_code);   \
//        printf("    Error text: %s\n",                \
//            cudaGetErrorString(error_code));          \
//        exit(1);                                      \
//    }                                                 \
//} while (0)
//
//const double EPSILON = 1.0e-15;
//const double a = 1.23;
//const double b = 2.34;
//const double c = 3.57;
//void __global__  add(const double* x, const double* y, double* z, const int N);
//void check(const double* z, const int N);
//
//int main(void)
//{
//    const int N = 100000000;
//    const int M = sizeof(double) * N;
//    double* h_x = (double*)malloc(M);
//    double* h_y = (double*)malloc(M);
//    double* h_z = (double*)malloc(M);
//
//    for (int n = 0; n < N; ++n)
//    {
//        h_x[n] = a;
//        h_y[n] = b;
//    }
//
//    double* d_x, * d_y, * d_z;
//    CHECK(cudaMalloc((void**)&d_x, M));
//    CHECK(cudaMalloc((void**)&d_y, M));
//    CHECK(cudaMalloc((void**)&d_z, M));
//    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
//
//    const int block_size = 1280;
//    const int grid_size = (N + block_size - 1) / block_size;
//    add << <grid_size, block_size >> > (d_x, d_y, d_z, N);
//    CHECK(cudaGetLastError());
//    CHECK(cudaDeviceSynchronize());
//
//    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
//    check(h_z, N);
//
//    free(h_x);
//    free(h_y);
//   
//void __global__ add(const double* x, const double* y, double* z, const int N)
//{
//    const int n = blockDim.x * blockIdx.x + threadIdx.x;
//    if (n < N)
//    {
//        z[n] = x[n] + y[n];
//    }
//}
//
//void check(const double* z, const int N)
//{
//    bool has_error = false;
//    for (int n = 0; n < N; ++n)
//    {
//        if (fabs(z[n] - c) > EPSILON)
//        {
//            has_error = true;
//        }
//    }
//    printf("%s\n", has_error ? "Has errors" : "No errors");
//}

