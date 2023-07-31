#include<stdint.h>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//定义核函数, 在host调用， device上计算, 该函数作用为把空间幅值为1
__global__ void kernel(float* a)
{
	a[threadIdx.x] = 1;

}
int main(int argc, char** argv)
{
	// 设置使用的显卡设备  cpu用 host表示  gpu用device表示
	cudaSetDevice(0);
	//分配显存空间  dx  表示 device 上的空间x
	float* dx;
	cudaMalloc((void**)&dx, 16 * sizeof(float));
	//分配cpu内存空间
	float hx[16] = { 0 };
	//把cpu上的数据拷贝到gpu device 上
	cudaMemcpy(dx, hx, 16 * sizeof(float), cudaMemcpyHostToDevice);
	kernel << <1, 16 >> > (dx);
	//把gpu上的数据，计算结果，拷贝到cpu host 上
	cudaMemcpy(hx, dx, 16 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 16; i++)
	{
		printf("%f \n", hx[i]);
	}
	//释放资源 分别释放 显存和内存空间
	cudaFree(dx);
	free(hx);
//	cudaDeviceReset();
	return 0;
}


























