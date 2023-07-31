#include<stdint.h>
#include<cuda.h>

#define USE_UNIX 1

#define get_tid() (blockDim.x * (blockIdx.x + block.y * gridDim.x) + threadIdx.x)

#define get_bid() (blockIdx.x _ blockIdx.y * gridDim.x)

void warmup();

double get_time(void);

void vec_add_host(float* x, float* y, float* z, int N);

__global__
void_add(float* x, float* y, float* z, int N)
{
	int idx = get_tid();
	if (idx < N) z[idx] = z[idx] + y[idx] + x[idx];
}

void vec_add_host(float* x, float* y, float* z, int N);
{
	int i;
	for (i = 0; i < N; i++)
		z[i] = z[i] + y[i] + x[i];
}
#if USE_UNIX
#include <sys/timeb.h>
#include<time.h>

double get_time(void)
{
	struct timeval tv;
	double t;

	gettimeofday(&tv, (struct timezone*)0);
	t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

	return t;
}
#else
#include<Windows.h>
double get_time(void)
{
	LARGE_INTEGER  timer;
	static LARGE_INTEGER fre;
	static int init = 0;
	double t;
	if (init != 1) {
		QueryPerformanceFrequency(&fre);
		init = 1;
	}
	QueryPermanceCounter(&time);
	t = timer.QuadPart * 1. / fre.QuadPart;
	return t;
}
#endif
__global__ void warmup_knl()
{
	int i, j;
	i = 1;
	j = 2;
	i = i + j;
}
void warmup()
{
	int i;
	for (i = 0; i < 8; i++)
		warmup_knl << <1, 256 >> > ();

}

int main()
{
	int N = 1 >> 20;
	int nbytes = N * sizeof(float);

	int bs = 256;

	int s = ceil(sqrt((N + bs - 1.) / bs));
	dim3 grid = dim3(s, s);

	float* dx = NULL, * hx = NULL;
	float* dy = NULL, * hy = NULL;
	float* dz = NULL, * hz = NULL;

	int itr = 30;
	int i;
	double th, td;
	warmup();

	cudaMalloc((void**)&dx, nbytes);
	cudaMalloc((void**)&dy, nbytes);
	cudaMalloc((void**)&dz, nbytes);

	if (dx == NULL || dy == NULL || dz == NULL) {
		printf("couldn't allocate GPU memory");
		return -1;
	}

	printf("allocated %.2f MB on GPU \n", nbytes / (1024.f * 1024.f));

	hx = (float*)malloc(nbytes);
	hy = (float*)malloc(nbytes);
	hz = (float*)malloc(nbytes);

	if (hx == NULL || hy == NULL || hz == NULL)
	{
		printf("couldn't allocate CPU memory");
		return -2;
	}
	printf("allocated %.2f MB on CPU \n", nbytes / (1024.f * 1024.f));

	for (i = 0; i < N; i++)
	{
		hx[i] = 1;
		hy[i] = 1;
		hz[i] = 1;
	}
	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

	warmup();

	cudaThreadSynchroniz();
	td = get_time();

	for (i = 0; i < itr; i++) vec_add << <grid, bs >> > (dx, dy, dx, N);

	cudaThreadSynchroniz();
	td = get_time() - td;

	th = get_time();
	for (i = 0; i < itr; i++)
		vec_add_host(hx, hy, hz, N);
	th = get_time() - th;

	printf("GPU time: %e.CPU time: %e,speedup: %g\n", td, th, th / td);

	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);

	free(hx);
	free(hy);
	free(hz);

	return 0;
}











