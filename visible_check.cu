#pragma warning(disable: 4819)

#include <iostream>

#include "visible_check.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using Vec3 = double[3];

__global__
void kernel_func_gpu(const double* const cloud, char* const is_visible, const double* const c, const double lambda_sqrd, const int i)
{
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	//const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + threadIdx.y * 16 + threadIdx.z * 16 * 16;
	if (j <= i) return;
	if (is_visible[i] == 0) return;
	if (is_visible[j] == 0) return;

	const double* const p = &cloud[3 * i];
	const double* const q = &cloud[3 * j];

	Vec3 v;
	v[0] = p[0] - c[0];
	v[1] = p[1] - c[1];
	v[2] = p[2] - c[2];

	const double v_sqrd = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

	Vec3 ta;
	ta[0] = q[0] - c[0];
	ta[1] = q[1] - c[1];
	ta[2] = q[2] - c[2];

	const double tb = v[0] * ta[0] + v[1] * ta[1] + v[2] * ta[2];

	const double k = tb / v_sqrd;

	Vec3 tc;
	tc[0] = ta[0] - k * v[0];
	tc[1] = ta[1] - k * v[1];
	tc[2] = ta[2] - k * v[2];

	const double r_sqrd = tc[0] * tc[0] + tc[1] * tc[1] + tc[2] * tc[2];
	const double s_sqrd = k * k * v_sqrd;

	if (r_sqrd < lambda_sqrd * s_sqrd)
	{
		is_visible[j] = 0;
	}
}

__global__
void kernel_func_gpu_off(const double* const cloud, char* const is_visible, const double* const c, const double lambda_sqrd, const int i, const int offset)
{
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + offset;
	//const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + threadIdx.y * 16 + threadIdx.z * 16 * 16;
	if (j <= i) return;
	if (is_visible[i] == 0) return;
	if (is_visible[j] == 0) return;

	const double* const p = &cloud[3 * i];
	const double* const q = &cloud[3 * j];

	Vec3 v;
	v[0] = p[0] - c[0];
	v[1] = p[1] - c[1];
	v[2] = p[2] - c[2];

	const double v_sqrd = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

	Vec3 ta;
	ta[0] = q[0] - c[0];
	ta[1] = q[1] - c[1];
	ta[2] = q[2] - c[2];

	const double tb = v[0] * ta[0] + v[1] * ta[1] + v[2] * ta[2];

	const double k = tb / v_sqrd;

	Vec3 tc;
	tc[0] = ta[0] - k * v[0];
	tc[1] = ta[1] - k * v[1];
	tc[2] = ta[2] - k * v[2];

	const double r_sqrd = tc[0] * tc[0] + tc[1] * tc[1] + tc[2] * tc[2];
	const double s_sqrd = k * k * v_sqrd;

	if (r_sqrd < lambda_sqrd * s_sqrd)
	{
		is_visible[j] = 0;
	}
}

__host__ void
call_kernel_func_gpu(const double* const cloud_dev, char* const is_visible_dev, const double* const c_dev, const double lambda_sqrd, const int i, const int numel)
{
	//for (int j = 0; j < numel; ++j)
	//{
	//	kernel_func_gpu <<< 1, 1 >>>(cloud_dev, is_visible_dev, c_dev, lambda_sqrd, i, j);
	//}

	const int grid = (int)std::floor(numel / 1024);

	kernel_func_gpu <<< grid, 1024 >>>(cloud_dev, is_visible_dev, c_dev, lambda_sqrd, i);

	if (numel % 1024 != 0)
	{
		const int offset = grid * 1024;
		kernel_func_gpu_off <<< 1, numel % 1024 >>>(cloud_dev, is_visible_dev, c_dev, lambda_sqrd, i, offset);
	}

	//CUDA_SAFE_CALL(cudaGetLastError());


	//cudaThreadSynchronize();


}