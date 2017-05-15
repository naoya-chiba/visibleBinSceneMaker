#pragma warning(disable: 4819)

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include "visible_check.cuh"

using Vec3 = float[3];

__constant__ float c_dev[3];

__global__
void kernel_func_gpu(const float* const __restrict__ cloud, char* const __restrict__ is_visible, const float lambda_sqrd, const int i, const int offset)
{
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + offset;

	const float* const p = &cloud[3 * i];
	const float* const q = &cloud[3 * j];

	if (is_visible[i] == 0) return;
	if (j <= i) return;
	if (is_visible[j] == 0) return;

	Vec3 v;
	v[0] = p[0] - c_dev[0];
	v[1] = p[1] - c_dev[1];
	v[2] = p[2] - c_dev[2];
	
	const float v_sqrd = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

	Vec3 ta;
	ta[0] = q[0] - c_dev[0];
	ta[1] = q[1] - c_dev[1];
	ta[2] = q[2] - c_dev[2];

	const float tb = v[0] * ta[0] + v[1] * ta[1] + v[2] * ta[2];

	const float k = tb / v_sqrd;

	Vec3 tc;
	tc[0] = ta[0] - k * v[0];
	tc[1] = ta[1] - k * v[1];
	tc[2] = ta[2] - k * v[2];

	const float r_sqrd = tc[0] * tc[0] + tc[1] * tc[1] + tc[2] * tc[2];
	const float s_sqrd = k * k * v_sqrd;

	if (r_sqrd < lambda_sqrd * s_sqrd)
	{
		is_visible[j] = 0;
	}
}

__host__
void set_constant_var(const float* const c)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_dev, c, sizeof(float) * 3, 0, cudaMemcpyHostToDevice));
}

__host__
void call_kernel_func_gpu(const float* const cloud_dev, char* const is_visible_dev, const float lambda_sqrd, const int i, const int numel)
{
	const int thread_num = 1024;
	const int grid_num = static_cast<int>(std::floor(numel / thread_num));

	kernel_func_gpu <<< grid_num, thread_num >>>(cloud_dev, is_visible_dev, lambda_sqrd, i, 0);

	if (numel % thread_num != 0)
	{
		const int offset = grid_num * thread_num;
		kernel_func_gpu <<< 1, numel % thread_num >>>(cloud_dev, is_visible_dev, lambda_sqrd, i, offset);
	}

	//cudaThreadSynchronize();
}