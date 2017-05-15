#pragma warning(disable: 4819)
#define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <iostream>

#ifdef USE_SIMD
#ifdef _MSC_VER
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif
#endif

#include "visible_check.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "visible_check.cuh"
#endif

#if defined(USE_GPU) && defined(USE_SIMD)
#error You cannot set USE_GPU and USE_SIMD simultaneously.
#endif

std::vector<float> visible_check(
	const Vec3& c,
	const std::vector<float>& cloud,
	const float lambda_sqrd)
{
	const int numel = static_cast<int>(cloud.size() / 3);

	std::vector<char> is_visible(numel, 1);

#ifdef USE_SIMD
	float* const cloud_aligned = static_cast<float*>(_mm_malloc(sizeof(float) * numel * 4, 32));
	char* const is_visible_aligned = static_cast<char*>(_mm_malloc(sizeof(char) * numel, 32));
	for (int i = 0; i < numel; ++i)
	{
		cloud_aligned[4 * i + 0] = cloud[3 * i + 0];
		cloud_aligned[4 * i + 1] = cloud[3 * i + 1];
		cloud_aligned[4 * i + 2] = cloud[3 * i + 2];
		cloud_aligned[4 * i + 3] = 0.0;
	}
	memset(is_visible_aligned, 1, sizeof(char) * numel);

	alignas(16) const float c_aligned[4] = { c[0], c[1], c[2], 0.0f };
#endif

#ifdef USE_GPU
	thrust::device_vector<float> cloud_dev_vec = cloud;
	thrust::device_vector<char> is_visible_dev_vec = is_visible;

	char* is_visible_pinned = nullptr;
	CUDA_SAFE_CALL(cudaMallocHost(&is_visible_pinned, sizeof(char) * numel, cudaHostAllocDefault));

	set_constant_var(c);
#endif

	for (int i = 0; i < numel; ++i)
	{
		if (i % 1024 == 0)
		{
			std::cout << i << " / " << numel << std::endl;
#ifdef USE_GPU
			CUDA_SAFE_CALL(cudaMemcpy(is_visible_pinned, thrust::raw_pointer_cast(is_visible_dev_vec.data()), sizeof(char) * numel, cudaMemcpyDeviceToHost));
#endif
		}

#if defined(USE_GPU)
		if (is_visible_pinned[i] == 0) continue;
		call_kernel_func_gpu(thrust::raw_pointer_cast(cloud_dev_vec.data()), thrust::raw_pointer_cast(is_visible_dev_vec.data()), lambda_sqrd, i, numel);
#elif defined(USE_SIMD)
		if (is_visible_aligned[i] == 0) continue;
#pragma omp parallel for
		for (int j = i + 1; j < numel; ++j)
		{
			if (is_visible_aligned[j] == 0) continue;

			__m128 p = _mm_load_ps(&cloud_aligned[4 * i]);
			__m128 q = _mm_load_ps(&cloud_aligned[4 * j]);
			__m128 c = _mm_load_ps(c_aligned);

			// v = p - c
			__m128 v = _mm_sub_ps(p, c);

			// v_sqrd = v' * v
			__m128 v_sqrd = _mm_mul_ps(v, v);
			v_sqrd = _mm_hadd_ps(v_sqrd, v_sqrd);
			v_sqrd = _mm_hadd_ps(v_sqrd, v_sqrd);

			// ta = q - c
			__m128 ta = _mm_sub_ps(q, c);

			// k = v' * ta / v_sqrd
			__m128 k = _mm_mul_ps(v, ta);
			k = _mm_hadd_ps(k, k);
			k = _mm_hadd_ps(k, k);
			k = _mm_div_ps(k, v_sqrd);

			// k * v - ta
			__m128 tc = _mm_fmsub_ps(k, v, ta);

			// r_sqrd = tc' * tc
			__m128 r_sqrd = _mm_mul_ps(tc, tc);
			r_sqrd = _mm_hadd_ps(r_sqrd, r_sqrd);
			r_sqrd = _mm_hadd_ps(r_sqrd, r_sqrd);

			alignas(16) float r_sqrd_mem;
			alignas(16) float v_sqrd_mem;
			alignas(16) float k_mem;

			_mm_store_ss(&r_sqrd_mem, r_sqrd);
			_mm_store_ss(&v_sqrd_mem, v_sqrd);
			_mm_store_ss(&k_mem, k);

			const float s_sqrd = k_mem * k_mem * v_sqrd_mem;

			if (r_sqrd_mem < lambda_sqrd * s_sqrd)
			{
				is_visible_aligned[j] = 0;
			}
		}
#else
		if (is_visible[i] == 0) continue;
#pragma omp parallel for
		for (int j = i + 1; j < numel; ++j)
		{
			if (is_visible[j] == 0) continue;

			const float* const p = &cloud[3 * i];
			const float* const q = &cloud[3 * j];

			Vec3 v;
			v[0] = p[0] - c[0];
			v[1] = p[1] - c[1];
			v[2] = p[2] - c[2];

			const float v_sqrd = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

			Vec3 ta;
			ta[0] = q[0] - c[0];
			ta[1] = q[1] - c[1];
			ta[2] = q[2] - c[2];

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
#endif
	}

#ifdef USE_GPU
	CUDA_SAFE_CALL(cudaMemcpy(is_visible.data(), thrust::raw_pointer_cast(is_visible_dev_vec.data()), sizeof(char) * numel, cudaMemcpyDeviceToHost));
	cudaFreeHost(is_visible_pinned);
#endif

#ifdef USE_SIMD
	memcpy(is_visible.data(), is_visible_aligned, sizeof(char) * numel);
	_mm_free(cloud_aligned);
	_mm_free(is_visible_aligned);
#endif

	std::vector<float> cloud_return;
	cloud_return.reserve(cloud.size());
	for (int i = 0; i < numel; ++i)
	{
		if (is_visible[i] == 1)
		{
			cloud_return.push_back(cloud[3 * i + 0]);
			cloud_return.push_back(cloud[3 * i + 1]);
			cloud_return.push_back(cloud[3 * i + 2]);
		}
	}

	return cloud_return;
}