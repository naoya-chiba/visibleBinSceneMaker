#pragma warning(disable: 4819)

#include <vector>
#include <iostream>

#include <nmmintrin.h>

#include "visible_check.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "visible_check.cuh"
#endif

std::vector<float> visible_check(
	const Vec3& c,
	const std::vector<float>& cloud,
	const float lambda_sqrd)
{
	const int numel = static_cast<int>(cloud.size() / 3);

	std::vector<char> is_visible(numel, 1);

#ifdef USE_GPU
	float* cloud_dev;
	char* is_visible_dev;
	CUDA_SAFE_CALL(cudaMalloc((void**)&cloud_dev, sizeof(float) * cloud.size()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&is_visible_dev, sizeof(char) * numel));
	CUDA_SAFE_CALL(cudaMemcpy(cloud_dev, cloud.data(), sizeof(float) * cloud.size(), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(is_visible_dev, is_visible.data(), sizeof(char) * numel, cudaMemcpyHostToDevice));
	set_constant_var(c);

#endif

	for (int i = 0; i < numel; ++i)
	{
		if (i % 1024 == 0)
		{
			std::cout << i << " / " << numel << std::endl;
#ifdef USE_GPU
			CUDA_SAFE_CALL(cudaMemcpy(is_visible.data(), is_visible_dev, sizeof(char) * numel, cudaMemcpyDeviceToHost));
#endif
		}

		if (is_visible[i] == 1)
		{
#ifdef USE_GPU
			call_kernel_func_gpu(cloud_dev, is_visible_dev, lambda_sqrd, i, numel);
#else
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
	}

#ifdef USE_GPU
	CUDA_SAFE_CALL(cudaMemcpy(is_visible.data(), is_visible_dev, sizeof(char) * numel, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(cloud_dev));
	CUDA_SAFE_CALL(cudaFree(is_visible_dev));

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