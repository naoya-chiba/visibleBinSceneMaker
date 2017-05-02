#pragma warning(disable: 4819)

#define GPU

#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "visible_check.h"

#ifdef GPU
#include "visible_check.cuh"
#endif

std::vector<double> visible_check(
	const Vec3& c,
	const std::vector<double>& cloud,
	const double lambda_sqrd)
{
	std::vector<char> is_visible(cloud.size() / 3, 1);

#ifdef GPU
	double* cloud_dev;
	double* c_dev;
	char* is_visible_dev;
	CUDA_SAFE_CALL(cudaMalloc((void**)&cloud_dev, sizeof(double) * cloud.size()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&c_dev, sizeof(double) * 3));
	CUDA_SAFE_CALL(cudaMalloc((void**)&is_visible_dev, sizeof(char) * is_visible.size()));
	CUDA_SAFE_CALL(cudaMemcpy(cloud_dev, cloud.data(), sizeof(double) * cloud.size(), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(c_dev, c, sizeof(double) * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(is_visible_dev, is_visible.data(), sizeof(char) * is_visible.size(), cudaMemcpyHostToDevice));
#endif

	for (int i = 0; i < cloud.size() / 3; ++i)
	{
		if (i % 1024 == 0)
		{
			std::cout << i << " / " << cloud.size() / 3 << std::endl;
		}

		if (is_visible[i] == 1)
		{
#ifdef GPU
			call_kernel_func_gpu(cloud_dev, is_visible_dev, c_dev, lambda_sqrd, i, cloud.size() / 3);
			CUDA_SAFE_CALL(cudaMemcpy(is_visible.data(), is_visible_dev, sizeof(char) * is_visible.size(), cudaMemcpyDeviceToHost));
#else
#pragma omp parallel for
			for (int j = i + 1; j < cloud.size() / 3; ++j)
			{
				if (is_visible[j] == 0) continue;

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
#endif
		}
	}

#ifdef GPU
	CUDA_SAFE_CALL(cudaMemcpy(is_visible.data(), is_visible_dev, sizeof(char) * is_visible.size(), cudaMemcpyDeviceToHost));
#endif

	std::vector<double> cloud_return;
	cloud_return.reserve(cloud.size());
	for (int i = 0; i < cloud.size() / 3; ++i)
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