#pragma once
#pragma warning(disable: 4819)

#include <vector>

using Vec3 = float[3];

std::vector<float> visible_check(const Vec3& c, const std::vector<float>& cloud, const float lambda_sqrd);