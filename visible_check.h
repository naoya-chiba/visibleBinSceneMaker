#pragma once
#pragma warning(disable: 4819)

#include <vector>

using Vec3 = double[3];

std::vector<double> visible_check(const Vec3& c, const std::vector<double>& cloud, const double lambda_sqrd);