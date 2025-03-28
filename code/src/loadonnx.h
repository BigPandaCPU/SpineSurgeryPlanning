#pragma once
#include <vector>

std::vector<float> pointCloudNormalize(const std::vector<float>& points);
std::vector<int> classfier(std::vector<float>& points, bool use_cuda = false);
const int POINT_NUM = 5000;
const int CLASS_NUM = 4;
