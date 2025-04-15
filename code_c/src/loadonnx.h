#pragma once
#include <vector>
#include "vtk_tools.h"

std::vector<float> pointCloudNormalize(const std::vector<float>& points);
std::vector<int> classfier(std::vector<float>& points, bool use_cuda = false);
const int CLASS_NUM = 4;
