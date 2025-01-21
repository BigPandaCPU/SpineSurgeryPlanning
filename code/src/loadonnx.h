#pragma once
#include <vector>

void pc_normalize(std::vector<float>& points);
std::vector<int> classfier(std::vector<float>& points);
const int point_num = 5000;
const int class_num = 4;
