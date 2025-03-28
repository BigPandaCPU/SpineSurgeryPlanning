#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <cwchar>  
#include <memory> 
#define  POINT_NUM 5000
#define  CLASS_NUM 4

//enum SPINE_POINT_LABEL { TOP = 1, LEFT = 2, RIGHT = 3 };

using namespace std;
class SpinePointCloudSeg
{
public:
	enum SPINE_POINT_LABEL { TOP = 1, LEFT = 2, RIGHT = 3 };
	static SpinePointCloudSeg* GetInstance(const string& model_dir, bool use_cuda);
	static vector<float> PointCloudNormalize(const vector<float>& points);
	static vector<float> GetAimPoints(const vector<float>& points, const vector<int>& labels, SPINE_POINT_LABEL aim_label);
	static wchar_t* ConvertStringToWchar(const string& str);
	vector<int> Classfier(const vector<float>& points);
	~SpinePointCloudSeg();

protected:
	SpinePointCloudSeg(const string& model_dir, bool use_cuda);
	static SpinePointCloudSeg* m_pInstance;
private:
	Ort::Env* m_env;
	Ort::SessionOptions* m_session_options;
	Ort::Session* m_session;
	vector<const char*> m_input_node_names;
	vector<const char*> m_output_node_names;
	string   m_model_dir;
	bool     m_use_cuda;
};
