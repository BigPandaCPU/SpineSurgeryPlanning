#include "SpinePointCloudSeg.h"
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
#include <assert.h>

SpinePointCloudSeg* SpinePointCloudSeg::m_pInstance = NULL;

SpinePointCloudSeg* SpinePointCloudSeg::GetInstance(const string& model_dir, bool use_cuda)
{
	if (NULL == m_pInstance)
	{
		m_pInstance = new SpinePointCloudSeg(model_dir, use_cuda);
	}
	return m_pInstance;
}
SpinePointCloudSeg::~SpinePointCloudSeg()
{
	delete m_env;
	delete m_session;
	delete m_session_options;
}

vector<float> SpinePointCloudSeg::GetAimPoints(const vector<float>& points, const vector<int>& labels, SPINE_POINT_LABEL aim_label)
{
	assert(points.size() / 3 == labels.size());
	std::vector<float> aim_points;
	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i] == aim_label)
		{
			aim_points.push_back(points[i * 3 + 0]);
			aim_points.push_back(points[i * 3 + 1]);
			aim_points.push_back(points[i * 3 + 2]);
		}
	}
	return aim_points;
}

wchar_t* SpinePointCloudSeg::ConvertStringToWchar(const string& str)
{
	size_t len = str.size();
	std::unique_ptr<wchar_t[]> wstr(new wchar_t[len + 1]);
	mbstowcs(wstr.get(), str.c_str(), len + 1);
	wchar_t* wptr = wstr.release();
	return wptr;
}

SpinePointCloudSeg::SpinePointCloudSeg(const string& model_dir, bool use_cuda)
{
	m_model_dir = model_dir;
	m_use_cuda = use_cuda;

	m_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "part_seg");
	m_session_options = new Ort::SessionOptions;
	m_session_options->SetIntraOpNumThreads(1);
	m_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	string model_file = model_dir + "/best_model.onnx";

	wchar_t* model_path = ConvertStringToWchar(model_file);

	m_input_node_names = { "input.1" };
	m_output_node_names = { "272" };
	if (m_use_cuda)
	{
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 1;
		m_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		m_session_options->AppendExecutionProvider_CUDA(cuda_option);
		string model_file_gpu = model_dir + "/best_model_gpu.onnx";
		delete model_path;
		model_path = ConvertStringToWchar(model_file_gpu);
		m_output_node_names[0] = "274";
	}

	m_session = new Ort::Session(*m_env, model_path, *m_session_options);

	//OrtCUDAProviderOptions cuda_option;
	//cuda_option.device_id = 0;
	//cuda_option.arena_extend_strategy = 0;
	//cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	//cuda_option.gpu_mem_limit = SIZE_MAX;
	//cuda_option.do_copy_in_default_stream = 1;
	//session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	//session_options.AppendExecutionProvider_CUDA(cuda_option);

	//const wchar_t* model_path = L"checkpoints/best_model.onnx";
	//Ort::Session session(env, model_path, session_options);
	Ort::AllocatorWithDefaultOptions allocator;
	size_t num_input_nodes = m_session->GetInputCount();
	//std::cout << "\nnum input nodes:" << num_input_nodes << std::endl;

}



vector<int> SpinePointCloudSeg::Classfier(const vector<float>& points)
{
	auto points_normal = PointCloudNormalize(points);

	const size_t input_tensor_size = 1 * 3 * POINT_NUM;
	std::vector<float> input_tensor_values(input_tensor_size);
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < POINT_NUM; j++)
		{
			input_tensor_values[POINT_NUM * i + j] = points_normal[3 * j + i];
		}
	}
	//clock_t end1 = clock();
	//double duration1 = static_cast<double>(end1 - start1) / CLOCKS_PER_SEC;
	//std::cout << "Load session used " << duration1 << std::endl;

	std::vector<int64_t> input_node_dims = { 1, 3, POINT_NUM };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> output_tensors = m_session->Run(Ort::RunOptions{ nullptr }, m_input_node_names.data(), ort_inputs.data(), m_input_node_names.size(), m_output_node_names.data(), m_output_node_names.size());

	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> output(rawOutput, rawOutput + count);

	/*clock_t end2 = clock();
	double duration2 = static_cast<double>(end2 - end1) / CLOCKS_PER_SEC;
	std::cout << "onnx predict used " << duration2 << std::endl;*/

	//int predict_label = std::max_element(output.begin(), output.end()) - output.begin();
	//std::cout << predict_label << std::endl;
	std::vector<int> output_label(POINT_NUM);
	float tmp[CLASS_NUM] = { 0.0 };
	for (int i = 0; i < POINT_NUM; i++)
	{
		for (int j = 0; j < CLASS_NUM; j++)
		{
			tmp[j] = output[i * CLASS_NUM + j];
		}
		int size = sizeof(tmp) / sizeof(tmp[0]); // 计算数组大小
		auto max_it = std::max_element(tmp, tmp + size); // 查找最大值
		int predict_label = std::distance(tmp, max_it);
		output_label[i] = predict_label;
	}
	/*clock_t end3 = clock();
	double duration3 = static_cast<double>(end3 - end2) / CLOCKS_PER_SEC;
	std::cout << "get predict result used " << duration3 << std::endl;*/
	return output_label;
}

vector<float> SpinePointCloudSeg::PointCloudNormalize(const vector<float>& points)
{
	vector<float> points_normal(points);
	int point_num = points.size() / 3;

	float mean_x = 0, mean_y = 0, mean_z = 0;
	for (int i = 0; i < point_num; ++i)
	{
		mean_x += points[3 * i];
		mean_y += points[3 * i + 1];
		mean_z += points[3 * i + 2];
	}

	mean_x /= point_num;
	mean_y /= point_num;
	mean_z /= point_num;

	for (int i = 0; i < point_num; ++i)
	{
		points_normal[3 * i] -= mean_x;
		points_normal[3 * i + 1] -= mean_y;
		points_normal[3 * i + 2] -= mean_z;
	}

	float m = 0;
	for (int i = 0; i < point_num; ++i)
	{
		if (sqrt(pow(points_normal[3 * i], 2) + pow(points_normal[3 * i + 1], 2) + pow(points_normal[3 * i + 2], 2)) > m)
			m = sqrt(pow(points_normal[3 * i], 2) + pow(points_normal[3 * i + 1], 2) + pow(points_normal[3 * i + 2], 2));
	}

	for (int i = 0; i < point_num; ++i)
	{
		points_normal[3 * i] /= m;
		points_normal[3 * i + 1] /= m;
		points_normal[3 * i + 2] /= m;
	}
	return points_normal;
}