#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <math.h>
#include <algorithm>
#include "loadonnx.h"
#include <ctime>




std::vector<int> classfier(std::vector<float>& points, bool use_cuda)
{
	clock_t start1 = clock();
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "part_seg");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	wchar_t* model_path = L"checkpoints/best_model.onnx";
	std::vector<const char*> input_node_names = { "input.1" };
	std::vector<const char*> output_node_names = {"272"};
	if (use_cuda)
	{
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 1;
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_options.AppendExecutionProvider_CUDA(cuda_option);
		model_path = L"checkpoints/best_model_gpu.onnx";
		output_node_names[0] = "274";
	}

	Ort::Session session(env, model_path, session_options);


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
	//Ort::AllocatorWithDefaultOptions allocator;
	//size_t num_input_nodes = session.GetInputCount();

	
	//std::vector<const char*> output_node_names = { "274"};

	const size_t input_tensor_size = 1 * 3 * POINT_NUM;
	std::vector<float> input_tensor_values(input_tensor_size);

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < POINT_NUM; j++)
		{
			input_tensor_values[POINT_NUM * i + j] = points[3 * j + i];
		}
	}
	clock_t end1 = clock();
	double duration1 = static_cast<double>(end1 - start1) / CLOCKS_PER_SEC;
	std::cout << "Load session used " << duration1 << std::endl;

	std::vector<int64_t> input_node_dims = { 1, 3, POINT_NUM };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> output(rawOutput, rawOutput + count);

	clock_t end2 = clock();
	double duration2 = static_cast<double>(end2 - end1) / CLOCKS_PER_SEC;
	std::cout << "onnx predict used " << duration2 << std::endl;

	//int predict_label = std::max_element(output.begin(), output.end()) - output.begin();
	//std::cout << predict_label << std::endl;
	std::vector<int> output_label;
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
		output_label.push_back(predict_label);
	}
	clock_t end3 = clock();
	double duration3 = static_cast<double>(end3 - end2) / CLOCKS_PER_SEC;
	std::cout << "get predict result used " << duration3 << std::endl;
	return output_label;
}


//int main()
//{
//	std::vector<float> points;
//	float x, y, z, nx, ny, nz;
//	char ch;
//	std::ifstream infile("close_0006.txt");
//	for (size_t i = 0; i < point_num; i++)
//	{
//		infile >> x >> ch >> y >> ch >> z >> ch >> nx >> ch >> ny >> ch >> nz;
//		points.push_back(x);
//		points.push_back(y);
//		points.push_back(z);
//	}
//	infile.close();
//	pc_normalize(points);
//	classfier(points);
//	return 0;
//}
