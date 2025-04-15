#pragma once
#include <string.h>
#include <iostream>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>

#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkMatrix4x4.h>
#include "PedicleSurgeryPlanning.h"

using namespace std;
class SpineRegistrationICP
{
public:
	static vector<float> PointsDecenter(const vector<float>& points, const vector<float>& center);
	static void PCA(const vector<float>& points, vector<float>& eigen_values, vector<vector<float>>& eigen_vectors, vector<float>& points_center);
	static vector<float> GetPointsDotMatrix(const vector<float>& points, vector<vector<float>>& matrix);
	static void GetTheMaxAndMinAxisPoint(const vector<float>& points, vector<float>& max_point, vector<float>& min_point);
	static vector<float> ICP(const vector<float>& source_points, const vector<float>& target_points);
	static vector<vector<float>> Matrix4DotMatrix4(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2);
	static void PointsDotVTKMatrix4x4(const vector<float>& points, const vtkSmartPointer<vtkMatrix4x4> left_matrix, vector<float>& points_new);
	static vector<Eigen::Vector3d> ConvertVector2EigenVector(const vector<float>& points);

	SpineRegistrationICP(string& target_stl_file, const string& target_label,const string& template_dir, const int &sample_num);
	bool LoadLandmarksFromFile(const std::string& landmark_file);
	vector<float> GetPointsFromSTL(const string& stl_file);
	void PreAlignedTwoPointClouds(bool use_icp=true);
	void PreAlignedTwoPointCloudsOpen3d(double voxel_size=3.0);
	
	void CreateAlignedPCARotates();
	void Registration(bool prealigned_use_icp=true);
	void GetAlignedPCARotateOfMinDistance(int& min_index, float& min_distance, bool use_icp = true);
	vector<float> GetTargetTopPoints();
	vector<float> GetTargetLeftPoints();
	vector<float> GetTargetRightPoints();

	vector<float> GetSourceTopPoints();
	vector<float> GetSourceLeftPoints();
	vector<float> GetSourceRightPoints();
	void CreateFinalActors();
	void ShowRegistrationResult();
	void SaveRegistrationResult2Png(const string& save_png_file);
	~SpineRegistrationICP();



private:
	string  m_target_stl_file;
	string  m_target_label;
	vector<float> m_target_points;
	vector<float> m_target_eigen_values;
	vector<vector<float>> m_target_eigen_vectors;
	vector<float> m_target_center;
	vector<float> m_target_points_decenter;
	vtkSmartPointer<vtkPolyData> m_target_poly_data;
	
	
	vector<float> m_source_points;
	vector<float> m_source_eigen_values;
	vector<vector<float>> m_source_eigen_vectors;
	vector<float> m_source_center;
	vector<float> m_source_points_decenter;
	vtkSmartPointer<vtkPolyData> m_source_poly_data;


	vector<float> m_source_top_points;
	vector<float> m_source_left_points;
	vector<float> m_source_right_points;
	

	vector<float> m_target_top_points;
	vector<float> m_target_left_points;
	vector<float> m_target_right_points;

	string m_template_stl_dir;
	string m_source_landmark_file;
	string m_source_stl_file;
	int m_sample_num;

	vector<vector<vector<float>>> m_aligned_pca_rotates;
	vtkSmartPointer<vtkMatrix4x4> m_prealigned_matrix;
	vtkSmartPointer<vtkMatrix4x4> m_icp_matrix;
	vtkSmartPointer<vtkMatrix4x4> m_final_matrix;

	vector<vtkSmartPointer<vtkActor>> m_all_actors;
};