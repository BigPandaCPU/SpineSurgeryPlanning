#include "SpineRegistrationICP.h"
#include "SpinePointCloudSeg.h"
#include "PedicleSurgeryPlanning.h"
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkSmartPointer.h>
#include <vtkLandmarkTransform.h>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVertexGlyphFilter.h>
SpineRegistrationICP::SpineRegistrationICP(string& target_stl_file, const string& target_label, const string& template_dir, const int& sample_num)
{
	m_target_stl_file = target_stl_file;
	m_template_stl_dir = template_dir;
	m_target_label = target_label;
	m_sample_num = sample_num;

	m_source_landmark_file = m_template_stl_dir + "/label_" + m_target_label + ".txt";
	m_source_stl_file = m_template_stl_dir + "/label_" + m_target_label + ".stl";
	LoadLandmarksFromFile(m_source_landmark_file);

	m_source_points = GetPointsFromSTL(m_source_stl_file);
	m_target_points = GetPointsFromSTL(m_target_stl_file);
	m_source_poly_data = PedicleSurgeryPlanning::CreatePolyDataFromSTL(m_source_stl_file);
	m_target_poly_data = PedicleSurgeryPlanning::CreatePolyDataFromSTL(m_target_stl_file);
	CreateAlignedPCARotates();
	m_prealigned_matrix = NULL;

}
/*
Func：
	基于模板匹配的方法，得到椎体的左、右椎弓根以及椎体顶面的特征点
Input:
	prealigned_use_icp:粗配准的时候是否使用ICP算法

Output:
	m_target_left_points:配准得到的左侧椎弓根峡部的特征点
	m_target_right_points:配准得到的右侧椎弓根峡部的特征点
	m_target_top_points:配准得到的椎体顶面的特征点

Author:BigPanda
Date:2025.03.28 

*/
void SpineRegistrationICP::Registration(bool prealigned_use_icp)
{
	PreAlignedTwoPointClouds(prealigned_use_icp);

	auto pre_aligned_transform = vtkSmartPointer<vtkTransform>::New();
	pre_aligned_transform->SetMatrix(m_prealigned_matrix);

	auto pre_aligned_trans_poly_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	pre_aligned_trans_poly_filter->SetTransform(pre_aligned_transform);
	pre_aligned_trans_poly_filter->SetInputData(m_source_poly_data);
	pre_aligned_trans_poly_filter->Update();

	auto icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
	icp->SetSource(pre_aligned_trans_poly_filter->GetOutput());
	icp->SetTarget(m_target_poly_data);
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaximumNumberOfIterations(100);
	icp->StartByMatchingCentroidsOn();
	icp->Modified();
	icp->Update();

	m_icp_matrix = icp->GetMatrix();

	auto icp_trans_form_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	icp_trans_form_filter->SetInputData(pre_aligned_trans_poly_filter->GetOutput());
	icp_trans_form_filter->SetTransform(icp);
	icp_trans_form_filter->Update();

	m_final_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
	vtkMatrix4x4::Multiply4x4(m_icp_matrix, m_prealigned_matrix, m_final_matrix);

	PointsDotVTKMatrix4x4(m_source_top_points, m_final_matrix, m_target_top_points);
	PointsDotVTKMatrix4x4(m_source_left_points, m_final_matrix, m_target_left_points);
	PointsDotVTKMatrix4x4(m_source_right_points, m_final_matrix, m_target_right_points);
}

void SpineRegistrationICP::PreAlignedTwoPointClouds(bool use_icp)
{
	PCA(m_source_points, m_source_eigen_values, m_source_eigen_vectors, m_source_center);
	PCA(m_target_points, m_target_eigen_values, m_target_eigen_vectors, m_target_center);

	m_source_points_decenter = PointsDecenter(m_source_points, m_source_center);
	m_target_points_decenter = PointsDecenter(m_target_points, m_target_center);

	int min_index = 0;
	float min_distance = (numeric_limits<float>::max)();
	GetAlignedPCARotateOfMinDistance(min_index, min_distance, use_icp);
	auto trans = m_aligned_pca_rotates[min_index];

	vector<vector<float>> matrix1 = { {1.0, 0.0, 0.0, -m_source_center[0]},
												{0.0, 1.0, 0.0, -m_source_center[1]},
												{0.0, 0.0, 1.0, -m_source_center[2]},
												{0.0, 0.0, 0.0, 1.0} };

	vector<vector<float>> matrix2 = { {m_source_eigen_vectors[0][0], m_source_eigen_vectors[0][1], m_source_eigen_vectors[0][2], 0.0},
												{m_source_eigen_vectors[1][0], m_source_eigen_vectors[1][1], m_source_eigen_vectors[1][2], 0.0},
												{m_source_eigen_vectors[2][0], m_source_eigen_vectors[2][1], m_source_eigen_vectors[2][2], 0.0},
												{0.0, 0.0, 0.0, 1.0}, };

	vector<vector<float>> matrix3 = { {trans[0][0], trans[0][1], trans[0][2], 0.0},
												{trans[1][0], trans[1][1], trans[1][2], 0.0},
												{trans[2][0], trans[2][1], trans[2][2], 0.0},
												{0.0, 0.0, 0.0, 1.0} };

	vector<vector<float>> matrix4 = { {m_target_eigen_vectors[0][0], m_target_eigen_vectors[1][0], m_target_eigen_vectors[2][0], 0.0},
												{m_target_eigen_vectors[0][1], m_target_eigen_vectors[1][1], m_target_eigen_vectors[2][1], 0.0},
												{m_target_eigen_vectors[0][2], m_target_eigen_vectors[1][2], m_target_eigen_vectors[2][2], 0.0},
												{0.0, 0.0, 0.0, 1.0} };

	vector<vector<float>> matrix5 = { {1.0, 0.0, 0.0, m_target_center[0]},
												{0.0, 1.0, 0.0, m_target_center[1]},
												{0.0, 0.0, 1.0, m_target_center[2]},
												{0.0, 0.0, 0.0, 1.0} };

	auto tmp = Matrix4DotMatrix4(matrix5, matrix4);
	tmp = Matrix4DotMatrix4(tmp, matrix3);
	tmp = Matrix4DotMatrix4(tmp, matrix2);
	tmp = Matrix4DotMatrix4(tmp, matrix1);

	m_prealigned_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_prealigned_matrix->SetElement(i, j, tmp[i][j]);
		}
	}
}

void SpineRegistrationICP::PointsDotVTKMatrix4x4(const vector<float>& points, const vtkSmartPointer<vtkMatrix4x4> left_matrix, vector<float>& points_new)
{
	points_new = points;
	double* data = left_matrix->GetData();
	for (int i = 0; i < points.size() / 3; i++)
	{
		auto curX = points[i * 3 + 0];
		auto curY = points[i * 3 + 1];
		auto curZ = points[i * 3 + 2];
		for (int j = 0; j < 3; j++)
		{
			double cur_data = data[j * 4 + 0] * curX + data[j * 4 + 1] * curY + data[j * 4 + 2] * curZ + data[j * 4 + 3] * 1.0;
			points_new[i * 3 + j] = float(cur_data);
		}
	}
}

void SpineRegistrationICP::ShowRegistrationResult()
{

	auto source_actor = PedicleSurgeryPlanning::CreateActorFromPolyData(m_source_poly_data, "red", 0.9);

	auto final_transform = vtkSmartPointer<vtkTransform>::New();
	final_transform->SetMatrix(m_final_matrix);

	auto final_trans_poly_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	final_trans_poly_filter->SetTransform(final_transform);
	final_trans_poly_filter->SetInputData(m_source_poly_data);
	final_trans_poly_filter->Update();
	auto transformed_source_actor = PedicleSurgeryPlanning::CreateActorFromPolyData(final_trans_poly_filter->GetOutput(), "yellow", 0.8);

	auto target_actor = PedicleSurgeryPlanning::CreateActorFromPolyData(m_target_poly_data, "Cornsilk", 0.9);
	auto top_points_actors = PedicleSurgeryPlanning::CreatePointsActor(m_target_top_points, 0.3, 1.0, "red");
	auto left_points_actors = PedicleSurgeryPlanning::CreatePointsActor(m_target_left_points, 0.3, 1.0, "green");
	auto right_points_actors = PedicleSurgeryPlanning::CreatePointsActor(m_target_right_points, 0.3, 1.0, "blue");

	vector<vtkSmartPointer<vtkActor>> all_actors;
	all_actors.push_back(target_actor);
	all_actors.push_back(transformed_source_actor);
	all_actors.insert(all_actors.end(), top_points_actors.begin(), top_points_actors.end());
	all_actors.insert(all_actors.end(), left_points_actors.begin(), left_points_actors.end());
	all_actors.insert(all_actors.end(), right_points_actors.begin(), right_points_actors.end());
	PedicleSurgeryPlanning::ShowActors(all_actors, "ICP result");
}

vector<float> SpineRegistrationICP::GetTargetTopPoints()
{
	return m_target_top_points;
}
vector<float> SpineRegistrationICP::GetTargetLeftPoints()
{
	return m_target_left_points;
}
vector<float> SpineRegistrationICP::GetTargetRightPoints()
{
	return m_target_right_points;
}

vector<float> SpineRegistrationICP::GetSourceTopPoints()
{
	return m_source_top_points;
}
vector<float> SpineRegistrationICP::GetSourceLeftPoints()
{
	return m_source_left_points;
}
vector<float> SpineRegistrationICP::GetSourceRightPoints()
{
	return m_source_right_points;
}

vector<float> SpineRegistrationICP::GetPointsFromSTL(const string& stl_file)
{
	// 读取STL文件
	auto mesh = open3d::io::CreateMeshFromFile(stl_file);
	if (mesh->IsEmpty())
	{
		cout << "Mesh loaded successfully!" << endl;
		throw runtime_error("Error: Failed to load mesh from " + stl_file);
	}

	// 使用Poisson Disk Sampling从网格中采样点
	auto pcd = mesh->SamplePointsPoissonDisk(m_sample_num);

	// 将采样点转换为Eigen矩阵
	Eigen::MatrixXd points_xyz(pcd->points_.size(), 3); // 创建 Nx3 的矩阵
	for (size_t i = 0; i < pcd->points_.size(); ++i)
	{
		points_xyz.row(i) = pcd->points_[i].transpose(); // 将每个点复制到矩阵的行
	}

	vector<float> vec(pcd->points_.size() * 3);

	for (int i = 0; i < points_xyz.rows(); ++i)
	{
		for (int j = 0; j < points_xyz.cols(); ++j)
		{
			vec[i * 3 + j] = points_xyz(i, j);
		}
	}
	return vec;
}

bool SpineRegistrationICP::LoadLandmarksFromFile(const string& landmark_file)
{
	ifstream input(landmark_file);
	if (!input.is_open())
	{
		cout << "error, failed to open " << landmark_file << endl;
		return false;
	}

	string line;
	while (getline(input, line))
	{
		stringstream ss(line);
		string value;
		vector<string> parts;
		while (getline(ss, value, ','))
		{
			parts.push_back(value);
		}
		int label = stoi(parts[0]);
		float cur_x = stof(parts[1]);
		float cur_y = stof(parts[2]);
		float cur_z = stof(parts[3]);

		if (label == SpinePointCloudSeg::SPINE_POINT_LABEL::TOP - 1)
		{
			m_source_top_points.push_back(cur_x);
			m_source_top_points.push_back(cur_y);
			m_source_top_points.push_back(cur_z);
		}
		if (label == SpinePointCloudSeg::SPINE_POINT_LABEL::LEFT - 1)
		{
			m_source_left_points.push_back(cur_x);
			m_source_left_points.push_back(cur_y);
			m_source_left_points.push_back(cur_z);
		}

		if (label == SpinePointCloudSeg::SPINE_POINT_LABEL::RIGHT - 1)
		{
			m_source_right_points.push_back(cur_x);
			m_source_right_points.push_back(cur_y);
			m_source_right_points.push_back(cur_z);
		}
	}
	return true;
}

void SpineRegistrationICP::GetAlignedPCARotateOfMinDistance(int& min_index, float& min_distance, bool use_icp)
{
	auto pre_aligned_source_points = GetPointsDotMatrix(m_source_points_decenter, m_source_eigen_vectors);
	auto pre_aligned_target_points = GetPointsDotMatrix(m_target_points_decenter, m_target_eigen_vectors);

	vector<float> target_max_point, target_min_point;
	GetTheMaxAndMinAxisPoint(pre_aligned_target_points, target_max_point, target_min_point);
	
	vector<float> source_max_point, source_min_point;
	GetTheMaxAndMinAxisPoint(pre_aligned_source_points, source_max_point, source_min_point);

	vector<float> target_bbox = { target_max_point[0] - target_min_point[0],
								   target_max_point[1] - target_min_point[1],
								   target_max_point[2] - target_min_point[2] };


	vector<Eigen::Vector3d> points(pre_aligned_target_points.size()/3);
	for (int i = 0; i < pre_aligned_target_points.size() / 3; i++)
	{
		double curX = pre_aligned_target_points[i * 3 + 0];
		double curY = pre_aligned_target_points[i * 3 + 1];
		double curZ = pre_aligned_target_points[i * 3 + 2];
		points[i] = Eigen::Vector3d(curX, curY, curZ);
	}
	open3d::geometry::PointCloud point_cloud(points);
	open3d::geometry::KDTreeFlann  kd_tree(point_cloud);

	double * all_distance = (double*)malloc(sizeof(double) * m_aligned_pca_rotates.size());
	for (int i = 0; i < m_aligned_pca_rotates.size(); i++)
	{
		auto curR = m_aligned_pca_rotates[i];
		double cur_distance_sum = 0.0;
		auto tmp_points = GetPointsDotMatrix(pre_aligned_source_points, curR);

		/*auto tmp_points_actors = createPointsActor(tmp_points, 0.7, 1.0, "yellow");
		tmp_points_actors.insert(tmp_points_actors.end(), source_axis_actors_pre_aligned.begin(), source_axis_actors_pre_aligned.end());*/

		//showActors(tmp_points_actors, std::string(buff));

		vector<float> source_max_point, source_min_point;
		GetTheMaxAndMinAxisPoint(tmp_points, source_max_point, source_min_point);
		
		vector<float> source_bbox = { source_max_point[0] - source_min_point[0],
										   source_max_point[1] - source_min_point[1],
										   source_max_point[2] - source_min_point[2] };

		vector<vector<float>> scale_matrix = { {target_bbox[0] / source_bbox[0],0.0, 0.0},
													 {0.0, target_bbox[1] / source_bbox[1],0.0},
													 {0.0, 0.0, target_bbox[2] / source_bbox[2] } };
		auto tmp_points_new = GetPointsDotMatrix(tmp_points, scale_matrix);


		vector<float> cur_points;
		if (use_icp)
		{
			cur_points = ICP(tmp_points_new, pre_aligned_target_points);
		}
		else
		{
			cur_points = tmp_points_new;
		}
		vector<int> cur_indices;
		vector<double> cur_distance;

		for (int j = 0; j < cur_points.size() / 3; j++)
		{
			double curX = cur_points[j * 3 + 0];
			double curY = cur_points[j * 3 + 1];
			double curZ = cur_points[j * 3 + 2];
			Eigen::Vector3d query_point(curX, curY, curZ);
			kd_tree.SearchKNN(query_point, 3, cur_indices, cur_distance);
			cur_distance_sum += sqrt((cur_distance[0] + cur_distance[1] + cur_distance[2]) / 3.0);
		}
		all_distance[i] = cur_distance_sum;
	}

	min_index = 0;
	min_distance = all_distance[min_index];
	for (int i = 1; i < m_aligned_pca_rotates.size(); i++)
	{
		if (min_distance > all_distance[i])
		{
			min_index = i;
			min_distance = all_distance[i];
		}
	}
	free(all_distance);
}

vector<float> SpineRegistrationICP::ICP(const vector<float>& source_points, const vector<float>& target_points)
{
	auto source_points_vtk = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < source_points.size() / 3; i++)
	{
		float x = source_points[i * 3 + 0];
		float y = source_points[i * 3 + 1];
		float z = source_points[i * 3 + 2];
		source_points_vtk->InsertNextPoint(x, y, z);
	}
	auto target_points_vtk = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < target_points.size() / 3; i++)
	{
		float x = target_points[i * 3 + 0];
		float y = target_points[i * 3 + 1];
		float z = target_points[i * 3 + 2];
		target_points_vtk->InsertNextPoint(x, y, z);
	}
	auto source = vtkSmartPointer<vtkPolyData>::New();
	source->SetPoints(source_points_vtk);
	auto target = vtkSmartPointer<vtkPolyData>::New();
	target->SetPoints(target_points_vtk);

	auto source_glyph = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	source_glyph->SetInputData(source);
	source_glyph->Update();

	auto target_glyph = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	target_glyph->SetInputData(target);
	target_glyph->Update();

	auto icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
	icp->SetSource(source_glyph->GetOutput());
	icp->SetTarget(target_glyph->GetOutput());
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaximumNumberOfIterations(50);
	icp->StartByMatchingCentroidsOn();
	icp->Modified();
	icp->Update();


	auto icp_trans_form_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	icp_trans_form_filter->SetInputData(source_glyph->GetOutput());
	icp_trans_form_filter->SetTransform(icp);
	icp_trans_form_filter->Update();

	auto points = icp_trans_form_filter->GetOutput()->GetPoints();
	size_t num_points = points->GetNumberOfPoints();
	size_t data_size = num_points * 3;
	vector<float> points_out(data_size);
	for (size_t i = 0; i < num_points; ++i)
	{
		double* point = points->GetPoint(i);
		size_t index = i * 3;
		points_out[index] = static_cast<float>(point[0]);
		points_out[index + 1] = static_cast<float>(point[1]);
		points_out[index + 2] = static_cast<float>(point[2]);
	}
	return points_out;
}


vector<vector<float>> SpineRegistrationICP::Matrix4DotMatrix4(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2)
{
	vector<vector<float>> matrix = matrix1;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			matrix[i][j] = matrix1[i][0] * matrix2[0][j] + matrix1[i][1] * matrix2[1][j] + matrix1[i][2] * matrix2[2][j] + matrix1[i][3] * matrix2[3][j];
		}
	}
	return matrix;
}

void SpineRegistrationICP::GetTheMaxAndMinAxisPoint(const vector<float>& points, vector<float>& max_point, vector<float>& min_point)
{
	float maxX = numeric_limits<float>::lowest();
	float maxY = numeric_limits<float>::lowest();
	float maxZ = numeric_limits<float>::lowest();

	float minX = (numeric_limits<float>::max)();
	float minY = (numeric_limits<float>::max)();
	float minZ = (numeric_limits<float>::max)();

	for (int i = 0; i < points.size() / 3; i++)
	{
		if (maxX < points[i * 3 + 0]) { maxX = points[i * 3 + 0]; }
		if (maxY < points[i * 3 + 1]) { maxY = points[i * 3 + 1]; }
		if (maxZ < points[i * 3 + 2]) { maxZ = points[i * 3 + 2]; }

		if (minX > points[i * 3 + 0]) { minX = points[i * 3 + 0]; }
		if (minY > points[i * 3 + 1]) { minY = points[i * 3 + 1]; }
		if (minZ > points[i * 3 + 2]) { minZ = points[i * 3 + 2]; }
	}
	max_point = { maxX, maxY, maxZ };
	min_point = { minX, minY,  minZ };
}

/*
	Func:
		计算点与特征向量的乘积

	Input:
		points:输入的点坐标，以vector的形式进行存储，x0,y0,z0,x1,y1,z1.....
		matrix:特征向量表示的矩阵，每一行表示一个特征向量，行向量的形式表示，
			   而不是列的形式表示，所以在乘的时候，是对每一行相乘。
	Output:
		points_new:变换后的点，以vector形式进行存储

	Author:BigPdanda
	Date:2025.03.17 14:01
*/
vector<float> SpineRegistrationICP::GetPointsDotMatrix(const vector<float>& points, vector<vector<float>>& matrix)
{
	vector<float> points_new = points;
	for (int i = 0; i < points.size() / 3; i++)
	{
		points_new[i * 3 + 0] = points[i * 3 + 0] * matrix[0][0] + points[i * 3 + 1] * matrix[0][1] + points[i * 3 + 2] * matrix[0][2];
		points_new[i * 3 + 1] = points[i * 3 + 0] * matrix[1][0] + points[i * 3 + 1] * matrix[1][1] + points[i * 3 + 2] * matrix[1][2];
		points_new[i * 3 + 2] = points[i * 3 + 0] * matrix[2][0] + points[i * 3 + 1] * matrix[2][1] + points[i * 3 + 2] * matrix[2][2];
	}
	return points_new;

}


void SpineRegistrationICP::PCA(const vector<float>& points, vector<float>& eigen_values, vector<vector<float>>& eigen_vectors, vector<float>& points_center)
{
	int num_points = points.size() / 3;
	Eigen::MatrixXd cloud(num_points, 3);

	for (int i = 0; i < num_points; ++i)
	{
		cloud(i, 0) = points[i * 3 + 0];
		cloud(i, 1) = points[i * 3 + 1];
		cloud(i, 2) = points[i * 3 + 2];
	}

	// 1、计算质心
	Eigen::RowVector3d centroid = cloud.colwise().mean();

	// 2、去质心
	Eigen::MatrixXd demean = cloud;
	demean.rowwise() -= centroid;

	points_center.push_back(float(centroid[0]));
	points_center.push_back(float(centroid[1]));
	points_center.push_back(float(centroid[2]));


	// 计算协方差矩阵
	Eigen::MatrixXd covariance = demean.transpose() * demean / (demean.rows());
	//cout << "\n协方差矩阵:\n" << covariance << endl;

	//计算特征值和特征向量
	Eigen::EigenSolver<Eigen::MatrixXd> eig(covariance);
	Eigen::MatrixXd eig_vectors = eig.eigenvectors().real();
	Eigen::MatrixXd eig_values = eig.eigenvalues().real();



	double values[3] = { eig_values(0), eig_values(1), eig_values(2) };
	int indexs[3] = { 0, 1, 2 };
	for (int i = 0; i < 3; i++)
	{
		for (int j = i + 1; j < 3; j++)
		{
			if (values[i] < values[j])
			{
				auto tmp_value = values[i];
				values[i] = values[j];
				values[j] = tmp_value;
				auto tmp_index = indexs[i];
				indexs[i] = indexs[j];
				indexs[j] = tmp_index;
			}
		}
	}
	Eigen::Vector3d normal;
	vector<float> normalX, normalY;
	for (int i = 0; i < 3; i++)
	{
		auto cur_index = indexs[i];
		eigen_values.push_back(eig_values(indexs[i], 0));

		normal = eig_vectors.col(cur_index);
		if (i == 0)
		{
			normalX = { float(normal[0]), float(normal[1]), float(normal[2]) };
		}
		if (i == 1)
		{
			normalY = { float(normal[0]), float(normal[1]), float(normal[2]) };
		}
	}
	//特征向量统一使用右手坐标系
	auto normalZ = PedicleSurgeryPlanning::GetTwoVectorCrossValue(normalX, normalY);
	eigen_vectors.push_back(normalX);
	eigen_vectors.push_back(normalY);
	eigen_vectors.push_back(normalZ);
}

void SpineRegistrationICP::CreateAlignedPCARotates()
{
	const float T = sqrt(2.0) / 2.0;

	vector<vector<float>> R0 = { {1.0, 0.0, 0.0},{0.0, 1.0, 0.0,},{0.0, 0.0, 1.0} };
	vector<vector<float>> R1 = { {T, T, 0.0},{-T, T, 0.0,},{0.0, 0.0, 1.0} };  //45度
	vector<vector<float>> R2 = { {0.0, 1.0, 0.0},{-1.0, 0.0, 0.0,},{0.0, 0.0, 1.0} }; //90度
	vector<vector<float>> R3 = { {-T, T, 0.0},{-T, -T, 0.0,},{0.0, 0.0, 1.0} }; //135度
	vector<vector<float>> R4 = { {-1.0, 0.0, 0.0},{0.0, -1.0, 0.0,},{0.0, 0.0, 1.0} }; //180度
	vector<vector<float>> R5 = { {-T, -T, 0.0},{T, -T, 0.0},{0.0, 0.0, 1.0} }; //225度
	vector<vector<float>> R6 = { {0.0, -1.0, 0.0},{1.0, 0.0, 0.0},{0.0, 0.0, 1.0} }; //270度
	vector<vector<float>> R7 = { {T, -T, 0.0},{T, T, 0.0},{0.0, 0.0, 1.0} }; //315度

	vector<vector<float>> R8 = { {0.0, 1.0, 0.0},{1.0, 0.0, 0.0,},{0.0, 0.0, -1.0} }; //反向0度
	vector<vector<float>> R9 = { {T, T, 0.0},{T, -T, 0.0,},{0.0, 0.0, -1.0} }; //旋转45度
	vector<vector<float>> R10 = { {1.0, 0.0, 0.0},{0.0, -1.0, 0.0,},{0.0, 0.0, -1.0} }; //旋转90度
	vector<vector<float>> R11 = { {T, -T, 0.0},{-T, -T, 0.0,},{0.0, 0.0, -1.0} }; //旋转135度
	vector<vector<float>> R12 = { {0.0, -1.0, 0.0},{-1.0, 0.0, 0.0,},{0.0, 0.0, -1.0} }; //旋转180度
	vector<vector<float>> R13 = { {-T, -T, 0.0},{-T, T, 0.0,},{0.0, 0.0, -1.0} }; //旋转225度
	vector<vector<float>> R14 = { {-1.0, 0.0, 0.0},{0.0, 1.0, 0.0,},{0.0, 0.0, -1.0} }; //旋转270度
	vector<vector<float>> R15 = { {-T, T, 0.0},{T, T, 0.0,},{0.0, 0.0, -1.0} }; //旋转315度
	m_aligned_pca_rotates = { R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 };

}

vector<float> SpineRegistrationICP::PointsDecenter(const vector<float>& points, const vector<float>& center)
{
	vector<float> points_new = points;
	for (int i = 0; i < points_new.size() / 3; i++)
	{
		points_new[i * 3 + 0] = points[i * 3 + 0] - center[0];
		points_new[i * 3 + 1] = points[i * 3 + 1] - center[1];
		points_new[i * 3 + 2] = points[i * 3 + 2] - center[2];
	}
	return points_new;
}

SpineRegistrationICP::~SpineRegistrationICP()
{

}