#include "PedicleSurgeryPlanning.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkStripper.h>
#include <vtkTriangle.h>
#include <vtkOBBTree.h>
#include <vtkLineSource.h>
#include <vtkTubeFilter.h>
#include <vtkDecimatePro.h>
#include <vtkWindowToImageFilter.h>
#include <vtkCamera.h>
#include <vtkPoints.h>
#include <vtkCutter.h>
#include <vtkStripper.h>
#include <vtkPNGWriter.h>
#include <vtkImageAppend.h>
#include <vtkMatrix4x4.h>
#include <vtkLineSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkLandmarkTransform.h>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVertexGlyphFilter.h>
#include <limits>

#include <vtkAutoInit.h>
#include <vtkRegularPolygonSource.h>

PedicleSurgeryPlanning::PedicleSurgeryPlanning(vtkSmartPointer<vtkPolyData> spine_poly_data,
	vector<float> top_points, vector<float> left_points, vector<float> right_points)
{
	m_spine_poly_data = spine_poly_data;
	m_top_points = top_points;
	m_left_points = left_points;
	m_right_points = right_points;
	m_plan_result = true;
}

vector<float> PedicleSurgeryPlanning::GetPointsMean(const vector<float>& points)
{
	int num_points = points.size() / 3;
	float x = 0.0;
	float y = 0.0;
	float z = 0.0;
	for (int i = 0; i < num_points; i++)
	{
		x += points[i * 3 + 0];
		y += points[i * 3 + 1];
		z += points[i * 3 + 2];
	}
	x = x / float(num_points);
	y = y / float(num_points);
	z = z / float(num_points);
	vector<float> mean_point;
	mean_point.push_back(x);
	mean_point.push_back(y);
	mean_point.push_back(z);

	return mean_point;
}
vtkSmartPointer<vtkActor> PedicleSurgeryPlanning::CreateActorFromSTL(string stl_file, const string &color, float opacity)
{
	// 创建颜色对象
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// 创建并配置STL读取器
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(stl_file.c_str());
	reader->Update();

	// 检查是否有错误发生
	if (reader->GetErrorCode() != 0)
	{
		throw runtime_error("Failed to read STL file");
	}
	// 创建多边形数据映射器
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(reader->GetOutputPort());

	// 创建演员对象并配置其属性
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// 设置材质属性
	actor->GetProperty()->SetDiffuse(0.8);
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetSpecular(0.3);
	actor->GetProperty()->SetSpecularPower(60.0);
	actor->GetProperty()->SetOpacity(opacity);
	return actor;

}

vtkSmartPointer<vtkPolyData> PedicleSurgeryPlanning::CreatePolyDataFromSTL(string stl_file)
{
	// 创建 STL 阅读器
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();

	// 设置 STL 文件路径
	reader->SetFileName(stl_file.c_str());

	// 执行读取操作
	reader->Update();
	if (reader->GetErrorCode() != 0)
	{
		throw runtime_error("Failed to read STL file");
	}
	// 获取多边形数据
	vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput();
	/*cout << "There are " << poly_data->GetNumberOfPoints() << " points." << endl;
	cout << "There are " << poly_data->GetNumberOfPolys() << " polygons." << endl;*/
	auto num_polys = poly_data->GetNumberOfPolys();

	if (num_polys > PolyDataDownSampleNumPolyThreshold)
	{
		clock_t start = clock();
		double reduction = 1.0 - double(PolyDataDownSampleNumPolyThreshold) / double(num_polys);
		auto decimate = vtkSmartPointer<vtkDecimatePro>::New();
		decimate->SetInputData(poly_data);
		decimate->SetTargetReduction(reduction);
		decimate->PreserveTopologyOn();
		decimate->Update();

		poly_data = decimate->GetOutput();
		clock_t end = clock();
		double duration = double(end - start) / CLOCKS_PER_SEC;
		cout << "Down sample stl used " << duration << endl;
	}
	return poly_data;
}

vtkSmartPointer<vtkActor> PedicleSurgeryPlanning::CreateActorFromPolyData(vtkSmartPointer<vtkPolyData> poly_data, const string &color, double opacity)
{
	// 创建颜色对象
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// 创建多边形数据映射器
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(poly_data);

	// 创建演员对象并配置其属性
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// 设置材质属性
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

vtkSmartPointer<vtkActor> PedicleSurgeryPlanning::CreateCylinderActor(const vector<float>& point0, const vector<float>& point1,
	const string& color, float opacity, float radius)
{
	auto colors = vtkSmartPointer<vtkNamedColors>::New();
	auto line = vtkSmartPointer<vtkLineSource>::New();
	line->SetPoint1(point0[0], point0[1], point0[2]);
	line->SetPoint2(point1[0], point1[1], point1[2]);

	auto tube_filter = vtkSmartPointer<vtkTubeFilter>::New();
	tube_filter->SetInputConnection(line->GetOutputPort());
	tube_filter->SetRadius(radius);
	tube_filter->SetNumberOfSides(100);
	tube_filter->CappingOn();
	tube_filter->Update();

	auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(tube_filter->GetOutput());
	auto actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

vtkSmartPointer<vtkActor> PedicleSurgeryPlanning::CreatePediclePipelineCylinderActor(const vector<float>& point0, const vector<float>& point1,
	float pedicle_pipleline_rate, float radius, const string& color)
{
	auto p0 = point0;
	vector<float> p1 = { float(p0[0] + (point1[0] - point0[0]) * pedicle_pipleline_rate),
		float(p0[1] + (point1[1] - point0[1]) * pedicle_pipleline_rate),
		float(p0[2] + (point1[2] - point0[2]) * pedicle_pipleline_rate),
	};

	auto cylinder_actor = CreateCylinderActor(p0, p1, color, 1.0, radius);
	return cylinder_actor;
}

vtkSmartPointer<vtkActor> PedicleSurgeryPlanning::CreateSphereActor(vector<float>& point, float radius, float opacity, const string &color)
{
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
	vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
	sphere->SetCenter(point[0], point[1], point[2]);
	sphere->SetRadius(radius);
	sphere->SetPhiResolution(100);
	sphere->SetThetaResolution(100);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(sphere->GetOutputPort());

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

vector<vtkSmartPointer<vtkActor>> PedicleSurgeryPlanning::CreatePointsActor(const vector<float>& points, float radius, float opacity, const string &color)
{
	vector<vtkSmartPointer<vtkActor>> points_actor;
	size_t num_points = points.size() / 3;
	for (size_t i = 0; i < num_points; i++)
	{
		vector<float> cur_point;
		cur_point.push_back(points[i * 3 + 0]);
		cur_point.push_back(points[i * 3 + 1]);
		cur_point.push_back(points[i * 3 + 2]);
		vtkSmartPointer<vtkActor> cur_point_actor = CreateSphereActor(cur_point, radius, opacity, color);
		points_actor.push_back(cur_point_actor);
	}
	return points_actor;
}

Eigen::MatrixXd PedicleSurgeryPlanning::GetPointsFromSTL(string stl_file, int num_points)
{
	// 读取STL文件
	auto mesh = open3d::io::CreateMeshFromFile(stl_file);
	if (mesh->IsEmpty())
	{
		cout << "Mesh loaded successfully!" << endl;
		throw runtime_error("Error: Failed to load mesh from " + stl_file);
	}

	// 使用Poisson Disk Sampling从网格中采样点
	auto pcd = mesh->SamplePointsPoissonDisk(num_points);

	// 将采样点转换为Eigen矩阵
	Eigen::MatrixXd points_xyz(pcd->points_.size(), 3); // 创建 Nx3 的矩阵
	for (size_t i = 0; i < pcd->points_.size(); ++i)
	{
		points_xyz.row(i) = pcd->points_[i].transpose(); // 将每个点复制到矩阵的行
	}
	return points_xyz;
}

vector<float> PedicleSurgeryPlanning::MatrixToVector(const Eigen::MatrixXd& matrix)
{
	vector<float> vec;
	for (int i = 0; i < matrix.rows(); ++i)
	{
		for (int j = 0; j < matrix.cols(); ++j)
		{
			vec.push_back(matrix(i, j)); // 将元素添加到 vector 中
		}
	}
	return vec;
}

void PedicleSurgeryPlanning::CreateFinalActors()
{
	vector<vtkSmartPointer<vtkActor>> top_points_actor = CreatePointsActor(m_top_points, 0.5, 1.0, "Red");
	//vector<vtkSmartPointer<vtkActor>> left_points_actor = createPointsActor(m_left_points, 0.5, 1.0, "Green");
	//vector<vtkSmartPointer<vtkActor>> right_points_actor = createPointsActor(m_right_points, 0.5, 1.0, "Blue");
	vtkSmartPointer<vtkActor> spine_actor = CreateActorFromPolyData(m_spine_poly_data, "Cornsilk", 0.8);
	m_all_actors.insert(m_all_actors.end(), top_points_actor.begin(), top_points_actor.end());
	m_all_actors.push_back(spine_actor);


	auto left_cylinder_actor = CreatePediclePipelineCylinderActor(m_left_pedicle_start_point, m_left_pedicle_end_point, 1.0, PedicleScrewRadius, "magenta");
	auto right_cylinder_actor = CreatePediclePipelineCylinderActor(m_right_pedicle_start_point, m_right_pedicle_end_point, 1.0, PedicleScrewRadius, "yellow");
	m_all_actors.push_back(left_cylinder_actor);
	m_all_actors.push_back(right_cylinder_actor);

	auto left_bound_points_min_actors = CreatePointsActor(m_left_bound_points_min, 0.4, 0.6, "purple");
	auto right_bound_points_min_actors = CreatePointsActor(m_right_bound_points_min, 0.4, 0.6, "purple");

	m_all_actors.insert(m_all_actors.end(), left_bound_points_min_actors.begin(), left_bound_points_min_actors.end());
	m_all_actors.insert(m_all_actors.end(), right_bound_points_min_actors.begin(), right_bound_points_min_actors.end());

}

void PedicleSurgeryPlanning::ShowActors(vector<vtkSmartPointer<vtkActor>> actors, const string& window_name)
{
	// 创建渲染器
	vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();

	// 将所有演员添加到渲染器中
	for (auto actor : actors)
	{
		ren->AddActor(actor);
	}

	// 设置背景颜色为白色
	ren->SetBackground(1.0, 1.0, 1.0);

	// 创建并配置渲染窗口
	//vtkRenderWindow* win = vtkRenderWindow::New();
	vtkSmartPointer<vtkRenderWindow> win = vtkSmartPointer<vtkRenderWindow>::New();
	win->AddRenderer(ren);
	win->SetWindowName(window_name.c_str());

	// 创建交互器
	//vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
	vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	iren->SetRenderWindow(win);

	// 设置交互器样式为TrackballCamera风格
	//vtkInteractorStyleTrackballCamera* style = vtkInteractorStyleTrackballCamera::New();
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	iren->SetInteractorStyle(style);

	// 重置视图以适应所有演员
	ren->ResetCamera();

	// 渲染并启动交互循环
	win->Render();
	iren->Initialize();
	iren->Start();

}

void PedicleSurgeryPlanning::ShowPlanResult()
{
	if (m_all_actors.size() > 0)
	{
		ShowActors(m_all_actors,"plan result");
	}
}

void PedicleSurgeryPlanning::ShowAllActors(vector<vector<vtkSmartPointer<vtkActor>>>& all_actors, const string& window_name)
{
	// 创建渲染器
	vtkSmartPointer<vtkRenderer> left_ren = vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkRenderer> right_ren = vtkSmartPointer<vtkRenderer>::New();

	// 将所有演员添加到渲染器中
	for (auto actor : all_actors[0])
	{
		left_ren->AddActor(actor);
	}
	for (auto actor : all_actors[1])
	{
		right_ren->AddActor(actor);
	}

	double left_view_port[4] = { 0.0, 0.0, 0.5, 1.0 };
	double right_view_port[4] = { 0.5, 0.0, 1.0, 1.0 };

	left_ren->SetViewport(left_view_port);
	right_ren->SetViewport(right_view_port);


	// 设置背景颜色为白色
	left_ren->SetBackground(1.0, 1.0, 1.0);
	right_ren->SetBackground(0.5, 0.5, 0.5);

	// 创建并配置渲染窗口
	//vtkRenderWindow* win = vtkRenderWindow::New();
	vtkSmartPointer<vtkRenderWindow> win = vtkSmartPointer<vtkRenderWindow>::New();
	win->AddRenderer(left_ren);
	win->AddRenderer(right_ren);
	win->SetWindowName(window_name.c_str());

	// 创建交互器
	//vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
	vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	iren->SetRenderWindow(win);

	// 设置交互器样式为TrackballCamera风格
	//vtkInteractorStyleTrackballCamera* style = vtkInteractorStyleTrackballCamera::New();
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	iren->SetInteractorStyle(style);

	// 重置视图以适应所有演员
	left_ren->ResetCamera();
	right_ren->ResetCamera();

	// 渲染并启动交互循环
	win->Render();
	iren->Initialize();
	iren->Start();
}

void PedicleSurgeryPlanning::FitPlaneFromPointsBySVD(vector<float>& fit_plane_center, vector<float>& fit_plane_normal, const vector<float>& points)
{
	size_t num_points = points.size() / 3;
	Eigen::MatrixXd cloud(num_points, 3);
	for (size_t i = 0; i < num_points; i++)
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

	// 计算协方差矩阵
	Eigen::MatrixXd covariance = demean.transpose() * demean / (demean.rows() - 1);

	// 对协方差矩阵进行SVD分解，只计算瘦的V矩阵
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinV);

	// 提取法线向量（最小奇异值对应的右奇异向量）
	Eigen::Vector3d normal;
	normal = svd.matrixV().col(2);

	fit_plane_normal.push_back(normal[0]);
	fit_plane_normal.push_back(normal[1]);
	fit_plane_normal.push_back(normal[2]);

	fit_plane_center.push_back(centroid[0]);
	fit_plane_center.push_back(centroid[1]);
	fit_plane_center.push_back(centroid[2]);
}

vector<float> PedicleSurgeryPlanning::NormalizeVector(const vector<float>& point)
{
	float x = point[0];
	float y = point[1];
	float z = point[2];

	float r = sqrt(x * x + y * y + z * z);
	vector<float> normal_point = { x / r, y / r, z / r };
	return normal_point;
}


vector<float> PedicleSurgeryPlanning::NormalizeVector(const vector<float>& point0, const vector<float>& point1)
{
	float x = point0[0] - point1[0];
	float y = point0[1] - point1[1];
	float z = point0[2] - point1[2];

	vector<float> point = { x, y, z };
	return NormalizeVector(point);
}


float PedicleSurgeryPlanning::GetTwoVectorDotValue(const vector<float>& normal0, const vector<float>& normal1)
{
	float value = normal0[0] * normal1[0] + normal0[1] * normal1[1] + normal0[2] * normal1[2];
	return value;
}

float PedicleSurgeryPlanning::CalculateAngle(const vector<float>& normal0, const vector<float>& normal1)
{
	float dot_product = GetTwoVectorDotValue(normal0, normal1);
	float norm_v1 = sqrt(normal0[0] * normal0[0] + normal0[1] * normal0[1] + normal0[2] * normal0[2]);
	float norm_v2 = sqrt(normal1[0] * normal1[0] + normal1[1] * normal1[1] + normal1[2] * normal1[2]);
	float cos_angle = dot_product / (norm_v1 * norm_v2);
	float angle_deg = acos(cos_angle) * (180.0 / PI);
	return angle_deg;
}

float PedicleSurgeryPlanning::GetAreaOfClosedCurvePoints(vector<float>& points)
{
	double area = 0.0;
	double p0[3];
	p0[0] = points[0];
	p0[1] = points[1];
	p0[2] = points[2];

	for (int i = 1; i < (points.size() / 3 - 2); i++)
	{
		double p1[3], p2[3];
		p1[0] = points[3 * i + 0];
		p1[1] = points[3 * i + 1];
		p1[2] = points[3 * i + 2];

		p2[0] = points[3 * (i + 1) + 0];
		p2[1] = points[3 * (i + 1) + 1];
		p2[2] = points[3 * (i + 1) + 2];

		area += vtkTriangle::TriangleArea(p0, p1, p2);
	}
	return float(area);
}

float PedicleSurgeryPlanning::GetDistanceOfTwoPoints(const vector<float>& point0, const vector<float>& point1)
{
	float dis = sqrt(pow((point0[0] - point1[0]), 2) + pow((point0[1] - point1[1]), 2) + pow((point0[2] - point1[2]), 2));
	return dis;
}

vector<float> PedicleSurgeryPlanning::GetDistanceOfPoints2AimPoint(const vector<vector<float>>& points, const vector<float>& aim_point)
{
	vector<float> distances;
	float x0, y0, z0;
	x0 = aim_point[0];
	y0 = aim_point[1];
	z0 = aim_point[2];

	for (int i = 0; i < points.size(); i++)
	{
		float cur_x, cur_y, cur_z;
		cur_x = points[i][0];
		cur_y = points[i][1];
		cur_z = points[i][2];
		float cur_dis = sqrt(pow((cur_x - x0), 2) + pow((cur_y - y0), 2) + pow((cur_z - z0), 2));
		distances.push_back(cur_dis);
	}
	return distances;
}
/*
	Func:
		此函数的功能是获取与当前拟合平面与polydata相切的切面的连通区域中个，距离当前拟合平面中心点最近的那个连通区域。
		拟合的平面与polydata相切，得到的切面可能有多个连通区域，需要从这些连通区域中找到正确的那个连通区域。
		需要对这些连通区域进行判断。判断的依据是，连通区域的中心点与拟合平面的中心点最近的那个连通区域。
	Input:
		center:拟合平面的中心点
		normal:拟合平面的法向
	Output:
		bound_points:切面的轮廓点
		center_new：新的拟合平面的中心点
	Author:BigPanda
	Date:2025.03.26
*/
void PedicleSurgeryPlanning::GetClipedCenterPoints(vector<float>& bound_points, vector<float>& center_new, float& cut_plane_area,
	 const vector<float>& center, const vector<float>& normal)
{
	vtkSmartPointer<vtkPlane> cut_plane = vtkSmartPointer<vtkPlane>::New();
	double plane_origin[3] = { center[0], center[1], center[2] };
	double plane_normal[3] = { normal[0], normal[1], normal[2] };
	cut_plane->SetOrigin(plane_origin);
	cut_plane->SetNormal(plane_normal);

	vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
	cutter->SetCutFunction(cut_plane);
	cutter->SetInputData(m_spine_poly_data);

	vtkSmartPointer<vtkStripper> stripper = vtkSmartPointer<vtkStripper>::New();
	stripper->SetInputConnection(cutter->GetOutputPort());
	stripper->JoinContiguousSegmentsOn();
	stripper->Update();

	vtkSmartPointer<vtkPoints> points = stripper->GetOutput()->GetPoints();

	vtkSmartPointer<vtkCellArray> cells = stripper->GetOutput()->GetLines();
	cells->InitTraversal();


	vtkSmartPointer<vtkIdList> indices = vtkSmartPointer<vtkIdList>::New();

	vector<vector<float>> all_line_points;
	vector<vector<float>> all_line_centers;
	vector<float> all_line_areas;

	while (cells->GetNextCell(indices))
	{
		vector<float> cur_line_points;
		int cur_line_points_count = 0;
		for (int i = 0; i < indices->GetNumberOfIds(); i++)
		{
			double *cur_point = points->GetPoint(indices->GetId(i));
			cur_line_points.push_back(cur_point[0]);
			cur_line_points.push_back(cur_point[1]);
			cur_line_points.push_back(cur_point[2]);
			cur_line_points_count += 1;
		}
		if (cur_line_points_count < 10)
			continue;
		all_line_points.push_back(cur_line_points);

		vector<float> cur_line_center = GetPointsMean(cur_line_points);
		all_line_centers.push_back(cur_line_center);
		all_line_areas.push_back(GetAreaOfClosedCurvePoints(cur_line_points));
	}
	vector<float> dis = GetDistanceOfPoints2AimPoint(all_line_centers, center);
	int min_index = 0;
	float min_dis = 99999.0;
	for (int i = 0; i < dis.size(); i++)
	{
		if (dis[i] < min_dis)
		{
			min_dis = dis[i];
			min_index = i;
		}
	}
	if (all_line_points.size() > 0)
	{
		bound_points = all_line_points[min_index];
		center_new = all_line_centers[min_index];
		cut_plane_area = all_line_areas[min_index];
	}
}

/*
	Func:
		此函数的功能是获取拟合的左、右椎弓根平面的正确的法向。
		由于平面拟合得到的法向具有二向性，为了确定正确的平面法向，
		通过计算拟合的平面法向与参考法向（椎弓根特征点中心点到椎体顶面
		特征点的中心点的连线方向）之间的值，来判断拟合的平面法向的正确
		与否。

	Input:
		points:椎弓根特征点
	Output:
		plane_center：基于上述椎弓根特征点拟合得到的平面的中心点
		plane_normal：基于上述椎弓根特征点拟合得到的正确的平面法向
*/
void PedicleSurgeryPlanning::GetTheTrueFitPlaneNormal(const vector<float>& points, vector<float>& plane_center, vector<float>& plane_normal)
{
	vector<float> fit_plane_center;
	vector<float> fit_plane_normal;
	//拟合得到平面的法向，但该法向方向不确定，需要与参考法向进行比对，确认
	PedicleSurgeryPlanning::FitPlaneFromPointsBySVD(fit_plane_center, fit_plane_normal, points);
	plane_center = fit_plane_center;

	vector<float> reference_normal = PedicleSurgeryPlanning::NormalizeVector(m_top_points_center, plane_center);
	if (GetTwoVectorDotValue(reference_normal, fit_plane_normal) < 0.0f)
	{
		plane_normal.push_back(-fit_plane_normal[0]);
		plane_normal.push_back(-fit_plane_normal[1]);
		plane_normal.push_back(-fit_plane_normal[2]);
	}
	else
	{
		plane_normal = fit_plane_normal;
	}

	float angle = CalculateAngle(reference_normal, plane_normal);
	if (angle > CrossAngleThreshold)
	{
		plane_normal = reference_normal;
	}
}

void PedicleSurgeryPlanning::GetTheTrueFitPlaneCenter(vector<float>& cut_plane_center, vector<float>& cut_plane_normal, vector<float>& bound_points_min)
{
	vector<float> cur_cut_center;
	vector<float> cur_bound_points;
	float cur_cut_area = 0.0;
	GetClipedCenterPoints(cur_bound_points, cur_cut_center, cur_cut_area, cut_plane_center, cut_plane_normal);

	if (GetDistanceOfTwoPoints(cut_plane_center, cur_cut_center) < TwoCentersDistanceThreshold)
	{
		cut_plane_center = cur_cut_center;
		bound_points_min = cur_bound_points;
	}
}
/*
	Func:
		此函数的功能是计算平面外一定，在平面上的投影点
	Input:
		source_point: 平面外的一点,xyz
		plane_normal: 平面法向量, xyz
		plane_center: 平面法向量与平面的交点, xyz
	Output:
		projected_point: 平面外一点在该平面上的投影点,xyz,
	Reference : https://blog.csdn.net/fsac213330/article/details/53219949
		https://blog.csdn.net/soaryy/article/details/82884691
	Author:BigPanda
	Date:2025.03.26
	*/
vector<float> PedicleSurgeryPlanning::GetProjectedPointOnPlane(const vector<float>& source_point, const vector<float>& plane_normal, const vector<float>& plane_center)
{
	double eps = 1.0e-8;
	double A = plane_normal[0];
	double B = plane_normal[1];
	double C = plane_normal[2];
	double r = A * A + B * B + C * C;
	assert(r > eps);
	double D = -(A*plane_center[0] + B * plane_center[1] + C * plane_center[2]);
	double x0 = source_point[0];
	double y0 = source_point[1];
	double z0 = source_point[2];

	float x_projected = ((B*B + C * C)*x0 - A * (B*y0 + C * z0 + D)) / r;
	float y_projected = ((A*A + C * C)*y0 - B * (A*x0 + C * z0 + D)) / r;
	float z_projected = ((A*A + B * B)*z0 - C * (A*x0 + B * y0 + D)) / r;
	vector<float> projected_point = { x_projected, y_projected, z_projected };
	return projected_point;
}

vector<float> PedicleSurgeryPlanning::GetTwoVectorCrossValue(const vector<float>& vector0, const vector<float>& vector1)
{
	double x0 = vector0[0];
	double y0 = vector0[1];
	double z0 = vector0[2];

	double x1 = vector1[0];
	double y1 = vector1[1];
	double z1 = vector1[2];

	float x = y0 * z1 - z0 * y1;
	float y = z0 * x1 - x0 * z1;
	float z = x0 * y1 - y0 * x1;
	vector<float> cross_value = { x, y, z };
	return cross_value;
}

/*
	Func:
		此函数的功能是，生成绕指定轴旋转的旋转矩阵，生成的旋转矩阵是右乘矩阵，point_new = point * Matrix.
		左乘矩阵和右乘矩阵的定义:根据相乘时矩阵在哪一侧，该矩阵就称为对应侧的矩阵.
	Input:
		rotate_normal:旋转轴
		rotate_angle:旋转角度
	Output:
		R:旋转矩阵
	Author:BigPanda
	Date:2025.03.26
*/
vector<vector<float>> PedicleSurgeryPlanning::CreateRotateMatrixAroundNormal(const vector<float>& rotate_normal, float rotate_angle)
{
	vector<float> r = NormalizeVector(rotate_normal);
	float rotated_rad = rotate_angle / 180.0 * PI;

	float R_x = r[0];
	float R_y = r[1];
	float R_z = r[2];

	float cos_alpha = cos(rotated_rad);
	float sin_alpha = sin(rotated_rad);

	vector<float> R0 = { float(cos_alpha + pow(R_x, 2) * (1.0 - cos_alpha)),
							 float(R_x * R_y * (1.0 - cos_alpha) - R_z * sin_alpha),
							 float(R_x * R_z * (1.0 - cos_alpha) + R_y * sin_alpha) };

	vector<float> R1 = { float(R_y * R_x * (1.0 - cos_alpha) + R_z * sin_alpha),
							 float(cos_alpha + pow(R_y, 2) * (1.0 - cos_alpha)),
							 float(R_y * R_z * (1.0 - cos_alpha) - R_x * sin_alpha) };

	vector<float> R2 = { float(R_z * R_x * (1.0 - cos_alpha) - R_y * sin_alpha),
							 float(R_z * R_y * (1.0 - cos_alpha) + R_x * sin_alpha),
							 float(cos_alpha + pow(R_z, 2) * (1.0 - cos_alpha)) };
	vector<vector<float>> R = { R0, R1, R2 };

	return R;
}
/*
	Func：
		此函数的功能是计算向量与旋转矩阵，经过变换后的向量。
	Input:
		normal:待旋转的向量
		rotate_matrix：旋转矩阵，该矩阵为行向量。
	Output:
		normal_new：旋转后的向量
*/

vector<float> PedicleSurgeryPlanning::GetVectorDotMatrixValue(const vector<float>& normal, const vector<vector<float>>& rotate_matrix)
{
	float x = normal[0];
	float y = normal[1];
	float z = normal[2];

	float x_new = x * rotate_matrix[0][0] + y * rotate_matrix[1][0] + z * rotate_matrix[2][0];
	float y_new = x * rotate_matrix[0][1] + y * rotate_matrix[1][1] + z * rotate_matrix[2][1];
	float z_new = x * rotate_matrix[0][2] + y * rotate_matrix[1][2] + z * rotate_matrix[2][2];
	vector<float> normal_new = { x_new, y_new, z_new };
	return normal_new;
}

/*
	Func:
		对切面绕着某一轴在一定范围内旋转，计算最小的那个截面。旋转向量target_normal绕着旋转轴rotate_normal旋转
	Input:
		rotate_normal : 旋转轴
		target_normal : 旋转向量(单位向量)
		target_center : 旋转向量的原点
		max_rotate_angle : 最大的搜索角度（ - rotate_angle, rotate_angle，1），递增角度为1度
		dis_threshold: 两个中心点（截面的中心点和初始的中心点）之间的允许的最大距离
		area_threshold: 截面面积阈值，小于这个面积的不考虑
	Output:
		cut_plane_area_min：最小截面面积
		bound_points_min：边解点
		rotate_matrix_min：旋转矩阵

	Author:BigPanda
	Date:2025.03.26
*/
void PedicleSurgeryPlanning::GetTheMinCutPlaneArea(float& cut_plane_area_min, vector<float>& bound_points_min,
	vector<vector<float>>& rotate_matrix_min,vector<float>& center_min, const vector<float>& rotate_normal,
	const vector<float>& target_normal,const vector<float>& target_center, float max_rotate_angle, 
	float dis_threshold, float area_threshold,int bound_points_threshold)
{
	vector<float> cut_plane_areas;
	vector<vector<vector<float>>> rotate_matrixs;
	vector<vector<float>> bound_points;
	vector<vector<float>> centers;

	for (int i = int(-max_rotate_angle); i< int(max_rotate_angle); i += 4)
	{
		float cur_rotate_angle = i;
		vector<vector<float>> cur_rotate_matrix = CreateRotateMatrixAroundNormal(rotate_normal, cur_rotate_angle);
		vector<float> target_normal_new = GetVectorDotMatrixValue(target_normal, cur_rotate_matrix);

		vector<float> cur_bound_points;
		vector<float> cur_fit_center_point;
		float cur_cut_plane_area = 0.0;
		GetClipedCenterPoints(cur_bound_points, cur_fit_center_point, cur_cut_plane_area,
			target_center, target_normal_new);

		if (GetDistanceOfTwoPoints(cur_fit_center_point, target_center) > dis_threshold) { continue; }
		if (cur_bound_points.size() < bound_points_threshold * 3) { continue; }
		if (cur_cut_plane_area < area_threshold) { continue; }


		rotate_matrixs.push_back(cur_rotate_matrix);
		cut_plane_areas.push_back(cur_cut_plane_area);
		bound_points.push_back(cur_bound_points);
		centers.push_back(cur_fit_center_point);

	}

	if (cut_plane_areas.size() > 0)
	{
		float min_area = 99999.0;
		int min_index = 0;
		for (int i = 0; i < cut_plane_areas.size(); i++)
		{
			float cur_area = cut_plane_areas[i];
			if (cur_area < min_area)
			{
				min_area = cur_area;
				min_index = i;
			}
		}
		cut_plane_area_min = cut_plane_areas[min_index];
		bound_points_min = bound_points[min_index];
		rotate_matrix_min = rotate_matrixs[min_index];
		center_min = centers[min_index];
	}
}

/*
	Func:
		对切面沿着某一条轴，在一定范围内移动，计算最小的那个截面。

	Output:
		cut_plane_area:最小截面的面积
		bound_points_min:最小界面的轮廓点
		center_min:最小截面的中心点
	Input:
		target_center : 截面初始的中心点
		target_normal : 截面的法向量
		target_target_poly_data : 脊柱的stl
		max_step : 沿着normal方向移动 - max_step到max_step
	Author:BigPanda
	Date:2025.03.26
	*/
void PedicleSurgeryPlanning::GetTheMinCutPlaneAreaAlongAxisY(float& cut_plane_area_min, vector<float>& bound_points_min, vector<float>& center_min,
	const vector<float>& target_center, const vector<float>& target_normal, float max_step)
{
	vector<float> all_cut_plane_areas;
	vector<vector<float>> all_bound_points;
	vector<vector<float>> all_centers;
	float cur_step = -max_step;
	while (cur_step <= max_step)
	{
		vector<float> cur_center = { target_center[0] + cur_step * target_normal[0],
										 target_center[1] + cur_step * target_normal[1],
										 target_center[2] + cur_step * target_normal[2] };
		vector<float> cur_bound_points;
		vector<float> cur_center_new;
		float cur_cut_plane_area = 0.0;
		GetClipedCenterPoints(cur_bound_points, cur_center_new, cur_cut_plane_area, cur_center, target_normal);
		cur_step += 0.5; //每个0.5mm递增

		if (cur_bound_points.size() < 10 * 3) { continue; }
		all_cut_plane_areas.push_back(cur_cut_plane_area);
		all_bound_points.push_back(cur_bound_points);
		all_centers.push_back(cur_center_new);
	}

	float min_area = 99999.0;
	int min_index = 0;
	for (int i = 0; i < all_cut_plane_areas.size(); i++)
	{
		float cur_area = all_cut_plane_areas[i];
		if (cur_area < min_area)
		{
			min_area = cur_area;
			min_index = i;
		}
	}
	cut_plane_area_min = all_cut_plane_areas[min_index];
	bound_points_min = all_bound_points[min_index];
	center_min = all_centers[min_index];
}

float PedicleSurgeryPlanning::GetAngleOfCutPlaneNormalAndSpineAxisNormalX(const vector<float>& cut_plane_center, const vector<float>& cut_plane_normal,
	const vector<float>& axis_origin, const vector<float>& axis_normalZ, const vector<float>& axis_normalX)
{
	vector<float> point2 = { float(cut_plane_center[0] + 10.0 * cut_plane_normal[0]),
		float(cut_plane_center[1] + 10.0 * cut_plane_normal[1]),
		float(cut_plane_center[2] + 10.0 * cut_plane_normal[2]),
	};
	auto project_point1 = cut_plane_center;
	auto project_point2 = GetProjectedPointOnPlane(point2, axis_normalZ, cut_plane_center);
	auto project_normal = NormalizeVector(project_point2, project_point1);
	float angle = CalculateAngle(project_normal, axis_normalX);
	return angle;
}

/*
	Func:
		此函数的功能是椎弓根通道的方向，具体做法是，已知脊柱的坐标系，将脊柱坐标系的x轴方向绕着z轴旋转一定角度，得到的
		 方向就是椎弓根通道的方向。
	Input:
		rotate_angle:旋转角度
		axis_normalX:
		axis_normalY:
		axis_normalZ:

	Output:
		normal:椎弓根通道法向量

	Author:BigPanda
	Date:2025.03.26
*/
vector<float> PedicleSurgeryPlanning::CreatePediclePipelineNormal(float rotate_angle, const vector<float>& axis_normalX,
	const vector<float>& aixs_normalY, const vector<float>& axis_normalZ)
{
	auto rotate_matrix = CreateRotateMatrixAroundNormal(axis_normalZ, rotate_angle);
	auto pedicle_pipeline_normal = GetVectorDotMatrixValue(axis_normalX, rotate_matrix);
	return pedicle_pipeline_normal;
}

/*
	Func:
		此函数的功能是计算椎弓根通道与椎体相交的两个交点

	Input:
		point0:椎弓根通道方向上的点0
		point1:椎弓根通道方向上的点1

	Output:

	Author:BigPanda
	Date:2025.03.26
*/
vector<vector<float>> PedicleSurgeryPlanning::GetIntersectPointsFromLineAndPolyData(vector<float>& point0, vector<float>& point1)
{
	auto tree = vtkSmartPointer<vtkOBBTree>::New();
	tree->SetDataSet(m_spine_poly_data);
	tree->BuildLocator();

	auto intersect_points = vtkSmartPointer<vtkPoints>::New();
	double p0[3] = { point0[0], point0[1], point0[2] };
	double p1[3] = { point1[0], point1[1], point1[2] };
	tree->IntersectWithLine(p0, p1, intersect_points, nullptr);

	auto num_points = intersect_points->GetNumberOfPoints();

	vector<vector<float>> all_points;
	for (int i = 0; i < num_points; i++)
	{
		double p[3];
		intersect_points->GetPoint(i, p);
		vector<float> cur_point = { float(p[0]), float(p[1]), float(p[2]) };
		all_points.push_back(cur_point);
	}
	return all_points;
}


/*
	Func:
		此函数的功能是，计算椎弓根通道与椎体的所有交点中，离椎弓根通道中心点最近的两个点

	Input:
		all_points:所有的交点
		source_point:目标点（椎弓根通道中心点）
		reference_normal:参考的向量（对最近的两个点按照该方向进行排序）

	Output:
		point0:目标点0
		point1:目标点1
	Author:BigPanda
	Date:2025.02.14
*/
void PedicleSurgeryPlanning::GetTheClosedTwoPoints(vector<float>& point0, vector<float>& point1, const vector<vector<float>>& all_points,
	const vector<float>& source_point, const vector<float>& reference_normal)
{
	if (all_points.size() < 2)
	{
		throw "Vector must contain at least two elements.";
	}
	vector<float> all_dis;
	float min_dis = 99999.0;
	float min_index = 0;


	for (int i = 0; i < all_points.size(); i++)
	{
		auto cur_point = all_points[i];
		auto cur_dis = GetDistanceOfTwoPoints(cur_point, source_point);
		all_dis.push_back(cur_dis);
		if (cur_dis < min_dis) { min_dis = cur_dis; min_index = i; }
	}

	float sec_min_dis = 99999.0;
	int sec_min_index = 0;
	for (int i = 0; i < all_points.size(); i++)
	{
		auto cur_dis = all_dis[i];
		auto p0 = all_points[min_index];
		auto p1 = all_points[i];
		vector<float> p0_normal = { float(p0[0] - source_point[0]), float(p0[1] - source_point[1]), float(p0[2] - source_point[2]) };
		vector<float> p1_normal = { float(p1[0] - source_point[0]), float(p1[1] - source_point[1]), float(p1[2] - source_point[2]) };
		auto dot_value = GetTwoVectorDotValue(p0_normal, p1_normal);

		if (cur_dis > min_dis && cur_dis < sec_min_dis && dot_value < 0.0)
		{
			sec_min_dis = cur_dis;
			sec_min_index = i;
		}
	}

	auto p0 = all_points[min_index];
	auto p1 = all_points[sec_min_index];

	if (reference_normal.size() > 0)
	{
		vector<float> p = { float(p0[0] - source_point[0]), float(p0[1] - source_point[1]), float(p0[2] - source_point[2]) };
		if (GetTwoVectorDotValue(p, reference_normal) > 0.0)
		{
			point0 = p1;
			point1 = p0;
		}
		else
		{
			point0 = p0;
			point1 = p1;
		}
	}
	else
	{
		point0 = p0;
		point1 = p1;
	}
}
/*
	Func：
		此函数的功能是计算两个点之间的相减之后得到的新的点
	Input: 
		pointsA:点A
		pointsB:点B
		factor:点B乘以的系数
	Output:
		pointsC: pointsC = pointsA - pointsB*factor;
*/
vector<float> PedicleSurgeryPlanning::GetTheMinusOfTwoPoints(const vector<float>& pointsA, const vector<float>& pointsB, float factor)
{
	//vector<float> pointsC = { pointsA[0] - pointsB[0], pointsA[1] - pointsB[1], pointsA[2] - pointsB[2] };
	assert(pointsA.size() == pointsB.size());
	vector<float> pointsC = pointsA;
	for (int i = 0; i < pointsA.size(); i++)
	{
		pointsC[i] = pointsA[i] - pointsB[i] * factor;
	}
	return pointsC;
}
vector<float> PedicleSurgeryPlanning::GetTheMeanOfTwoPoints(const vector<float>& pointsA, const vector<float>& pointsB)
{
	assert(pointsA.size() == pointsB.size());
	vector<float> pointsC = pointsA;
	for (int i = 0; i < pointsA.size(); i++)
	{
		pointsC[i] = float((pointsA[i] + pointsB[i]) / 2.0);
	}
	return pointsC;
}
void PedicleSurgeryPlanning::ReverseNormal(vector<float>& normal)
{
	for (int i = 0; i < normal.size(); i++)
	{
		normal[i] = -1.0*normal[i];
	}
}


void PedicleSurgeryPlanning::Planning()
{
	m_left_bound_points_min = m_left_points;
	m_right_bound_points_min = m_right_points;

	m_top_points_center = GetPointsMean(m_top_points);
	m_left_points_center = GetPointsMean(m_left_points);
	m_right_points_center = GetPointsMean(m_right_points);

	GetTheTrueFitPlaneNormal(m_left_points, m_left_cut_plane_center, m_left_cut_plane_normal);
	GetTheTrueFitPlaneNormal(m_right_points, m_right_cut_plane_center, m_right_cut_plane_normal);
	GetTheTrueFitPlaneCenter(m_left_cut_plane_center, m_left_cut_plane_normal, m_left_bound_points_min);
	GetTheTrueFitPlaneCenter(m_right_cut_plane_center, m_right_cut_plane_normal, m_right_bound_points_min);

	
	vector<float> right_center_point_projected_on_left_plane;
	right_center_point_projected_on_left_plane = GetProjectedPointOnPlane(m_right_cut_plane_center, m_left_cut_plane_normal, m_left_cut_plane_center);
	m_left_axis_normalX = NormalizeVector(right_center_point_projected_on_left_plane, m_left_cut_plane_center);
	m_left_axis_normalY = m_left_cut_plane_normal;
	m_left_axis_normalZ = GetTwoVectorCrossValue(m_left_axis_normalX, m_left_axis_normalY);


	vector<float> left_center_point_projected_on_right_plane;
	left_center_point_projected_on_right_plane = GetProjectedPointOnPlane(m_left_cut_plane_center, m_right_cut_plane_normal, m_right_cut_plane_center);
	m_right_axis_normalX = NormalizeVector(left_center_point_projected_on_right_plane, m_right_cut_plane_center);
	m_right_axis_normalY = m_right_cut_plane_normal;
	m_right_axis_normalZ = GetTwoVectorCrossValue(m_right_axis_normalX, m_right_axis_normalY);


	//############## step1_1: rotate around X, left side  ###################
	float left_cut_plane_area_min1 = 0.0;
	vector<float> left_bound_points_min1;
	vector<vector<float>> left_rotate_matrix_min1;
	vector<float> left_center_min1;

	GetTheMinCutPlaneArea(left_cut_plane_area_min1, left_bound_points_min1, left_rotate_matrix_min1, left_center_min1,
		m_left_axis_normalX, m_left_axis_normalY, m_left_cut_plane_center, SearchRotateAngle, TwoCentersDistanceThreshold,
		CutPlaneAreaThreshold, CutBoundPointsThreshold);
	
	//############## step1_2: rotate around X, right side  ###################
	float right_cut_plane_area_min1 = 0.0;
	vector<float> right_bound_points_min1;
	vector<vector<float>> right_rotate_matrix_min1;
	vector<float> right_center_min1;

	GetTheMinCutPlaneArea(right_cut_plane_area_min1, right_bound_points_min1, right_rotate_matrix_min1, right_center_min1,
		m_right_axis_normalX, m_right_axis_normalY, m_right_cut_plane_center, SearchRotateAngle, TwoCentersDistanceThreshold,
		CutPlaneAreaThreshold, CutBoundPointsThreshold);
	
	//############step1_3:  更新椎弓根坐标系、中心点和边界点  ##############

	if (left_rotate_matrix_min1.size() > 0)
	{
		m_left_axis_normalY = GetVectorDotMatrixValue(m_left_axis_normalY, left_rotate_matrix_min1);
		m_left_axis_normalZ = GetVectorDotMatrixValue(m_left_axis_normalZ, left_rotate_matrix_min1);
		m_left_cut_plane_center = left_center_min1;
		m_left_bound_points_min = left_bound_points_min1;
	}
	if (right_rotate_matrix_min1.size() > 0)
	{
		m_right_axis_normalY = GetVectorDotMatrixValue(m_right_axis_normalY, right_rotate_matrix_min1);
		m_right_axis_normalZ = GetVectorDotMatrixValue(m_right_axis_normalZ, right_rotate_matrix_min1);
		m_right_cut_plane_center = right_center_min1;
		m_right_bound_points_min = right_bound_points_min1;
	}

	//############## step2_1: rotate around Z, left side  ###############
	float left_cut_plane_area_min2 = 0.0;
	vector<float> left_bound_points_min2;
	vector<vector<float>> left_rotate_matrix_min2;
	vector<float> left_center_min2;

	GetTheMinCutPlaneArea(left_cut_plane_area_min2, left_bound_points_min2, left_rotate_matrix_min2, left_center_min2,
		m_left_axis_normalZ, m_left_axis_normalY, m_left_cut_plane_center, SearchRotateAngle, TwoCentersDistanceThreshold,
		CutPlaneAreaThreshold, CutBoundPointsThreshold);

	//############## step2_2: rotate around Z, right side  ###############
	float right_cut_plane_area_min2 = 0.0;
	vector<float> right_bound_points_min2;
	vector<vector<float>> right_rotate_matrix_min2;
	vector<float> right_center_min2;
	GetTheMinCutPlaneArea(right_cut_plane_area_min2, right_bound_points_min2, right_rotate_matrix_min2, right_center_min2,
		m_right_axis_normalZ, m_right_axis_normalY, m_right_cut_plane_center,SearchRotateAngle, TwoCentersDistanceThreshold,
		CutPlaneAreaThreshold, CutBoundPointsThreshold);

	//############step2_3:  更新椎弓根坐标系和中心点  ##############
	if (left_rotate_matrix_min2.size() > 0)
	{
		m_left_axis_normalX = GetVectorDotMatrixValue(m_left_axis_normalX, left_rotate_matrix_min2);
		m_left_axis_normalY = GetVectorDotMatrixValue(m_left_axis_normalY, left_rotate_matrix_min2);
		m_left_cut_plane_center = left_center_min2;
		m_left_bound_points_min = left_bound_points_min2;
	}

	if (right_rotate_matrix_min1.size() > 0)
	{
		m_right_axis_normalX = GetVectorDotMatrixValue(m_right_axis_normalX, right_rotate_matrix_min2);
		m_right_axis_normalY = GetVectorDotMatrixValue(m_right_axis_normalY, right_rotate_matrix_min2);
		m_right_cut_plane_center = right_center_min2;
	}

	m_left_cut_plane_normal = m_left_axis_normalY;
	m_right_cut_plane_normal = m_right_axis_normalY;

	//############step3_1: 沿着y轴方向搜索最小截面 ##############
	vector<float> left_bound_points_min3;
	vector<float> left_center_min3;
	float left_cut_plane_area_min3 = 0.0;
	GetTheMinCutPlaneAreaAlongAxisY(left_cut_plane_area_min3, left_bound_points_min3,
		left_center_min3, m_left_cut_plane_center, m_left_cut_plane_normal);

	vector<float> right_bound_points_min3;
	vector<float> right_center_min3;
	float right_cut_plane_area_min3 = 0.0;
	GetTheMinCutPlaneAreaAlongAxisY(right_cut_plane_area_min3, right_bound_points_min3,
		right_center_min3, m_right_cut_plane_center, m_right_cut_plane_normal);

	//############step3_2: 更新截面中心点   ##############
	if (GetDistanceOfTwoPoints(left_center_min3, m_left_cut_plane_center) < TwoCentersDistanceThreshold)
	{
		m_left_cut_plane_center = left_center_min3;
		m_left_bound_points_min = left_bound_points_min3;
	}

	if (GetDistanceOfTwoPoints(right_center_min3, m_right_cut_plane_center) < TwoCentersDistanceThreshold)
	{
		m_right_cut_plane_center = right_center_min3;
		m_right_bound_points_min = right_bound_points_min3;
	}

	//############## step4: 检查左、右椎弓根坐标系的方向  #############
	auto tmp_left = GetTheMinusOfTwoPoints(m_top_points_center, m_left_cut_plane_center);
	float left_flag = GetTwoVectorDotValue(tmp_left, m_left_axis_normalY);

	if (left_flag < 0.0f)
	{
		ReverseNormal(m_left_axis_normalZ);
		ReverseNormal(m_left_axis_normalY);
	}

	auto tmp_right = GetTheMinusOfTwoPoints(m_top_points_center, m_right_cut_plane_center);
	float right_flag = GetTwoVectorDotValue(tmp_right, m_right_axis_normalY);

	if (right_flag > 0.0f)
	{
		ReverseNormal(m_right_axis_normalZ);
	}
	else
	{
		ReverseNormal(m_right_axis_normalY);
	}

	//step5:建立脊柱坐标系，脊柱坐标系的Z方向是，top_points拟合的平面的法向方向
	FitPlaneFromPointsBySVD(m_top_plane_center, m_top_plane_normal, m_top_points);
	vector<float> left_right_axis_normalZ_mean = GetTheMeanOfTwoPoints(m_left_axis_normalZ, m_right_axis_normalZ);
	
	if (GetTwoVectorDotValue(m_top_plane_normal, left_right_axis_normalZ_mean) < 0.0f)
	{
		ReverseNormal(m_top_plane_normal);
	}

	if (CalculateAngle(m_top_plane_normal, left_right_axis_normalZ_mean) > CrossAngleThreshold)
	{
		m_spine_axis_normalZ = left_right_axis_normalZ_mean;
	}
	else
	{
		auto tmp = GetTheMeanOfTwoPoints(m_left_axis_normalZ, m_right_axis_normalZ);
		m_spine_axis_normalZ = GetTheMeanOfTwoPoints(tmp, m_top_plane_normal);
	}
	auto tmp = NormalizeVector(m_spine_axis_normalZ);
	m_spine_axis_normalZ = tmp;

	m_spine_axis_center = GetTheMeanOfTwoPoints(m_left_cut_plane_center, m_right_cut_plane_center);

	auto left_project_center = GetProjectedPointOnPlane(m_left_cut_plane_center, m_spine_axis_normalZ, m_spine_axis_center);
	auto right_project_center = GetProjectedPointOnPlane(m_right_cut_plane_center, m_spine_axis_normalZ, m_spine_axis_center);

	m_spine_axis_normalY = NormalizeVector(left_project_center, right_project_center);
	m_spine_axis_normalX = GetTwoVectorCrossValue(m_spine_axis_normalY, m_spine_axis_normalZ);

	/*auto spine_axis_center_actor = CreateSphereActor(m_spine_axis_center, 1.0, 1.0, "red");
	auto spine_axis_actors = CreateAxisActors(m_spine_axis_center, m_spine_axis_normalX, m_spine_axis_normalY, m_spine_axis_normalZ);*/


	//############ step6_1:计算椎弓根通道法向  #############
	float rot_angle = PedicleCrossAngle;
	float left_angle = GetAngleOfCutPlaneNormalAndSpineAxisNormalX(m_left_cut_plane_center, m_left_cut_plane_normal,
		m_spine_axis_center, m_spine_axis_normalZ, m_spine_axis_normalX);
	if (left_angle < rot_angle) { left_angle = rot_angle; }

	float right_angle = GetAngleOfCutPlaneNormalAndSpineAxisNormalX(m_right_cut_plane_center, m_right_cut_plane_normal,
		m_spine_axis_center, m_spine_axis_normalZ, m_spine_axis_normalX);
	if (right_angle < rot_angle) { right_angle = rot_angle; }

	m_pedicle_pipeline_L_normal = CreatePediclePipelineNormal(left_angle, m_spine_axis_normalX, m_spine_axis_normalY, m_spine_axis_normalZ);
	m_pedicle_pipeline_R_normal = CreatePediclePipelineNormal(-right_angle, m_spine_axis_normalX, m_spine_axis_normalY, m_spine_axis_normalZ);

	vector<float> left_point0 = GetTheMinusOfTwoPoints(m_left_cut_plane_center, m_pedicle_pipeline_L_normal, 50.0);
	vector<float> left_point1 = GetTheMinusOfTwoPoints(m_left_points_center, m_pedicle_pipeline_L_normal, -70.0);

	vector<float> right_point0 = GetTheMinusOfTwoPoints(m_right_cut_plane_center, m_pedicle_pipeline_R_normal, 50.0);
	vector<float> right_point1 = GetTheMinusOfTwoPoints(m_right_cut_plane_center, m_pedicle_pipeline_R_normal, -70.0);

	//############ step6_2:计算椎弓根通道法向与椎体交点  #############
	auto left_intersect_points = GetIntersectPointsFromLineAndPolyData(left_point0, left_point1);
	vector<float> left_intersect_point0;
	vector<float> left_intersect_point1;

	if (left_intersect_points.size() < 2)
	{
		cout << "left side instersect points is less than two!" << endl;
		m_plan_result = false;
	}
	else
	{
		GetTheClosedTwoPoints(left_intersect_point0, left_intersect_point1, left_intersect_points, m_left_cut_plane_center, m_pedicle_pipeline_L_normal);
		/*auto left_intersect_point0_actor = createSphereActor(left_intersect_point0, 1.0, 1.0, "magenta");
		auto left_intersect_point1_actor = createSphereActor(left_intersect_point1, 1.0, 1.0, "magenta");
		all_actors.push_back(left_intersect_point0_actor);
		all_actors.push_back(left_intersect_point1_actor);*/

		m_left_pedicle_start_point = left_intersect_point0;
		auto cur_point = GetTheMinusOfTwoPoints(left_intersect_point1, left_intersect_point0);
		m_left_pedicle_end_point = GetTheMinusOfTwoPoints(left_intersect_point0, cur_point, -0.8);

		//auto left_cylinder_actor = createPediclePipelineCylinderActor(left_intersect_point0, left_intersect_point1, 0.8, 3.5 / 2.0, "magenta");
	}


	auto right_intersect_points = GetIntersectPointsFromLineAndPolyData(right_point0, right_point1);
	vector<float> right_intersect_point0;
	vector<float> right_intersect_point1;
	if (right_intersect_points.size() < 2)
	{
		cout << "right side intersect points is less than two!" << endl;
		m_plan_result = false;
	}
	else
	{
		GetTheClosedTwoPoints(right_intersect_point0, right_intersect_point1, right_intersect_points, m_right_cut_plane_center, m_pedicle_pipeline_R_normal);
		/*auto right_intersect_point0_actor = createSphereActor(right_intersect_point0, 1.0, 1.0, "yellow");
		auto right_intersect_point1_actor = createSphereActor(right_intersect_point1, 1.0, 1.0, "yellow");
		all_actors.push_back(right_intersect_point0_actor);
		all_actors.push_back(right_intersect_point1_actor);*/

		//auto right_cylinder_actor = CreatePediclePipelineCylinderActor(right_intersect_point0, right_intersect_point1, PediclePiplelineRate, 3.5 / 2.0, "yellow");
		//all_actors.push_back(right_cylinder_actor);
		m_right_pedicle_start_point = right_intersect_point0;
		auto cur_point = GetTheMinusOfTwoPoints(right_intersect_point1, right_intersect_point0);
		m_right_pedicle_end_point = GetTheMinusOfTwoPoints(right_intersect_point0, cur_point, -0.8);
	}
}

void PedicleSurgeryPlanning::SaveSpineSurgeryPlanning2Png(const string& save_png_file)
{
	auto ren = vtkSmartPointer<vtkRenderer>::New();
	for (int i = 0; i < m_all_actors.size(); i++)
	{
		ren->AddActor(m_all_actors[i]);
	}
	auto win = vtkSmartPointer<vtkRenderWindow>::New();
	win->AddRenderer(ren);
	win->SetWindowName("show spine");
	win->SetSize(600, 600);
	win->SetMultiSamples(4);

	auto iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	iren->SetRenderWindow(win);

	auto style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	iren->SetInteractorStyle(style);

	auto camera = vtkSmartPointer<vtkCamera>::New();
	double pos[3];
	pos[0] = m_spine_axis_center[0] + 10.0 * m_spine_axis_normalZ[0];
	pos[1] = m_spine_axis_center[1] + 10.0 * m_spine_axis_normalZ[1];
	pos[2] = m_spine_axis_center[2] + 10.0 * m_spine_axis_normalZ[2];

	camera->SetPosition(pos[0], pos[1], pos[2]);
	camera->SetViewUp(-m_spine_axis_normalX[0], -m_spine_axis_normalX[1], -m_spine_axis_normalX[2]);
	camera->SetFocalPoint(m_spine_axis_center[0], m_spine_axis_center[1], m_spine_axis_center[2]);
	camera->ComputeViewPlaneNormal();
	camera->Zoom(2.0);
	ren->SetActiveCamera(camera);
	ren->ResetCamera();
	win->ShowWindowOff();
	win->Render();

	auto w2if1 = vtkSmartPointer<vtkWindowToImageFilter>::New();
	w2if1->SetInput(win);
	w2if1->SetInputBufferTypeToRGB();
	w2if1->ReadFrontBufferOff();
	w2if1->Update();

	double pos2[3];
	pos2[0] = m_spine_axis_center[0] - 5.0 * m_spine_axis_normalZ[0] - 10.0 * m_spine_axis_normalY[0];
	pos2[1] = m_spine_axis_center[1] - 5.0 * m_spine_axis_normalZ[1] - 10.0 * m_spine_axis_normalY[1];
	pos2[2] = m_spine_axis_center[2] - 5.0 * m_spine_axis_normalZ[2] - 10.0 * m_spine_axis_normalY[2];

	camera->SetPosition(pos2[0], pos2[1], pos2[2]);
	camera->SetFocalPoint(m_spine_axis_center[0] - 5.0 * m_spine_axis_normalZ[0], 
						  m_spine_axis_center[1] - 5.0 * m_spine_axis_normalZ[1], 
		                  m_spine_axis_center[2] - 5.0 * m_spine_axis_normalZ[2]);
	ren->SetActiveCamera(camera);
	ren->ResetCamera();

	win->Render();
	auto w2if2 = vtkSmartPointer<vtkWindowToImageFilter>::New();
	w2if2->SetInput(win);
	w2if2->SetInputBufferTypeToRGB();
	w2if2->ReadFrontBufferOff();
	w2if2->Update();

	auto image_append = vtkSmartPointer<vtkImageAppend>::New();
	image_append->SetInputConnection(w2if1->GetOutputPort());
	image_append->AddInputConnection(w2if2->GetOutputPort());

	auto writer = vtkSmartPointer<vtkPNGWriter>::New();
	writer->SetFileName(save_png_file.c_str());
	writer->SetInputConnection(image_append->GetOutputPort());
	writer->Write();
}

PedicleSurgeryPlanning::~PedicleSurgeryPlanning()
{

}