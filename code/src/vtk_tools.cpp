#include "vtk_tools.h"
#include "loadonnx.h"
#include <fstream>
#include <sstream>
#include <cmath>
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
#include <vtkPNGWriter.h>
#include <vtkImageAppend.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkLandmarkTransform.h>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <limits>

#include <vtkAutoInit.h>
#include <vtkRegularPolygonSource.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2) // VTK was built with vtkRenderingOpenGL2
VTK_MODULE_INIT(vtkInteractionStyle)
VTK_MODULE_INIT(vtkRenderingFreeType)

std::vector<std::string> list_directory(const std::string& dirPath, bool recursive )
{
	std::vector<std::string> items;

	try 
	{
#if defined(_WIN32)
		// Windows implementation using FindFirstFile/FindNextFile
		HANDLE hFind;
		WIN32_FIND_DATAA data;

		std::string searchPath = dirPath + "\\*";
		hFind = FindFirstFileA(searchPath.c_str(), &data);

		if (hFind == INVALID_HANDLE_VALUE) 
		{
			throw std::runtime_error("Error accessing directory: " + dirPath);
		}

		do 
		{
			if (std::string(data.cFileName) != "." && std::string(data.cFileName) != "..") 
			{
				items.push_back(std::string(data.cFileName));

				// For recursive traversal, check if the entry is a directory
				if (recursive && (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) 
				{
					std::string subDir = dirPath + "\\" + data.cFileName;
					std::vector<std::string> subItems = list_directory(subDir, true);
					for (const auto& item : subItems) 
					{
						items.push_back(std::string(data.cFileName) + "\\" + item);
					}
				}
			}
		} while (FindNextFileA(hFind, &data));

		FindClose(hFind);

#elif defined(__linux__) || defined(__APPLE__)
		// POSIX implementation using opendir/readdir
		DIR* dir = opendir(dirPath.c_str());

		if (!dir) 
		{
			throw runtime_error("Error accessing directory: " + dirPath);
		}

		struct dirent* entry;
		while ((entry = readdir(dir)) != nullptr) 
		{
			string name = entry->d_name;

			if (name == "." || name == "..") 
			{
				continue;
			}

			items.push_back(name);

			// For recursive traversal, check if the entry is a directory
			if (recursive && (entry->d_type & DT_DIR)) 
			{
				string subDir = dirPath + "/" + name;
				vector<string> subItems = list_directory(subDir, true);
				for (const auto& item : subItems) 
				{
					items.push_back(name + "/" + item);
				}
			}
		}
		closedir(dir);
#endif

	}
	catch (const std::exception& e) 
	{
		cerr << "Exception: " << e.what() << endl;
		return std::vector<std::string>();
	}
	return items;
}

std::vector<float> pointCloudNormalize(const std::vector<float>& points)
{
	std::vector<float> points_normal(points);
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


std::vector<float> matrixToVector(const Eigen::MatrixXd& matrix)
{
	std::vector<float> vec;
	for (int i = 0; i < matrix.rows(); ++i)
	{
		for (int j = 0; j < matrix.cols(); ++j)
		{
			vec.push_back(matrix(i, j)); // 将元素添加到 vector 中
		}
	}
	return vec;
}


std::vector<float> randomChoice(const std::vector<float>& spine_points, int num_points, bool replace)
{
	std::vector<float> result;
	std::vector<size_t> indices(spine_points.size() / 3);
	std::iota(indices.begin(), indices.end(), 0); // 填充索引 0, 1, 2, ..., spine_points.size()/3-1

	// 随机数生成器
	std::random_device rd;
	std::mt19937 g(rd());

	if (replace)
	{
		// 允许重复采样
		std::uniform_int_distribution<> dist(0, spine_points.size() / 3 - 1);
		for (int i = 0; i < num_points; ++i)
		{
			int idx = dist(g);
			result.push_back(spine_points[idx * 3 + 0]);
			result.push_back(spine_points[idx * 3 + 1]);
			result.push_back(spine_points[idx * 3 + 2]);
		}
	}
	else
	{
		// 不允许重复采样
		if (num_points > spine_points.size() / 3)
		{
			throw std::invalid_argument("num_points cannot be greater than spine_points.size() when replace is false");
		}
		std::shuffle(indices.begin(), indices.end(), g);
		for (int i = 0; i < num_points; ++i)
		{
			result.push_back(spine_points[indices[i] * 3 + 0]);
			result.push_back(spine_points[indices[i] * 3 + 1]);
			result.push_back(spine_points[indices[i] * 3 + 2]);
		}
	}
	return result;
}


Eigen::MatrixXd getPointsFromSTL(std::string stl_file, int num_points)
{
	// 读取STL文件
	auto mesh = open3d::io::CreateMeshFromFile(stl_file);
	if (mesh->IsEmpty())
	{
		std::cout << "Mesh loaded successfully!" << std::endl;
		throw std::runtime_error("Error: Failed to load mesh from " + stl_file);
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


vtkSmartPointer<vtkPolyData> createPolyDataFromSTL(const std::string& stl_file)
{
	// 创建 STL 阅读器
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();

	// 设置 STL 文件路径
	reader->SetFileName(stl_file.c_str());

	// 执行读取操作
	reader->Update();
	if (reader->GetErrorCode() != 0)
	{
		throw std::runtime_error("Failed to read STL file");
	}
	// 获取多边形数据
	vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput();
	/*std::cout << "There are " << poly_data->GetNumberOfPoints() << " points." << std::endl;
	std::cout << "There are " << poly_data->GetNumberOfPolys() << " polygons." << std::endl;*/
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
		std::cout << "Down sample stl used " << duration << std::endl;
	}
	return poly_data;

}

vtkSmartPointer<vtkActor> createActorFromPolyData(vtkSmartPointer<vtkPolyData> polydata, const vtkStdString &color, double opacity)
{
	// 创建颜色对象
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// 创建多边形数据映射器
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);

	// 创建演员对象并配置其属性
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// 设置材质属性
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

vtkSmartPointer<vtkActor> createActorFromSTL(const std::string& stlFile, const vtkStdString &color, double opacity)
{
	// 创建颜色对象
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// 创建并配置STL读取器
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(stlFile.c_str());
	reader->Update();

	// 检查是否有错误发生
	if (reader->GetErrorCode() != 0) 
	{
		throw std::runtime_error("Failed to read STL file");
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

void showActors(std::vector<vtkSmartPointer<vtkActor>> actors, const std::string& window_name) 
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
vtkSmartPointer<vtkActor> createCirclePlaneActor(const std::vector<float>& plane_center,
	const std::vector<float>& plane_normal, float radius, float opacity, const vtkStdString &color)
{
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
	vtkSmartPointer<vtkRegularPolygonSource> polygon_source = vtkSmartPointer<vtkRegularPolygonSource>::New();
	
	polygon_source->GeneratePolygonOff();
	polygon_source->SetNumberOfSides(500);
	polygon_source->SetRadius(radius);
	polygon_source->SetGeneratePolygon(true);
	polygon_source->SetCenter(plane_center[0], plane_center[1], plane_center[2]);
	polygon_source->SetNormal(plane_normal[0], plane_normal[1], plane_normal[2]);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(polygon_source->GetOutputPort());

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

vtkSmartPointer<vtkActor> createSphereActor(std::vector<float>& point, float radius, float opacity, const vtkStdString &color)
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

std::vector<vtkSmartPointer<vtkActor>> createPointsActor(std::vector<float>& points, float radius, float opacity, const vtkStdString &color)
{
	std::vector<vtkSmartPointer<vtkActor>> points_actor;
	int num_points = points.size()/3;
	for (int i = 0; i < num_points; i++)
	{
		std::vector<float> cur_point;
		cur_point.push_back(points[i * 3 + 0]);
		cur_point.push_back(points[i * 3 + 1]);
		cur_point.push_back(points[i * 3 + 2]);
		vtkSmartPointer<vtkActor> cur_point_actor = createSphereActor(cur_point, radius, opacity, color);
		points_actor.push_back(cur_point_actor);
	}
	return points_actor;
}

std::vector<float> getAimPoints(const std::vector<float>& points, const std::vector<int>& labels, SPINE_POINT_LABEL aim_label)
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

std::vector<float> getPointsMean(const std::vector<float>& points)
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
	std::vector<float> mean_point;
	mean_point.push_back(x);
	mean_point.push_back(y);
	mean_point.push_back(z);

	return mean_point;
}
void fitPlaneFromPointsBySVD(std::vector<float>& fit_plane_center, std::vector<float>& fit_plane_normal, const std::vector<float>& points)
{
	size_t num_points = points.size()/3;
	Eigen::MatrixXd cloud(num_points, 3);

	for (size_t i = 0; i < num_points; ++i) 
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
	Eigen::MatrixXd covariance = demean.transpose() * demean/(demean.rows()-1);

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


vtkSmartPointer<vtkActor> fitPlaneActorFromPoints(std::vector<float>& fit_plane_center,	
	std::vector<float>& fit_plane_normal, std::vector<float> points, const vtkStdString &color, float radius)
{
	fitPlaneFromPointsBySVD(fit_plane_center, fit_plane_normal, points);
	vtkSmartPointer<vtkActor> fit_plane_actor = vtkSmartPointer<vtkActor>::New();
	fit_plane_actor = createCirclePlaneActor(fit_plane_center, fit_plane_normal, radius, 0.9, color);
	return fit_plane_actor;
}

std::vector<float> normalizeVector(const std::vector<float>& point)
{
	float x = point[0];
	float y = point[1];
	float z = point[2];

	float r = sqrt(x * x + y * y + z * z);
	std::vector<float> normal_point = { x / r, y / r, z / r };
	return normal_point;
}

std::vector<float> normalizeVector(const std::vector<float>& point0, const std::vector<float>& point1)
{

	float x = point0[0] - point1[0];
	float y = point0[1] - point1[1];
	float z = point0[2] - point1[2];

	std::vector<float> point = {x, y, z};
	return normalizeVector(point);
}

float getTwoVectorDotValue(const std::vector<float>& normal0, const std::vector<float>& normal1)
{
	float value = normal0[0]* normal1[0] + normal0[1]*normal1[1]+normal0[2]*normal1[2];
	return value;
}

float calculateAngle(const std::vector<float>& normal0, const std::vector<float>& normal1)
{
	float dot_product = getTwoVectorDotValue(normal0, normal1);
	float norm_v1 = sqrt( normal0[0] * normal0[0] + normal0[1] * normal0[1] + normal0[2] * normal0[2]);
	float norm_v2 = sqrt( normal1[0] * normal1[0] + normal1[1] * normal1[1] + normal1[2] * normal1[2]);
	float cos_angle = dot_product / (norm_v1 * norm_v2);

	float angle_deg = acos(cos_angle) * (180.0 / PI);

	return angle_deg;
}

vtkSmartPointer<vtkActor> createLineActorByPoints(const std::vector<float>& point0, const std::vector<float>& point1, float line_width, const std::string& color)
{
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	float p0[3] = { point0[0], point0[1], point0[2] };
	float p1[3] = { point1[0], point1[1], point1[2] };
	points->InsertNextPoint(p0);
	points->InsertNextPoint(p1);

	vtkSmartPointer<vtkPolyLine> poly_line = vtkSmartPointer<vtkPolyLine>::New();
	poly_line->GetPointIds()->SetNumberOfIds(2);

	for (int i = 0; i < 2; i++)
	{
		poly_line->GetPointIds()->SetId(i, i);
	}

	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	cells->InsertNextCell(poly_line);

	vtkSmartPointer<vtkPolyData> poly_data = vtkSmartPointer<vtkPolyData>::New();
	poly_data->SetPoints(points);
	poly_data->SetLines(cells);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(poly_data);

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();

	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetLineWidth(line_width);
	return actor;
}


vtkSmartPointer<vtkActor> createLineActorByNormal(const std::vector<float>& point0, const std::vector<float>& normal, float length, float line_width, const std::string& color)
{
	std::vector<float> point1;
	float x = point0[0] + length * normal[0];
	float y = point0[1] + length * normal[1];
	float z = point0[2] + length * normal[2];

	point1.push_back(x);
	point1.push_back(y);
	point1.push_back(z);

	return createLineActorByPoints(point0, point1, line_width, color);
}


void getClipedCenterPoints(std::vector<float>& bound_points, std::vector<float>& center_new, float& cut_plane_area,
	const std::vector<float>& center, const std::vector<float>& normal,	vtkSmartPointer<vtkPolyData> target_poly_data)
{
	vtkSmartPointer<vtkPlane> cut_plane = vtkSmartPointer<vtkPlane>::New();
	double plane_origin[3] = {center[0], center[1], center[2]};
	double plane_normal[3] = { normal[0], normal[1], normal[2] };
	cut_plane->SetOrigin(plane_origin);
	cut_plane->SetNormal(plane_normal);

	vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
	cutter->SetCutFunction(cut_plane);
	cutter->SetInputData(target_poly_data);

	vtkSmartPointer<vtkStripper> stripper = vtkSmartPointer<vtkStripper>::New();
	stripper->SetInputConnection(cutter->GetOutputPort());
	stripper->JoinContiguousSegmentsOn();
	stripper->Update();

	vtkSmartPointer<vtkPoints> points = stripper->GetOutput()->GetPoints();

	vtkSmartPointer<vtkCellArray> cells = stripper->GetOutput()->GetLines();
	cells->InitTraversal();


	vtkSmartPointer<vtkIdList> indices = vtkSmartPointer<vtkIdList>::New();

	std::vector<std::vector<float>> all_line_points;
	std::vector<std::vector<float>> all_line_centers;
	std::vector<float> all_line_areas;

	while (cells->GetNextCell(indices))
	{
		std::vector<float> cur_line_points;
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

		std::vector<float> cur_line_center = getPointsMean(cur_line_points);
		all_line_centers.push_back(cur_line_center);
		all_line_areas.push_back(getAreaOfClosedCurvePoints(cur_line_points));
	}
	std::vector<float> dis = getDistanceOfPoints2AimPoint(all_line_centers, center);
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


float getAreaOfClosedCurvePoints(std::vector<float>& points)
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

std::vector<float> getDistanceOfPoints2AimPoint(const std::vector<std::vector<float>>& points, const std::vector<float>& aim_point)
{
	std::vector<float> distances;
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

float getDistanceOfTwoPoints(const std::vector<float>& point0, const std::vector<float>& point1)
{
	float dis = sqrt(pow((point0[0] - point1[0]), 2) + pow((point0[1] - point1[1]), 2) + pow((point0[2] - point1[2]), 2));
	return dis;
}

std::vector<float> getProjectedPointOnPlane(const std::vector<float>& source_point, const std::vector<float>& plane_normal, const std::vector<float>& plane_center)
{
	/*
		:param source_point: 平面外的一点,xyz
		:param plane_normal: 平面法向量, xyz
		:param plane_center: 平面法向量与平面的交点, xyz
		:return : projected_point: 平面外一点在该平面上的投影点,xyz,
		method reference : https://blog.csdn.net/fsac213330/article/details/53219949
		https://blog.csdn.net/soaryy/article/details/82884691
	*/
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
	std::vector<float> projected_point = {x_projected, y_projected, z_projected};
	return projected_point;
}

std::vector<float> getTwoVectorCrossValue(const std::vector<float>& vector0, const std::vector<float>& vector1)
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
	std::vector<float> cross_value = {x, y, z};
	return cross_value;
}

std::vector<vtkSmartPointer<vtkActor>> createAxisActors(const std::vector<float>& axis_origin, const std::vector<float>& axis_normalX,
	const std::vector<float>& axis_normalY, const std::vector<float>& axis_normalZ)
{
	vtkSmartPointer<vtkActor> axis_normalX_actor = createLineActorByNormal( axis_origin, axis_normalX, 30.0, 3.0, "Red");
	vtkSmartPointer<vtkActor> axis_normalY_actor = createLineActorByNormal( axis_origin, axis_normalY, 30.0, 3.0, "Green");
	vtkSmartPointer<vtkActor> axis_normalZ_actor = createLineActorByNormal( axis_origin, axis_normalZ, 30.0, 3.0, "Blue");
	std::vector<vtkSmartPointer<vtkActor>> axis_actors = { axis_normalX_actor, axis_normalY_actor, axis_normalZ_actor };
	return axis_actors;
}

void getTheMinCutPlaneArea(float& cut_plane_area_min, std::vector<float>& bound_points_min, std::vector<std::vector<float>>& rotate_matrix_min,
	std::vector<float>& center_min, const std::vector<float>& rotate_normal, const std::vector<float>& target_normal, 
	const std::vector<float>& target_center, vtkSmartPointer<vtkPolyData> target_poly_data, float max_rotate_angle,
	float dis_threshold, float area_threshold, int bound_points_threshold) 
{
	/*
		func:对切面绕着某一轴在一定范围内旋转，计算最小的那个截面。旋转向量target_normal绕着旋转轴rotate_normal旋转
		rotate_normal : 旋转轴
		target_normal : 旋转向量(单位向量)
		target_center : 旋转向量的原点
		target_poly_data : 脊柱的stl，计算平面（target_center, target_normal）与stl切的切面
		max_rotate_angle : 最大的搜索角度（ - rotate_angle, rotate_angle，1），递增角度为1度
		dis_threshold: 两个中心点（截面的中心点和初始的中心点）之间的允许的最大距离
	*/
	std::vector<float> cut_plane_areas;
	std::vector<std::vector<std::vector<float>>> rotate_matrixs;
	std::vector<std::vector<float>> bound_points;
	std::vector<std::vector<float>> centers;

	for (int i = int(-max_rotate_angle); i< int(max_rotate_angle); i += 4)
	{
		float cur_rotate_angle = i;
		std::vector<std::vector<float>> cur_rotate_matrix = createRotateMatrixAroundNormal(rotate_normal, cur_rotate_angle);
		std::vector<float> target_normal_new = getVectorDotMatrixValue(target_normal, cur_rotate_matrix);
		
		std::vector<float> cur_bound_points;
		std::vector<float> cur_fit_center_point;
		float cur_cut_plane_area = 0.0;
		getClipedCenterPoints(cur_bound_points, cur_fit_center_point, cur_cut_plane_area,
			target_center, target_normal_new, target_poly_data);

		if (getDistanceOfTwoPoints(cur_fit_center_point, target_center) > dis_threshold) { continue; }
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

std::vector<std::vector<float>> createRotateMatrixAroundNormal(const std::vector<float>& rotate_normal, float rotate_angle)
{
	/*
	func:此函数的功能是，生成绕指定轴旋转的旋转矩阵，生成的旋转矩阵是右乘矩阵，point_new = point * Matrix.
	左乘矩阵和右乘矩阵的定义:根据相乘时矩阵在哪一侧，该矩阵就称为对应侧的矩阵.
	*/
	std::vector<float> r = normalizeVector(rotate_normal);
	float rotated_rad = rotate_angle / 180.0 * PI;

	float R_x = r[0];
	float R_y = r[1];
	float R_z = r[2];

	float cos_alpha = cos(rotated_rad);
	float sin_alpha = sin(rotated_rad);

	std::vector<float> R0 = { float(cos_alpha + pow(R_x, 2) * (1.0 - cos_alpha)),
							 float(R_x * R_y * (1.0 - cos_alpha) - R_z * sin_alpha),
							 float(R_x * R_z * (1.0 - cos_alpha) + R_y * sin_alpha) };
	
	std::vector<float> R1 = { float(R_y * R_x * (1.0 - cos_alpha) + R_z * sin_alpha),
							 float(cos_alpha + pow(R_y, 2) * (1.0 - cos_alpha)),
							 float(R_y * R_z * (1.0 - cos_alpha) - R_x * sin_alpha) };

	std::vector<float> R2 = { float(R_z * R_x * (1.0 - cos_alpha) - R_y * sin_alpha),
							 float(R_z * R_y * (1.0 - cos_alpha) + R_x * sin_alpha),
							 float(cos_alpha + pow(R_z, 2) * (1.0 - cos_alpha)) };
	std::vector<std::vector<float>> R = { R0, R1, R2 };

	return R;
}

//Eigen::Matrix3f createRotateMatrixAroundNormal2(Eigen::Vector3f normal, float angle)
//{
//	Eigen::Matrix3f m1;
//	Eigen::Vector3f p1;
//	p1 = m1 * p1;
//	return Eigen::AngleAxisf(angle * M_PI, normal.normalized());
//}

std::vector<float> getVectorDotMatrixValue(const std::vector<float>& normal, const std::vector<std::vector<float>>& rotate_matrix)
{
	float x = normal[0];
	float y = normal[1];
	float z = normal[2];

	float x_new = x * rotate_matrix[0][0] + y * rotate_matrix[1][0] + z * rotate_matrix[2][0];
	float y_new = x * rotate_matrix[0][1] + y * rotate_matrix[1][1] + z * rotate_matrix[2][1];
	float z_new = x * rotate_matrix[0][2] + y * rotate_matrix[1][2] + z * rotate_matrix[2][2];
	std::vector<float> normal_new = { x_new, y_new, z_new };
	return normal_new;
}

void getTheMinCutPlaneAreaAlongAxisY(float& cut_plane_area_min, std::vector<float>& bound_points_min, std::vector<float>& center_min,
	const std::vector<float>& target_center, const std::vector<float>& target_normal, vtkSmartPointer<vtkPolyData> target_poly_data, float max_step)
{
	/*
		func:对切面沿着某一条轴，在一定范围内移动，计算最小的那个截面。 

		return:
		cut_plane_area:最小截面的面积
		bound_points_min:最小界面的轮廓点
		center_min:最小截面的中心点

		input:
		target_center : 截面初始的中心点
		target_normal : 截面的法向量
		target_target_poly_data : 脊柱的stl
		max_step : 沿着normal方向移动 - max_step到max_step
	*/

	std::vector<float> all_cut_plane_areas;
	std::vector<std::vector<float>> all_bound_points;
	std::vector<std::vector<float>> all_centers;
	float cur_step = -max_step;
	while (cur_step <= max_step)
	{
		std::vector<float> cur_center = { target_center[0] + cur_step * target_normal[0],
										 target_center[1] + cur_step * target_normal[1],
										 target_center[2] + cur_step * target_normal[2] };
		std::vector<float> cur_bound_points;
		std::vector<float> cur_center_new;
		float cur_cut_plane_area = 0.0;
		getClipedCenterPoints(cur_bound_points, cur_center_new, cur_cut_plane_area, cur_center, target_normal, target_poly_data);
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

std::vector<float> createPediclePipelineNormal(float rotate_angle, const std::vector<float>& axis_normalX, const std::vector<float>& aixs_normalY,
	const std::vector<float>& axis_normalZ)
{
	/*
	func:此函数的功能是椎弓根通道的方向，具体做法是，已知脊柱的坐标系，将脊柱坐标系的x轴方向绕着z轴旋转一定角度，得到的
         方向就是椎弓根通道的方向。
	input:
	rotate_angle:旋转角度
	axis_normalX:
	axis_normalY:
	axis_normalZ:

	return:
	normal:椎弓根通道法向量
	*/
	auto rotate_matrix = createRotateMatrixAroundNormal(axis_normalZ, rotate_angle);
	auto pedicle_pipeline_normal = getVectorDotMatrixValue(axis_normalX, rotate_matrix);
	return pedicle_pipeline_normal;
}

std::vector<std::vector<float>> getIntersectPointsFromLineAndPolyData(std::vector<float>& point0, std::vector<float>& point1, vtkSmartPointer<vtkPolyData> poly_data)
{
	/*
	func:此函数的功能是计算椎弓根通道与椎体相交的两个交点
	author:BigPanda
	date:2025.02.14
	
	input: 
		point0:椎弓根通道方向上的点0
		point1:椎弓根通道方向上的点1
		target_poly_data:脊柱的polydata

	return:
		
	*/
	auto tree = vtkSmartPointer<vtkOBBTree>::New();
	tree->SetDataSet(poly_data);
	tree->BuildLocator();

	auto intersect_points = vtkSmartPointer<vtkPoints>::New();
	double p0[3] = { point0[0], point0[1], point0[2] };
	double p1[3] = { point1[0], point1[1], point1[2] };
	tree->IntersectWithLine(p0, p1, intersect_points, nullptr);

	auto num_points = intersect_points->GetNumberOfPoints();

	std::vector<std::vector<float>> all_points;
	for (int i = 0; i < num_points; i++)
	{
		double p[3];
		intersect_points->GetPoint(i, p);
		std::vector<float> cur_point = { float(p[0]), float(p[1]), float(p[2]) };
		all_points.push_back(cur_point);
	}
	return all_points;
}


void getTheClosedTwoPoints(std::vector<float>& point0, std::vector<float>& point1, const std::vector<std::vector<float>>& all_points,
	const std::vector<float>& source_point, const std::vector<float>& reference_normal)
{
	/*
	func:此函数的功能是，计算椎弓根通道与椎体的所有交点中，离椎弓根通道中心点最近的两个点
	author:BigPanda
	date:2025.02.14

	input:
		all_points:所有的交点
		source_point:目标点（椎弓根通道中心点）
		reference_normal:参考的向量（对最近的两个点按照该方向进行排序）

	return:
		point0:目标点0
		point1:目标点1
	*/
	if (all_points.size() < 2) 
	{
		throw "Vector must contain at least two elements.";
	}
	std::vector<float> all_dis;
	float min_dis = 99999.0;
	float min_index = 0;

	
	for (int i = 0; i < all_points.size(); i++)
	{
		auto cur_point = all_points[i];
		auto cur_dis = getDistanceOfTwoPoints(cur_point, source_point);
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
		std::vector<float> p0_normal = { float(p0[0] - source_point[0]), float(p0[1] - source_point[1]), float(p0[2] - source_point[2]) };
		std::vector<float> p1_normal = { float(p1[0] - source_point[0]), float(p1[1] - source_point[1]), float(p1[2] - source_point[2]) };
		auto dot_value = getTwoVectorDotValue(p0_normal, p1_normal);

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
		std::vector<float> p = { float(p0[0] - source_point[0]), float(p0[1] - source_point[1]), float(p0[2] - source_point[2]) };
		if (getTwoVectorDotValue(p, reference_normal) > 0.0)
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

vtkSmartPointer<vtkActor> createPediclePipelineCylinderActor(const std::vector<float>& point0, const std::vector<float>& point1,
	float pedicle_pipleline_rate, float radius, const std::string& color)
{
	auto p0 = point0;
	std::vector<float> p1 = { float(p0[0] + (point1[0] - point0[0]) * pedicle_pipleline_rate),
		float(p0[1] + (point1[1] - point0[1]) * pedicle_pipleline_rate),
		float(p0[2] + (point1[2] - point0[2]) * pedicle_pipleline_rate),
	};

	auto cylinder_actor = createCylinderActor(p0, p1, color, 1.0, radius);
	return cylinder_actor;
}

vtkSmartPointer<vtkActor> createCylinderActor(const std::vector<float>& point0, const std::vector<float>& point1,
	const std::string& color, float opacity , float radius)
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

void saveSpineSurgicalPlanning2Png(std::vector<vtkSmartPointer<vtkActor>> all_actors, const std::vector<float>& axis_origin,
	const std::vector<float>& axis_normalX, const std::vector<float>& axis_normalY, const std::vector<float>& axis_normalZ, std::string& save_png_file)
{
	auto ren = vtkSmartPointer<vtkRenderer>::New();
	for (int i = 0; i < all_actors.size(); i++)
	{
		ren->AddActor(all_actors[i]);
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
	pos[0] = axis_origin[0] + 10.0 * axis_normalZ[0];
	pos[1] = axis_origin[1] + 10.0 * axis_normalZ[1];
	pos[2] = axis_origin[2] + 10.0 * axis_normalZ[2];

	camera->SetPosition(pos[0], pos[1], pos[2]);
	camera->SetViewUp(-axis_normalX[0], -axis_normalX[1], -axis_normalX[2]);
	camera->SetFocalPoint(axis_origin[0], axis_origin[1], axis_origin[2]);
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
	pos2[0] = axis_origin[0] - 5.0 * axis_normalZ[0] - 10.0 * axis_normalY[0];
	pos2[1] = axis_origin[1] - 5.0 * axis_normalZ[1] - 10.0 * axis_normalY[1];
	pos2[2] = axis_origin[2] - 5.0 * axis_normalZ[2] - 10.0 * axis_normalY[2];

	camera->SetPosition(pos2[0], pos2[1], pos2[2]);
	camera->SetFocalPoint(axis_origin[0] - 5.0 * axis_normalZ[0], axis_origin[1] - 5.0 * axis_normalZ[1], axis_origin[2] - 5.0 * axis_normalZ[2]);
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

float getAngleOfCutPlaneNormalAndSpineAxisNormalX(const std::vector<float>& cut_plane_center, const std::vector<float>& cut_plane_normal,
	const std::vector<float>& axis_origin, const std::vector<float>& axis_normalZ, const std::vector<float>& axis_normalX)
{
	std::vector<float> point2 = { float(cut_plane_center[0] + 10.0 * cut_plane_normal[0]),
		float(cut_plane_center[1] + 10.0 * cut_plane_normal[1]),
		float(cut_plane_center[2] + 10.0 * cut_plane_normal[2]),
	};

	auto project_point1 = cut_plane_center;
	auto project_point2 = getProjectedPointOnPlane(point2, axis_normalZ, cut_plane_center);
	auto project_normal = normalizeVector(project_point2, project_point1);

	float angle = calculateAngle(project_normal, axis_normalX);
	
	return angle;
}
bool loadLandmarksFromFile(const std::string& landmark_file, std::vector<float>& top_points, std::vector<float>& left_points,
	std::vector<float>& right_points)
{
	std::ifstream input(landmark_file);
	if (!input.is_open())
	{
		std::cout << "error, failed to open " << landmark_file << std::endl;
		return false;
	}

	std::string line;
	while (std::getline(input, line))
	{
		std::stringstream ss(line);
		std::string value;
		std::vector<std::string> parts;
		while (getline(ss, value,','))
		{
			parts.push_back(value);
		}
		int label = stoi(parts[0]);
		float cur_x = stof(parts[1]);
		float cur_y = stof(parts[2]);
		float cur_z = stof(parts[3]);

		if (label == SPINE_POINT_LABEL::TOP - 1)
		{
			top_points.push_back(cur_x);
			top_points.push_back(cur_y);
			top_points.push_back(cur_z);
		}
		if (label == SPINE_POINT_LABEL::LEFT - 1)
		{
			left_points.push_back(cur_x);
			left_points.push_back(cur_y);
			left_points.push_back(cur_z);
		}

		if (label == SPINE_POINT_LABEL::RIGHT - 1)
		{
			right_points.push_back(cur_x);
			right_points.push_back(cur_y);
			right_points.push_back(cur_z);
		}
	}
	return true;
}
std::vector<float> getTheMinAxisPoint(const std::vector<float>& points)
{
	float minX = (std::numeric_limits<float>::max)();
	float minY = (std::numeric_limits<float>::max)();
	float minZ = (std::numeric_limits<float>::max)();
	for (int i = 0; i < points.size() / 3; i++)
	{
		if (minX > points[i * 3 + 0]) { minX = points[i * 3 + 0]; }
		if (minY > points[i * 3 + 1]) { minY = points[i * 3 + 1]; }
		if (minZ > points[i * 3 + 2]) { minZ = points[i * 3 + 2]; }
	}
	std::vector<float> min_point = { minX, minY, minZ };
	return min_point;
}

std::vector<float> getTheMaxAxisPoint(const std::vector<float>& points)
{
	float maxX = std::numeric_limits<float>::lowest();
	float maxY = std::numeric_limits<float>::lowest();
	float maxZ = std::numeric_limits<float>::lowest();
	for (int i = 0; i < points.size() / 3; i++)
	{
		if (maxX < points[i * 3 + 0]) { maxX = points[i * 3 + 0]; }
		if (maxY < points[i * 3 + 1]) { maxY = points[i * 3 + 1]; }
		if (maxZ < points[i * 3 + 2]) { maxZ = points[i * 3 + 2]; }
	}
	std::vector<float> max_point = { maxX, maxY, maxZ };
	return max_point;
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

std::vector<float> getPointsDotMatrix(const std::vector<float>& points, const std::vector<std::vector<float>>& matrix)
{
	std::vector<float> points_new = points;
	for (int i = 0; i < points.size() / 3; i++)
	{
		points_new[i * 3 + 0] = points[i * 3 + 0] * matrix[0][0] + points[i * 3 + 1] * matrix[0][1] + points[i * 3 + 2] * matrix[0][2];
		points_new[i * 3 + 1] = points[i * 3 + 0] * matrix[1][0] + points[i * 3 + 1] * matrix[1][1] + points[i * 3 + 2] * matrix[1][2];
		points_new[i * 3 + 2] = points[i * 3 + 0] * matrix[2][0] + points[i * 3 + 1] * matrix[2][1] + points[i * 3 + 2] * matrix[2][2];
	}
	return points_new;
}

std::vector<std::vector<float>> matrix4DotMatrix4(const std::vector<std::vector<float>>& matrix1, const std::vector<std::vector<float>>& matrix2)
{
	std::vector<std::vector<float>> matrix = matrix1;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			matrix[i][j] = matrix1[i][0] * matrix2[0][j] + matrix1[i][1] * matrix2[1][j] + matrix1[i][2] * matrix2[2][j] + matrix1[i][3] * matrix2[3][j];
		}
	}
	return matrix;
}

/*
	Func: 点云去中心
	Return:去中心后的点

	Author:BigPanda
	Date:2025.03.12

*/
std::vector<float> pointsDecenter(const std::vector<float>& points, const std::vector<float>& center)
{
	std::vector<float> points_new = points;
	for (int i = 0; i < points_new.size() / 3; i++)
	{
		points_new[i * 3 + 0] = points[i * 3 + 0] - center[0];
		points_new[i * 3 + 1] = points[i * 3 + 1] - center[1];
		points_new[i * 3 + 2] = points[i * 3 + 2] - center[2];
	}
	return points_new;
}

/*
	Func: PCA算法计算三维点云的PCA轴的特征值和特征向量
	Input: points，待计算PCA的电源
	Output: 特征值和对应的特征向量、点云的中心点
	
	Author:BigPanda
	Date:2025.03.12
*/

void PCA(const std::vector<float> points, std::vector<float>& eigen_values, std::vector<std::vector<float>>& eigen_vectors,
	std::vector<float>& points_center)
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
	Eigen::MatrixXd covariance = demean.transpose() * demean/(demean.rows());
	//std::cout << "\n协方差矩阵:\n" << covariance << std::endl;

	//计算特征值和特征向量
	Eigen::EigenSolver<Eigen::MatrixXd> eig(covariance);
	Eigen::MatrixXd eig_vectors = eig.eigenvectors().real();
	Eigen::MatrixXd eig_values = eig.eigenvalues().real();



	double values[3] = { eig_values(0), eig_values(1), eig_values(2) };
	int indexs[3] = { 0, 1, 2 };
	for (int i = 0; i < 3; i++)
	{
		for (int j = i+1; j < 3; j++)
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
	std::vector<float> normalX, normalY;
	for (int i = 0; i < 3; i++)
	{
		auto cur_index = indexs[i];
		eigen_values.push_back(eig_values(indexs[i],0));

		normal = eig_vectors.col(cur_index);
		if (i == 0)
		{
			normalX.push_back(float(normal[0]));
			normalX.push_back(float(normal[1]));
			normalX.push_back(float(normal[2]));
		}
		if (i == 1)
		{
			normalY.push_back(float(normal[0]));
			normalY.push_back(float(normal[1]));
			normalY.push_back(float(normal[2]));
		}
	}
	//std::cout << "\n特征值为:" << eigen_values[0] << "," << eigen_values[1] << "," << eigen_values[2] << std::endl;

   //特征向量统一使用右手坐标系
	auto normalZ = getTwoVectorCrossValue(normalX, normalY);
	eigen_vectors.push_back(normalX);
	eigen_vectors.push_back(normalY);
	eigen_vectors.push_back(normalZ);
}

void printMatrix(const std::vector<std::vector<float>>& matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
		{
			std::cout << matrix[i][j];
			if (j == (matrix[0].size() - 1))
			{
				std::cout << endl;
			}
			else
			{
				std::cout << ",";
			}
		}
	}
	//std::cout << matrix[0][0] << "," << matrix[0][1] << "," << matrix[0][2] << "," << matrix[0][3] << std::endl;
	//std::cout << matrix[1][0] << "," << matrix[1][1] << "," << matrix[1][2] << "," << matrix[1][3] << std::endl;
	//std::cout << matrix[2][0] << "," << matrix[2][1] << "," << matrix[2][2] << "," << matrix[2][3] << std::endl;
	//std::cout << matrix[3][0] << "," << matrix[3][1] << "," << matrix[3][2] << "," << matrix[3][3] << std::endl;
}

/*
	func：将source_points对齐到target_points
	算法描述:
			A.首先将中心归零，计算点云的中心点，然后减去这个中心点
			B.利用PCA算法分别计算每个点云的pca轴，按照特征值由大到小进行排列
			C.将source和target点云分别变换到对应的pca轴下，此时就实现了点云的粗对齐
			source_points_aligned = (source_points - source_center)*source_vectors
			target_points_aligned = (target_points - target_center)*target_vectors
			D.进一步验证对齐是否正确。（将两个pca坐标系对齐之后，由于存在正反向的问题，需要进一步验证是否真正对齐）
			在x、y、z这三个方向上，分别变换方向，然后再计算两个点云之间的最小距离，取距离最小的那个变换T
			E.得到最终的对齐结果
			source_points_aligned_final = source_points_aligned * (target_vectors.T) * T+target_canter
	:param source_points:
	:param target_points:
	:return:粗配准矩阵，std::vector<std::vector<float>>
*/
vtkSmartPointer<vtkMatrix4x4> preAlignedTwoPointClouds(const std::vector<float>& source_points, const std::vector<float>& target_points,
	std::vector<std::vector<float>>& source_pca_vectors, std::vector<std::vector<float>>& target_pca_vectors)
{
	std::vector<float> source_eigen_values;
	std::vector<float> source_center;
	PCA(source_points, source_eigen_values, source_pca_vectors, source_center);
	std::vector<std::vector<float>> source_eigen_vectors = source_pca_vectors;

	std::vector<float> target_eigen_values;
	std::vector<float> target_center;
	PCA(target_points, target_eigen_values, target_pca_vectors, target_center);
	std::vector<std::vector<float>> target_eigen_vectors = target_pca_vectors;


	auto source_points_decenter = pointsDecenter(source_points, source_center);
	auto target_points_decenter = pointsDecenter(target_points, target_center);

	auto pre_aligned_source_points = getPointsDotMatrix(source_points_decenter, source_eigen_vectors);
	auto pre_aligned_target_points = getPointsDotMatrix(target_points_decenter, target_eigen_vectors);

	/*auto source_points_actors = createPointsActor(pre_aligned_source_points, 0.3, 1.0, "yellow");
	auto target_points_actors = createPointsActor(pre_aligned_target_points, 0.3, 1.0, "red");
	std::vector<vtkSmartPointer<vtkActor>> all_actors;
	all_actors.insert(all_actors.end(), source_points_actors.begin(), source_points_actors.end());
	all_actors.insert(all_actors.end(), target_points_actors.begin(), target_points_actors.end());
	showActors(all_actors);
*/




	auto source_max_point = getTheMaxAxisPoint(pre_aligned_source_points);
	auto source_min_point = getTheMinAxisPoint(pre_aligned_source_points);
	
	auto target_max_point = getTheMaxAxisPoint(pre_aligned_target_points);
	auto target_min_point = getTheMinAxisPoint(pre_aligned_target_points);

	std::vector<float> source_bbox = { source_max_point[0] - source_min_point[0],
									   source_max_point[1] - source_min_point[1],
									   source_max_point[2] - source_min_point[2] };

	std::vector<float> target_bbox = { target_max_point[0] - target_min_point[0],
									   target_max_point[1] - target_min_point[1],
									   target_max_point[2] - target_min_point[2] };

	std::vector<std::vector<float>> scale_matrix = { {target_bbox[0] / source_bbox[0],0.0, 0.0},
													 {0.0, target_bbox[1] / source_bbox[1],0.0},
													 {0.0, 0.0, target_bbox[2] / source_bbox[2] } };


	auto pre_aligned_source_points_scaled = getPointsDotMatrix(pre_aligned_source_points, scale_matrix);


	//两个点云的主轴方向，有可能出现反向的情况，需要对其进行处理
	std::vector<std::vector<float>> R0 = { {1.0, 0.0, 0.0},{0.0, 1.0, 0.0,},{0.0, 0.0, 1.0} };
	std::vector<std::vector<float>> R1 = { {1.0, 0.0, 0.0},{0.0, -1.0, 0.0,},{0.0, 0.0, -1.0} };
	std::vector<std::vector<float>> R2 = { {-1.0, 0.0, 0.0},{0.0, -1.0, 0.0,},{0.0, 0.0, 1.0} };
	std::vector<std::vector<float>> R3 = { {-1.0, 0.0, 0.0},{0.0, 1.0, 0.0,},{0.0, 0.0, -1.0} };
	std::vector<std::vector<std::vector<float>>> R = { R0, R1, R2, R3 };

	
	std::vector<Eigen::Vector3d> points;
	for (int i = 0; i < pre_aligned_target_points.size() / 3; i++)
	{
		double curX = pre_aligned_target_points[i * 3 + 0];
		double curY = pre_aligned_target_points[i * 3 + 1];
		double curZ = pre_aligned_target_points[i * 3 + 2];
		points.push_back(Eigen::Vector3d(curX, curY, curZ));
	}
	open3d::geometry::PointCloud point_cloud(points);
	open3d::geometry::KDTreeFlann  kd_tree(point_cloud);

	std::vector<int> cur_indices;
	std::vector<double> cur_distance;
	double all_distance[4] = { 0.0 };
	for (int i = 0; i < R.size(); i++)
	{
		auto curR = R[i];
		double cur_distance_sum = 0.0;
		auto cur_points = getPointsDotMatrix(pre_aligned_source_points, curR);
		for (int j = 0; j < cur_points.size() / 3; j++)
		{
			double curX = cur_points[j * 3 + 0];
			double curY = cur_points[j * 3 + 1];
			double curZ = cur_points[j * 3 + 2];
			Eigen::Vector3d query_point(curX, curY, curZ);
			kd_tree.SearchKNN(query_point, 1, cur_indices, cur_distance);
	
			cur_distance_sum += sqrt(cur_distance[0]);
		}
		all_distance[i] = cur_distance_sum;
	}

	int min_index = 0;
	auto min_distance = all_distance[min_index];
	for (int i = 1; i < 4; i++)
	{
		if (min_distance > all_distance[i])
		{
			min_index = i;
			min_distance = all_distance[i];
		}
	}
	auto trans = R[min_index];
	/*std::cout << "source center:" << source_center[0] << "," << source_center[1] << "," << source_center[2] << std::endl << std::endl;
	std::cout << "target center:" << target_center[0] << "," << target_center[1] << "," << target_center[2] << std::endl << std::endl;*/

	std::vector<std::vector<float>> matrix1 = { {1.0, 0.0, 0.0, -source_center[0]}, 
												{0.0, 1.0, 0.0, -source_center[1]},
												{0.0, 0.0, 1.0, -source_center[2]}, 
	                                            {0.0, 0.0, 0.0, 1.0} };
	/*std::cout << "\nMatrix1:" << std::endl;
	printMatrix(matrix1);
*/
	std::vector<std::vector<float>> matrix2 = { {source_eigen_vectors[0][0], source_eigen_vectors[0][1], source_eigen_vectors[0][2], 0.0},
												{source_eigen_vectors[1][0], source_eigen_vectors[1][1], source_eigen_vectors[1][2], 0.0},
												{source_eigen_vectors[2][0], source_eigen_vectors[2][1], source_eigen_vectors[2][2], 0.0},
												{0.0, 0.0, 0.0, 1.0}, };
	//std::cout << "\nMatrix2:" << std::endl;
	//printMatrix(matrix2);


	std::vector<std::vector<float>> matrix3 = { {trans[0][0], trans[0][1], trans[0][2], 0.0},
												{trans[1][0], trans[1][1], trans[1][2], 0.0},
												{trans[2][0], trans[2][1], trans[2][2], 0.0},
												{0.0, 0.0, 0.0, 1.0}};
	/*std::cout << "\nMatrix3:" << std::endl;
	printMatrix(matrix3);
*/
	std::vector<std::vector<float>> matrix4 = { {target_eigen_vectors[0][0], target_eigen_vectors[1][0], target_eigen_vectors[2][0], 0.0},
												{target_eigen_vectors[0][1], target_eigen_vectors[1][1], target_eigen_vectors[2][1], 0.0},
												{target_eigen_vectors[0][2], target_eigen_vectors[1][2], target_eigen_vectors[2][2], 0.0},
											    {0.0, 0.0, 0.0, 1.0} };
	/*std::cout << "\nMatrix4:" << std::endl;
	printMatrix(matrix4);*/

	std::vector<std::vector<float>> matrix5 = { {1.0, 0.0, 0.0, target_center[0]},
												{0.0, 1.0, 0.0, target_center[1]},
												{0.0, 0.0, 1.0, target_center[2]},
												{0.0, 0.0, 0.0, 1.0} };
	//std::cout << "\nMatrix5:" << std::endl;
	//printMatrix(matrix5);


	auto tmp = matrix4DotMatrix4(matrix5, matrix4);
	tmp = matrix4DotMatrix4(tmp, matrix3);
	tmp = matrix4DotMatrix4(tmp, matrix2);
	tmp = matrix4DotMatrix4(tmp, matrix1);
	
	auto vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			vtk_matrix->SetElement(i, j, tmp[i][j]);
		}
	}
	
	return vtk_matrix;
}

vtkSmartPointer<vtkMatrix4x4> convertVectorMatrix3x3TovtkMatrix4x4(const std::vector<std::vector<float>>& matrix)
{
	auto vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (i < 3 && j < 3)
			{
				vtk_matrix->SetElement(i, j, matrix[i][j]);
			}
			else if (i == 3 && j == 3)
			{
				vtk_matrix->SetElement(i, j, 1.0);
			}
			else
			{
				vtk_matrix->SetElement(i, j, 0.0);
			}
		}
	}
	return vtk_matrix;
}
/*
Func:
	计算点与左乘矩阵后得到的新的点的坐标

*/

void vectorPointsDotvtkMatrix4x4(const std::vector<float>& points, const vtkSmartPointer<vtkMatrix4x4> left_matrix, std::vector<float>& points_new)
{
	double* data = left_matrix->GetData();
	for (int i = 0; i < points.size() / 3; i++)
	{
		auto curX = points[i * 3 + 0];
		auto curY = points[i * 3 + 1];
		auto curZ = points[i * 3 + 2];
		for (int i = 0; i < 3; i++)
		{
			double cur_data = data[i * 4 + 0] * curX + data[i * 4 + 1] * curY + data[i * 4 + 2] * curZ + data[i * 4 + 3] * 1.0;
			points_new.push_back(float(cur_data));
		}
	}
}


/*
Func：
	基于模板匹配的方法，得到椎体的左、右椎弓根以及椎体顶面的特征点
Input:
	label_name:待配准的椎段的名称,取值范围为"08"-"25"
	template_stl_dir:模板stl文件所在的路径
	target_polydata:待配准的椎体的polydata
	target_points:待配准的椎体的点云

Output:
	left_points:配准得到的左侧椎弓根峡部的特征点
	right_points:配准得到的右侧椎弓根峡部的特征点
	top_points:配准得到的椎体顶面的特征点

Author:BigPanda
Date:2025.03.17 14:52

*/
void registrationPolydata(const std::string& label_name, const std::string& template_stl_dir, 
	const vtkSmartPointer<vtkPolyData> target_polydata ,const std::vector<float>& target_points,
	std::vector<float>& left_points,std::vector<float>& right_points, std::vector<float>& top_points)
{
	std::string src_landmark_file = template_stl_dir + "/label_" + label_name + ".txt";
	std::string src_stl_file = template_stl_dir + "/label_" + label_name + ".stl";
	std::vector<float> src_top_points;
	std::vector<float> src_left_points;
	std::vector<float> src_right_points;
	
	Eigen::MatrixXd source_points_eigen = getPointsFromSTL(src_stl_file, POINT_NUM);
	auto source_polydata = createPolyDataFromSTL(src_stl_file);
	std::vector<float> source_points = matrixToVector(source_points_eigen);

	std::vector<std::vector<float>> source_pca_vectors;
	std::vector<std::vector<float>> target_pca_vectors;


	auto status = loadLandmarksFromFile(src_landmark_file, src_top_points, src_left_points, src_right_points);

	auto pre_aligned_matrix = preAlignedTwoPointClouds(source_points, target_points, source_pca_vectors, target_pca_vectors);


	/*auto source_vtk_matrix = convertVectorMatrix3x3TovtkMatrix4x4(source_pca_vectors);
	auto target_vtk_matrix = convertVectorMatrix3x3TovtkMatrix4x4(target_pca_vectors); 

	auto source_center = getPointsMean(source_points);
	auto target_center = getPointsMean(target_points);*/

	//auto source_transform1 = vtkSmartPointer<vtkTransform>::New();
	//source_transform1->Translate(-source_center[0], -source_center[1], -source_center[2]);
	//auto source_transform_filter1 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	//source_transform_filter1->SetTransform(source_transform1);
	//source_transform_filter1->SetInputData(source_polydata);
	//source_transform_filter1->Update();

	//auto source_transform2 = vtkSmartPointer<vtkTransform>::New();
	//source_transform2->SetMatrix(source_vtk_matrix);

	//auto source_transform_filter2 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	//source_transform_filter2->SetTransform(source_transform2);
	//source_transform_filter2->SetInputData(source_transform_filter1->GetOutput());
	//source_transform_filter2->Update();

	/*auto source_transform3 = vtkSmartPointer<vtkTransform>::New();
	source_transform3->Translate(source_center[0], source_center[1], source_center[2]);
	auto source_transform_filter3 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	source_transform_filter3->SetTransform(source_transform3);
	source_transform_filter3->SetInputData(source_transform_filter2->GetOutput());
	source_transform_filter3->Update();*/


	//auto source_transform_actor = createActorFromPolyData(source_transform_filter2->GetOutput(), "red", 1.0);

	//auto target_transform1 = vtkSmartPointer<vtkTransform>::New();
	//target_transform1->Translate(-target_center[0], -target_center[1], -target_center[2]);
	//auto target_transform_filter1 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	//target_transform_filter1->SetTransform(target_transform1);
	//target_transform_filter1->SetInputData(target_polydata);
	//target_transform_filter1->Update();

	//auto target_transform2 = vtkSmartPointer<vtkTransform>::New();
	//target_transform2->SetMatrix(target_vtk_matrix);

	//auto target_transform_filter2 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	//target_transform_filter2->SetTransform(target_transform2);
	//target_transform_filter2->SetInputData(target_transform_filter1->GetOutput());
	//target_transform_filter2->Update();

	//auto target_transform_actor = createActorFromPolyData(target_transform_filter2->GetOutput(), "blue", 1.0);

	//auto source_axis_actors = createAxisActors(source_center, source_pca_vectors[0], source_pca_vectors[1], source_pca_vectors[2]);
	//auto source_main_vector_actor2 = createLineActorByNormal({0.0, 0.0, 0.0}, { 1.0, 0.0, 0.0 }, 100.0, 6, "red");

	//auto target_axis_actors = createAxisActors(target_center, target_pca_vectors[0], target_pca_vectors[1], target_pca_vectors[2]);


	//auto target_main_vector_actor = createLineActorByNormal(target_center, target_pca_vectors[0], 100.0, 4, "red");


	auto pre_aligned_transform = vtkSmartPointer<vtkTransform>::New();
	pre_aligned_transform->SetMatrix(pre_aligned_matrix);

	auto pre_aligned_trans_poly_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	pre_aligned_trans_poly_filter->SetTransform(pre_aligned_transform);
	pre_aligned_trans_poly_filter->SetInputData(source_polydata);
	pre_aligned_trans_poly_filter->Update();

	auto icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
	icp->SetSource(pre_aligned_trans_poly_filter->GetOutput());
	icp->SetTarget(target_polydata);
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaximumNumberOfIterations(100);
	icp->StartByMatchingCentroidsOn();
	icp->Modified();
	icp->Update();

	auto icp_matrix = icp->GetMatrix();

	auto icp_trans_form_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	icp_trans_form_filter->SetInputData(pre_aligned_trans_poly_filter->GetOutput());
	icp_trans_form_filter->SetTransform(icp);
	icp_trans_form_filter->Update();

	auto final_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
	vtkMatrix4x4::Multiply4x4(icp_matrix, pre_aligned_matrix, final_matrix);


	vectorPointsDotvtkMatrix4x4(src_top_points, final_matrix, top_points);
	vectorPointsDotvtkMatrix4x4(src_left_points, final_matrix, left_points);
	vectorPointsDotvtkMatrix4x4(src_right_points, final_matrix, right_points);

	/*auto data = final_matrix->GetData();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j<4; j++)
		{
			printf("%f,", data[i * 4 + j]);
		}
		printf("\n");
	}

*/
	//auto source_actor = createActorFromPolyData(source_polydata, "red", 0.9);
	//auto pre_aligned_source_actor = createActorFromPolyData(icp_trans_form_filter->GetOutput(), "yellow", 0.8);

	//auto target_actor = createActorFromPolyData(target_polydata, "Cornsilk", 0.9);
	//auto top_points_actors = createPointsActor(top_points, 0.3, 1.0, "red");
	//auto left_points_actors = createPointsActor(left_points, 0.3, 1.0, "green");
	//auto right_points_actors = createPointsActor(right_points, 0.3, 1.0, "blue");

	//std::vector<vtkSmartPointer<vtkActor>> all_actors;
	//all_actors.push_back(target_actor);
	//all_actors.push_back(pre_aligned_source_actor);
	//all_actors.insert(all_actors.end(), top_points_actors.begin(), top_points_actors.end());
	//all_actors.insert(all_actors.end(), left_points_actors.begin(), left_points_actors.end());
	//all_actors.insert(all_actors.end(), right_points_actors.begin(), right_points_actors.end());
	//showActors(all_actors);




}

void pedicleSurgeryPlanning(std::vector<float>& top_points, std::vector<float>& left_points, std::vector<float>& right_points,
	vtkSmartPointer<vtkPolyData> spine_poly_data, std::string window_name, std::string save_axis_dir, std::string save_png_dir)
{
	std::vector<vtkSmartPointer<vtkActor>> all_actors;
	clock_t start = clock();
	std::vector<vtkSmartPointer<vtkActor>> top_points_actor = createPointsActor(top_points, 0.5, 1.0, "Red");
	std::vector<vtkSmartPointer<vtkActor>> left_points_actor = createPointsActor(left_points, 0.5, 1.0, "Green");
	std::vector<vtkSmartPointer<vtkActor>> right_points_actor = createPointsActor(right_points, 0.5, 1.0, "Blue");
	vtkSmartPointer<vtkActor> spine_actor = createActorFromPolyData(spine_poly_data, "Cornsilk", 0.8);
	clock_t end0 = clock();
	std::cout << "create points actor used:"<< double(end0 - start) / CLOCKS_PER_SEC 
		<< "/" << double(end0 - start) / CLOCKS_PER_SEC << std::endl;


	std::vector<float> left_cut_plane_center;
	std::vector<float> left_cut_plane_normal;

	std::vector<float> right_cut_plane_center;
	std::vector<float> right_cut_plane_normal;


	std::vector<float> left_bound_points_min;
	std::vector<float> right_bound_points_min;

	left_bound_points_min = left_points;
	right_bound_points_min = right_points;



	std::vector<float> top_points_center = getPointsMean(top_points);
	std::vector<float> left_points_center = getPointsMean(left_points);
	std::vector<float> right_points_center = getPointsMean(right_points);

	left_cut_plane_center = left_points_center;
	vtkSmartPointer<vtkActor> left_fit_plane_actor;
	std::vector<float> left_fit_plane_center;
	std::vector<float> left_fit_plane_normal;
	left_fit_plane_actor = fitPlaneActorFromPoints(left_fit_plane_center, left_fit_plane_normal, left_points, "Green");
	left_cut_plane_center = left_fit_plane_center;



	std::vector<float> left_reference_normal = normalizeVector(top_points_center, left_points_center);
	if (getTwoVectorDotValue(left_reference_normal, left_fit_plane_normal) < 0.0f)
	{
		left_cut_plane_normal.push_back(-left_fit_plane_normal[0]);
		left_cut_plane_normal.push_back(-left_fit_plane_normal[1]);
		left_cut_plane_normal.push_back(-left_fit_plane_normal[2]);
	}
	else
	{
		left_cut_plane_normal = left_fit_plane_normal;
	}
	float left_angle = calculateAngle(left_reference_normal, left_cut_plane_normal);
	if (left_angle > CrossAngleThreshold)
	{
		left_cut_plane_normal = left_reference_normal;
	}


	vtkSmartPointer<vtkActor> right_fit_plane_actor;
	std::vector<float> right_fit_plane_center;
	std::vector<float> right_fit_plane_normal;
	right_fit_plane_actor = fitPlaneActorFromPoints(right_fit_plane_center, right_fit_plane_normal, right_points, "Blue");
	std::vector<float> right_reference_normal = normalizeVector(top_points_center, right_points_center);
	right_cut_plane_center = right_fit_plane_center;

	if (getTwoVectorDotValue(right_reference_normal, right_fit_plane_normal) < 0.0)
	{
		right_cut_plane_normal.push_back(-right_fit_plane_normal[0]);
		right_cut_plane_normal.push_back(-right_fit_plane_normal[1]);
		right_cut_plane_normal.push_back(-right_fit_plane_normal[2]);
	}
	else
	{
		right_cut_plane_normal = right_fit_plane_normal;
	}
	float right_angle = calculateAngle(right_reference_normal, right_cut_plane_normal);
	if (right_angle > CrossAngleThreshold)
	{
		right_cut_plane_normal = right_reference_normal;
	}

	vtkSmartPointer<vtkActor> left_cut_plane_normal_actor = createLineActorByNormal(left_cut_plane_center, left_cut_plane_normal, 20.0, 4.0, "red");
	vtkSmartPointer<vtkActor> right_cut_plane_normal_actor = createLineActorByNormal(right_cut_plane_center, right_cut_plane_normal, 20.0, 4.0, "red");

	std::vector<float> left_cut_center;
	std::vector<float>left_bound_points;
	float left_cut_area = 0.0;
	getClipedCenterPoints(left_bound_points, left_cut_center, left_cut_area, left_cut_plane_center, left_cut_plane_normal, spine_poly_data);
	std::vector<vtkSmartPointer<vtkActor>> left_bound_points_actors;
	left_bound_points_actors = createPointsActor(left_bound_points, 0.3, 1.0, "Red");

	std::vector<float> right_cut_center;
	std::vector<float> right_bound_points;
	float right_cut_area = 0.0;
	getClipedCenterPoints(right_bound_points, right_cut_center, right_cut_area, right_cut_plane_center, right_cut_plane_normal, spine_poly_data);
	
	std::vector<vtkSmartPointer<vtkActor>> right_bound_points_actors;
	right_bound_points_actors = createPointsActor(right_bound_points, 0.3, 1.0, "Red");

	clock_t end1 = clock();
	std::cout << "create bound points used:" << double(end1 - end0) / CLOCKS_PER_SEC 
		<< "/" << double(end1 - start) / CLOCKS_PER_SEC << std::endl;
	//############### 将切面的中心点，替换之前由4个特征点拟合的中心点   #############
	if (getDistanceOfTwoPoints(left_cut_plane_center, left_cut_center) < TwoCentersDistanceThreshold)
	{
		left_cut_plane_center = left_cut_center;
		left_bound_points_min = left_bound_points;
	}

	if (getDistanceOfTwoPoints(right_cut_plane_center, right_cut_center) < TwoCentersDistanceThreshold)
	{
		right_cut_plane_center = right_cut_center;
		right_bound_points_min = right_bound_points;
	}

	std::vector<float> right_center_point_projected_on_left_plane;
	right_center_point_projected_on_left_plane = getProjectedPointOnPlane(right_cut_plane_center, left_cut_plane_normal, left_cut_plane_center);
	std::vector<float> left_axis_normalX = normalizeVector(right_center_point_projected_on_left_plane, left_cut_plane_center);
	std::vector<float> left_axis_normalY = left_cut_plane_normal;
	std::vector<float> left_axis_normalZ = getTwoVectorCrossValue(left_axis_normalX, left_axis_normalY);


	std::vector<float> left_center_point_projected_on_right_plane;
	left_center_point_projected_on_right_plane = getProjectedPointOnPlane(left_cut_plane_center, right_cut_plane_normal, right_cut_plane_center);
	std::vector<float> right_axis_normalX = normalizeVector(left_center_point_projected_on_right_plane, right_cut_plane_center);
	std::vector<float> right_axis_normalY = right_cut_plane_normal;
	std::vector<float> right_axis_normalZ = getTwoVectorCrossValue(right_axis_normalX, right_axis_normalY);

	//std::vector<vtkSmartPointer<vtkActor>> left_axis_actors;
	//left_axis_actors = createAxisActors(left_cut_plane_center, left_axis_normalX, left_axis_normalY, left_axis_normalZ);

	//std::vector<vtkSmartPointer<vtkActor>> right_axis_actors;
	//right_axis_actors = createAxisActors(right_cut_plane_center, right_axis_normalX, right_axis_normalY, right_axis_normalZ);


	//############## step1_1: rotate around X, left side  ###################
	float left_cut_plane_area_min1 = 0.0;
	std::vector<float> left_bound_points_min1;
	std::vector<std::vector<float>> left_rotate_matrix_min1;
	std::vector<float> left_center_min1;

	getTheMinCutPlaneArea(left_cut_plane_area_min1, left_bound_points_min1, left_rotate_matrix_min1, left_center_min1,
		left_axis_normalX, left_axis_normalY, left_cut_plane_center, spine_poly_data, SearchRotateAngle, TwoCentersDistanceThreshold);


	if (left_bound_points_min1.size() > 0)
	{
		std::vector<vtkSmartPointer<vtkActor>> left_bound_points_min1_actor = createPointsActor(left_bound_points_min1, 0.2, 1.0, "red");
		all_actors.insert(all_actors.end(), left_bound_points_min1_actor.begin(), left_bound_points_min1_actor.end());
		left_bound_points_min = left_bound_points_min1;
	}

	//############## step1_2: rotate around X, right side  ###################
	float right_cut_plane_area_min1 = 0.0;
	std::vector<float> right_bound_points_min1;
	std::vector<std::vector<float>> right_rotate_matrix_min1;
	std::vector<float> right_center_min1;

	getTheMinCutPlaneArea(right_cut_plane_area_min1, right_bound_points_min1, right_rotate_matrix_min1, right_center_min1,
		right_axis_normalX, right_axis_normalY, right_cut_plane_center, spine_poly_data, SearchRotateAngle, TwoCentersDistanceThreshold);


	if (right_bound_points_min1.size() > 0)
	{
		std::vector<vtkSmartPointer<vtkActor>> right_bound_points_min1_actor = createPointsActor(right_bound_points_min1, 0.2, 1.0, "red");
		all_actors.insert(all_actors.end(), right_bound_points_min1_actor.begin(), right_bound_points_min1_actor.end());
		right_bound_points_min = right_bound_points_min1;
	}

	//############step1_3:  更新椎弓根坐标系和中心点  ##############
	std::vector<float> left_axis_normalX_new, left_axis_normalY_new, left_axis_normalZ_new;

	if (left_rotate_matrix_min1.size() > 0)
	{
		left_axis_normalX_new = left_axis_normalX;
		left_axis_normalY_new = getVectorDotMatrixValue(left_axis_normalY, left_rotate_matrix_min1);
		left_axis_normalZ_new = getVectorDotMatrixValue(left_axis_normalZ, left_rotate_matrix_min1);
		left_cut_plane_center = left_center_min1;
	}
	else
	{
		left_axis_normalX_new = left_axis_normalX;
		left_axis_normalY_new = left_axis_normalY;
		left_axis_normalZ_new = left_axis_normalZ;
	}

	std::vector<float> right_axis_normalX_new, right_axis_normalY_new, right_axis_normalZ_new;
	if (right_rotate_matrix_min1.size() > 0)
	{
		right_axis_normalX_new = right_axis_normalX;
		right_axis_normalY_new = getVectorDotMatrixValue(right_axis_normalY, right_rotate_matrix_min1);
		right_axis_normalZ_new = getVectorDotMatrixValue(right_axis_normalZ, right_rotate_matrix_min1);
		right_cut_plane_center = right_center_min1;
	}
	else
	{
		right_axis_normalX_new = right_axis_normalX;
		right_axis_normalY_new = right_axis_normalY;
		right_axis_normalZ_new = right_axis_normalZ;
	}
	clock_t end2 = clock();
	std::cout << "rotate around X used:" << double(end2 - end1) / CLOCKS_PER_SEC 
		<< "/" << double(end2 - start) / CLOCKS_PER_SEC << std::endl;
	//############## step2_1: rotate around Z, left side  ###############
	float left_cut_plane_area_min2 = 0.0;
	std::vector<float> left_bound_points_min2;
	std::vector<std::vector<float>> left_rotate_matrix_min2;
	std::vector<float> left_center_min2;

	getTheMinCutPlaneArea(left_cut_plane_area_min2, left_bound_points_min2, left_rotate_matrix_min2, left_center_min2,
		left_axis_normalZ_new, left_axis_normalY, left_cut_plane_center, spine_poly_data, SearchRotateAngle, TwoCentersDistanceThreshold);


	if (left_bound_points_min2.size() > 0)
	{
		std::vector<vtkSmartPointer<vtkActor>> left_bound_points_min2_actor = createPointsActor(left_bound_points_min2, 0.2, 0.8, "yellow");
		all_actors.insert(all_actors.end(), left_bound_points_min2_actor.begin(), left_bound_points_min2_actor.end());
		left_bound_points_min = left_bound_points_min2;
	}

	//############## step2_2: rotate around Z, right side  ###############
	float right_cut_plane_area_min2 = 0.0;
	std::vector<float> right_bound_points_min2;
	std::vector<std::vector<float>> right_rotate_matrix_min2;
	std::vector<float> right_center_min2;

	getTheMinCutPlaneArea(right_cut_plane_area_min2, right_bound_points_min2, right_rotate_matrix_min2, right_center_min2,
		right_axis_normalZ_new, right_axis_normalY_new, right_cut_plane_center, spine_poly_data, SearchRotateAngle, TwoCentersDistanceThreshold);


	if (right_bound_points_min2.size() > 0)
	{
		std::vector<vtkSmartPointer<vtkActor>> right_bound_points_min2_actor = createPointsActor(right_bound_points_min2, 0.2, 0.8, "yellow");
		all_actors.insert(all_actors.end(), right_bound_points_min2_actor.begin(), right_bound_points_min2_actor.end());
		right_bound_points_min = right_bound_points_min2;
	}

	//############step2_3:  更新椎弓根坐标系和中心点  ##############
	std::vector<float> left_axis_normalX_new2, left_axis_normalY_new2, left_axis_normalZ_new2;

	if (left_rotate_matrix_min2.size() > 0)
	{
		left_axis_normalX_new2 = getVectorDotMatrixValue(left_axis_normalX_new, left_rotate_matrix_min2);
		left_axis_normalY_new2 = getVectorDotMatrixValue(left_axis_normalY_new, left_rotate_matrix_min2);
		left_axis_normalZ_new2 = left_axis_normalZ_new;
		left_cut_plane_center = left_center_min2;
	}
	else
	{
		left_axis_normalX_new2 = left_axis_normalX_new;
		left_axis_normalY_new2 = left_axis_normalY_new;
		left_axis_normalZ_new2 = left_axis_normalZ_new;
	}
	left_cut_plane_normal = left_axis_normalY_new2;

	std::vector<float> right_axis_normalX_new2, right_axis_normalY_new2, right_axis_normalZ_new2;
	if (right_rotate_matrix_min1.size() > 0)
	{
		right_axis_normalX_new2 = getVectorDotMatrixValue(right_axis_normalX_new, right_rotate_matrix_min2);
		right_axis_normalY_new2 = getVectorDotMatrixValue(right_axis_normalY_new, right_rotate_matrix_min2);
		right_axis_normalZ_new2 = right_axis_normalZ_new;
		right_cut_plane_center = right_center_min2;
	}
	else
	{
		right_axis_normalX_new2 = right_axis_normalX_new;
		right_axis_normalY_new2 = right_axis_normalY_new;
		right_axis_normalZ_new2 = right_axis_normalZ_new;
	}
	right_cut_plane_normal = right_axis_normalY_new2;

	clock_t end3 = clock();
	std::cout << "rotate around Z used:" << double(end3 - end2) / CLOCKS_PER_SEC 
		<< "/" << double(end3 - start) / CLOCKS_PER_SEC << std::endl;

	//############step3_1: 沿着y轴方向搜索最小截面 ##############
	std::vector<float> left_bound_points_min3;
	std::vector<float> left_center_min3;
	float left_cut_plane_area_min3 = 0.0;
	getTheMinCutPlaneAreaAlongAxisY(left_cut_plane_area_min3, left_bound_points_min3,
		left_center_min3, left_cut_plane_center, left_cut_plane_normal, spine_poly_data);

	std::vector<float> right_bound_points_min3;
	std::vector<float> right_center_min3;
	float right_cut_plane_area_min3 = 0.0;
	getTheMinCutPlaneAreaAlongAxisY(right_cut_plane_area_min3, right_bound_points_min3,
		right_center_min3, right_cut_plane_center, right_cut_plane_normal, spine_poly_data);

	//############step3_2: 更新截面中心点   ##############
	if (getDistanceOfTwoPoints(left_center_min3, left_cut_plane_center) < TwoCentersDistanceThreshold)
	{
		left_cut_plane_center = left_center_min3;
		std::vector<vtkSmartPointer<vtkActor>> left_bound_points_min3_actors;
		left_bound_points_min3_actors = createPointsActor(left_bound_points_min3, 0.4, 0.6, "purple");
		all_actors.insert(all_actors.end(), left_bound_points_min3_actors.begin(), left_bound_points_min3_actors.end());
		left_bound_points_min = left_bound_points_min3;
	}

	if (getDistanceOfTwoPoints(right_center_min3, right_cut_plane_center) < TwoCentersDistanceThreshold)
	{
		right_cut_plane_center = right_center_min3;
		std::vector<vtkSmartPointer<vtkActor>> right_bound_points_min3_actors;
		right_bound_points_min3_actors = createPointsActor(right_bound_points_min3, 0.4, 0.6, "purple");

		all_actors.insert(all_actors.end(), right_bound_points_min3_actors.begin(), right_bound_points_min3_actors.end());

		right_bound_points_min = right_bound_points_min3;
	}

	clock_t end4 = clock();
	std::cout << "沿着 Y轴搜索最小截面用时 used:" << double(end4 - end3) / CLOCKS_PER_SEC 
		<<"/"<< double(end4 - start)/ CLOCKS_PER_SEC << std::endl;

	//############## step4: 检查左、右椎弓根坐标系的方向  #############
	float left_flag = (top_points_center[0] - left_cut_plane_center[0]) * left_axis_normalY_new2[0] +
		(top_points_center[1] - left_cut_plane_center[1]) * left_axis_normalY_new2[1] +
		(top_points_center[2] - left_cut_plane_center[2]) * left_axis_normalY_new2[2];
	if (left_flag < 0.0f)
	{
		left_axis_normalZ_new2[0] = -left_axis_normalZ_new2[0];
		left_axis_normalZ_new2[1] = -left_axis_normalZ_new2[1];
		left_axis_normalZ_new2[2] = -left_axis_normalZ_new2[2];

		left_axis_normalY_new2[0] = -left_axis_normalY_new2[0];
		left_axis_normalY_new2[1] = -left_axis_normalY_new2[1];
		left_axis_normalY_new2[2] = -left_axis_normalY_new2[2];
	}

	float right_flag = (top_points_center[0] - right_cut_plane_center[0]) * right_axis_normalY_new2[0] +
		(top_points_center[1] - right_cut_plane_center[1]) * right_axis_normalY_new2[1] +
		(top_points_center[2] - right_cut_plane_center[2]) * right_axis_normalY_new2[2];

	if (right_flag > 0.0f)
	{
		right_axis_normalZ_new2[0] = -right_axis_normalZ_new2[0];
		right_axis_normalZ_new2[1] = -right_axis_normalZ_new2[1];
		right_axis_normalZ_new2[2] = -right_axis_normalZ_new2[2];
	}
	else
	{

		right_axis_normalY_new2[0] = -right_axis_normalY_new2[0];
		right_axis_normalY_new2[1] = -right_axis_normalY_new2[1];
		right_axis_normalY_new2[2] = -right_axis_normalY_new2[2];
	}

	auto left_axis_actors = createAxisActors(left_cut_plane_center, left_axis_normalX_new2, left_axis_normalY_new2, left_axis_normalZ_new2);
	auto right_axis_actors = createAxisActors(right_cut_plane_center, right_axis_normalX_new2, right_axis_normalY_new2, right_axis_normalZ_new2);



	auto left_cut_plane_center_actor = createSphereActor(left_cut_plane_center, 2.0, 1.0, "green");
	auto right_cut_plane_center_actor = createSphereActor(right_cut_plane_center, 2.0, 1.0, "blue");

	//step5:建立脊柱坐标系，脊柱坐标系的Z方向是，top_points拟合的平面的法向方向
	std::vector<float> top_fit_plane_normal;
	vtkSmartPointer<vtkActor> top_fit_plane_actor;
	std::vector<float> top_fit_plane_center;
	top_fit_plane_actor = fitPlaneActorFromPoints(top_fit_plane_center, top_fit_plane_normal, top_points, "Red");

	std::vector<float> spine_axis_normalZ;
	std::vector<float> left_right_axis_normalZ_mean = { float((left_axis_normalZ_new2[0] + right_axis_normalZ_new2[0]) / 2.0),
													    float((left_axis_normalZ_new2[1] + right_axis_normalZ_new2[1]) / 2.0),
													    float((left_axis_normalZ_new2[2] + right_axis_normalZ_new2[2]) / 2.0) };

	if (getTwoVectorDotValue(top_fit_plane_normal, left_right_axis_normalZ_mean) < 0.0f)
	{
		top_fit_plane_normal[0] = -top_fit_plane_normal[0];
		top_fit_plane_normal[1] = -top_fit_plane_normal[1];
		top_fit_plane_normal[2] = -top_fit_plane_normal[2];

	}

	if (calculateAngle(top_fit_plane_normal, left_right_axis_normalZ_mean) > CrossAngleThreshold)
	{
		spine_axis_normalZ = left_right_axis_normalZ_mean;
	}
	else
	{
		spine_axis_normalZ.push_back((top_fit_plane_normal[0] + (left_axis_normalZ_new2[0] + right_axis_normalZ_new2[0]) / 2.0) / 2.0);
		spine_axis_normalZ.push_back((top_fit_plane_normal[1] + (left_axis_normalZ_new2[1] + right_axis_normalZ_new2[1]) / 2.0) / 2.0);
		spine_axis_normalZ.push_back((top_fit_plane_normal[2] + (left_axis_normalZ_new2[2] + right_axis_normalZ_new2[2]) / 2.0) / 2.0);
	}
	auto tmp = normalizeVector(spine_axis_normalZ);
	spine_axis_normalZ[0] = tmp[0];
	spine_axis_normalZ[1] = tmp[1];
	spine_axis_normalZ[2] = tmp[2];

	std::vector<float> spine_axis_center = { float((left_cut_plane_center[0] + right_cut_plane_center[0]) / 2.0),
		float((left_cut_plane_center[1] + right_cut_plane_center[1]) / 2.0),
		float((left_cut_plane_center[2] + right_cut_plane_center[2]) / 2.0) };

	auto left_project_center = getProjectedPointOnPlane(left_cut_plane_center, spine_axis_normalZ, spine_axis_center);
	auto right_project_center = getProjectedPointOnPlane(right_cut_plane_center, spine_axis_normalZ, spine_axis_center);

	auto spine_axis_normalY = normalizeVector(left_project_center, right_project_center);
	auto spine_axis_normalX = getTwoVectorCrossValue(spine_axis_normalY, spine_axis_normalZ);

	auto spine_axis_center_actor = createSphereActor(spine_axis_center, 1.0, 1.0, "red");
	auto spine_axis_actors = createAxisActors(spine_axis_center, spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ);

	clock_t end5 = clock();
	std::cout << "建立脊柱坐标系用时:" << double(end5 - end4) / CLOCKS_PER_SEC
		<< "/" << double(end5 - start) / CLOCKS_PER_SEC << std::endl;

	//############ step6_1:计算椎弓根通道法向  #############
	float rot_angle = 15;

	left_angle = getAngleOfCutPlaneNormalAndSpineAxisNormalX(left_cut_plane_center, left_cut_plane_normal,
		spine_axis_center, spine_axis_normalZ, spine_axis_normalX);

	std::cout << "cal left angle:" << left_angle << std::endl;
	if (left_angle < rot_angle) {left_angle = rot_angle;}

	right_angle = getAngleOfCutPlaneNormalAndSpineAxisNormalX(right_cut_plane_center, right_cut_plane_normal,
		spine_axis_center, spine_axis_normalZ, spine_axis_normalX);
	std::cout << "cal right angle:" << right_angle << std::endl;

	if (right_angle < rot_angle) { right_angle = rot_angle; }

	auto pedicle_pipeline_L_normal = createPediclePipelineNormal(left_angle, spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ);
	auto pedicle_pipeline_R_normal = createPediclePipelineNormal(-right_angle, spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ);

	std::vector<float> left_point0 = { float(left_cut_plane_center[0] - 50.0*pedicle_pipeline_L_normal[0]),
		float(left_cut_plane_center[1] - 50.0*pedicle_pipeline_L_normal[1]),
		float(left_cut_plane_center[2] - 50.0*pedicle_pipeline_L_normal[2]) };
	std::vector<float> left_point1 = { float(left_cut_plane_center[0] + 70.0*pedicle_pipeline_L_normal[0]),
	float(left_cut_plane_center[1] + 70.0*pedicle_pipeline_L_normal[1]),
	float(left_cut_plane_center[2] + 70.0*pedicle_pipeline_L_normal[2]) };
	auto pedicle_pipeline_actor_L = createLineActorByPoints(left_point0, left_point1, 4.0, "magenta");

	std::vector<float> right_point0 = { float(right_cut_plane_center[0] - 50.0*pedicle_pipeline_R_normal[0]),
		float(right_cut_plane_center[1] - 50.0*pedicle_pipeline_R_normal[1]),
		float(right_cut_plane_center[2] - 50.0*pedicle_pipeline_R_normal[2]) };

	std::vector<float> right_point1 = { float(right_cut_plane_center[0] + 70.0*pedicle_pipeline_R_normal[0]),
		float(right_cut_plane_center[1] + 70.0*pedicle_pipeline_R_normal[1]),
		float(right_cut_plane_center[2] + 70.0*pedicle_pipeline_R_normal[2]) };
	auto pedicle_pipeline_actor_R = createLineActorByPoints(right_point0, right_point1, 4.0, "yellow");

	clock_t end6 = clock();
	std::cout << "建立椎弓根通道方向用时:" << double(end6 - end5) / CLOCKS_PER_SEC
		<< "/" << double(end6 - start) / CLOCKS_PER_SEC << std::endl;

	//############ step6_2:计算椎弓根通道法向与椎体交点  #############

	auto left_intersect_points = getIntersectPointsFromLineAndPolyData(left_point0, left_point1, spine_poly_data);
	std::vector<float> left_intersect_point0;
	std::vector<float> left_intersect_point1;

	if (left_intersect_points.size() < 2)
	{
		std::cout << "left side instersect points is less than two!" << std::endl;
	}
	else
	{
		getTheClosedTwoPoints(left_intersect_point0, left_intersect_point1, left_intersect_points, left_cut_plane_center, pedicle_pipeline_L_normal);
		auto left_intersect_point0_actor = createSphereActor(left_intersect_point0, 1.0, 1.0, "magenta");
		auto left_intersect_point1_actor = createSphereActor(left_intersect_point1, 1.0, 1.0, "magenta");
		all_actors.push_back(left_intersect_point0_actor);
		all_actors.push_back(left_intersect_point1_actor);

		auto left_cylinder_actor = createPediclePipelineCylinderActor(left_intersect_point0, left_intersect_point1, 0.8, 3.5 / 2.0, "magenta");
		all_actors.push_back(left_cylinder_actor);
	}

	auto right_intersect_points = getIntersectPointsFromLineAndPolyData(right_point0, right_point1, spine_poly_data);
	std::vector<float> right_intersect_point0;
	std::vector<float> right_intersect_point1;
	if (right_intersect_points.size() < 2)
	{
		std::cout << "right side intersect points is less than two!" << std::endl;
	}
	else
	{
		getTheClosedTwoPoints(right_intersect_point0, right_intersect_point1, right_intersect_points, right_cut_plane_center, pedicle_pipeline_R_normal);
		auto right_intersect_point0_actor = createSphereActor(right_intersect_point0, 1.0, 1.0, "yellow");
		auto right_intersect_point1_actor = createSphereActor(right_intersect_point1, 1.0, 1.0, "yellow");
		all_actors.push_back(right_intersect_point0_actor);
		all_actors.push_back(right_intersect_point1_actor);

		auto right_cylinder_actor = createPediclePipelineCylinderActor(right_intersect_point0, right_intersect_point1, 0.8, 3.5 / 2.0, "yellow");
		all_actors.push_back(right_cylinder_actor);

	}

	clock_t end7 = clock();
	std::cout << "椎弓根通道法向与椎体交点用时:" << double(end7 - end6) / CLOCKS_PER_SEC
		<< "/" << double(end7 - start) / CLOCKS_PER_SEC << std::endl;

	all_actors.push_back(spine_actor);
	all_actors.push_back(spine_axis_center_actor);
	all_actors.insert(all_actors.end(), spine_axis_actors.begin(), spine_axis_actors.end());
	all_actors.insert(all_actors.end(), top_points_actor.begin(), top_points_actor.end());
	all_actors.push_back(top_fit_plane_actor);
	//all_actors.insert(all_actors.end(), left_points_actor.begin(), left_points_actor.end());
	//all_actors.insert(all_actors.end(), right_points_actor.begin(), right_points_actor.end());

	all_actors.push_back(left_cut_plane_center_actor);
	all_actors.push_back(right_cut_plane_center_actor);
	all_actors.push_back(pedicle_pipeline_actor_L);
	all_actors.push_back(pedicle_pipeline_actor_R);

	//all_actors.push_back(left_fit_plane_actor);
	//all_actors.push_back(right_fit_plane_actor);

	//all_actors.push_back(left_cut_plane_normal_actor);
	//all_actors.push_back(right_cut_plane_normal_actor);

	//all_actors.insert(all_actors.end(), left_bound_points_actors.begin(), left_bound_points_actors.end());
	//all_actors.insert(all_actors.end(), right_bound_points_actors.begin(), right_bound_points_actors.end());

	//all_actors.insert(all_actors.end(), left_axis_actors.begin(), left_axis_actors.end());
	//all_actors.insert(all_actors.end(), right_axis_actors.begin(), right_axis_actors.end());


	//showActors(all_actors);
	std::string save_png_file = save_png_dir + "/" + window_name.replace(window_name.size() - 4, 4, ".png");
	std::cout << "save png file: " << save_png_file << std::endl;
	saveSpineSurgicalPlanning2Png(all_actors, spine_axis_center, spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ, save_png_file);

	clock_t end8 = clock();
	std::cout << "保存图片用时:" << double(end8 - end7) / CLOCKS_PER_SEC
		<< "/" << double(end8 - start) / CLOCKS_PER_SEC << std::endl;

}
