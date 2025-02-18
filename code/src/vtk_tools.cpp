#include "vtk_tools.h"
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
#include <vtkWindowToImageFilter.h>
#include <vtkCamera.h>
#include <vtkPNGWriter.h>
#include <vtkImageAppend.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>

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
			vec.push_back(matrix(i, j)); // ��Ԫ����ӵ� vector ��
		}
	}
	return vec;
}


std::vector<float> randomChoice(const std::vector<float>& spine_points, int num_points, bool replace)
{
	std::vector<float> result;
	std::vector<size_t> indices(spine_points.size() / 3);
	std::iota(indices.begin(), indices.end(), 0); // ������� 0, 1, 2, ..., spine_points.size()/3-1

	// �����������
	std::random_device rd;
	std::mt19937 g(rd());

	if (replace)
	{
		// �����ظ�����
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
		// �������ظ�����
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
	// ��ȡSTL�ļ�
	auto mesh = open3d::io::CreateMeshFromFile(stl_file);
	if (mesh->IsEmpty())
	{
		std::cout << "Mesh loaded successfully!" << std::endl;
		throw std::runtime_error("Error: Failed to load mesh from " + stl_file);
	}

	// ʹ��Poisson Disk Sampling�������в�����
	auto pcd = mesh->SamplePointsPoissonDisk(num_points);

	// ��������ת��ΪEigen����
	Eigen::MatrixXd points_xyz(pcd->points_.size(), 3); // ���� Nx3 �ľ���
	for (size_t i = 0; i < pcd->points_.size(); ++i)
	{
		points_xyz.row(i) = pcd->points_[i].transpose(); // ��ÿ���㸴�Ƶ��������
	}
	return points_xyz;
}


vtkSmartPointer<vtkPolyData> createPolyDataFromSTL(const std::string& stl_file)
{
	// ���� STL �Ķ���
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();

	// ���� STL �ļ�·��
	reader->SetFileName(stl_file.c_str());

	// ִ�ж�ȡ����
	reader->Update();
	if (reader->GetErrorCode() != 0)
	{
		throw std::runtime_error("Failed to read STL file");
	}
	// ��ȡ���������
	vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput();

	return poly_data;
}

vtkSmartPointer<vtkActor> createActorFromPolyData(vtkSmartPointer<vtkPolyData> polydata, const vtkStdString &color, double opacity)
{
	// ������ɫ����
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// �������������ӳ����
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);

	// ������Ա��������������
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// ���ò�������
	actor->GetProperty()->SetDiffuse(0.8);
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetSpecular(0.3);
	actor->GetProperty()->SetSpecularPower(60.0);
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

vtkSmartPointer<vtkActor> createActorFromSTL(const std::string& stlFile, const vtkStdString &color, double opacity)
{
	// ������ɫ����
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();

	// ����������STL��ȡ��
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(stlFile.c_str());
	reader->Update();

	// ����Ƿ��д�����
	if (reader->GetErrorCode() != 0) 
	{
		throw std::runtime_error("Failed to read STL file");
	}
	// �������������ӳ����
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(reader->GetOutputPort());

	// ������Ա��������������
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// ���ò�������
	actor->GetProperty()->SetDiffuse(0.8);
	actor->GetProperty()->SetColor(colors->GetColor3d(color).GetData());
	actor->GetProperty()->SetSpecular(0.3);
	actor->GetProperty()->SetSpecularPower(60.0);
	actor->GetProperty()->SetOpacity(opacity);
	return actor;
}

void showActors(std::vector<vtkSmartPointer<vtkActor>> actors, const std::string& window_name) 
{
	// ������Ⱦ��
	vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();

	// ��������Ա��ӵ���Ⱦ����
	for (auto actor : actors) 
	{
		ren->AddActor(actor);
	}

	// ���ñ�����ɫΪ��ɫ
	ren->SetBackground(1.0, 1.0, 1.0);

	// ������������Ⱦ����
	//vtkRenderWindow* win = vtkRenderWindow::New();
	vtkSmartPointer<vtkRenderWindow> win = vtkSmartPointer<vtkRenderWindow>::New();
	win->AddRenderer(ren);
	win->SetWindowName(window_name.c_str());

	// ����������
	//vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
	vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	iren->SetRenderWindow(win);

	// ���ý�������ʽΪTrackballCamera���
	//vtkInteractorStyleTrackballCamera* style = vtkInteractorStyleTrackballCamera::New();
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	iren->SetInteractorStyle(style);

	// ������ͼ����Ӧ������Ա
	ren->ResetCamera();

	// ��Ⱦ����������ѭ��
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

std::vector<float> getPointsMean(std::vector<float>& points)
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

	// 1����������
	Eigen::RowVector3d centroid = cloud.colwise().mean();

	// 2��ȥ����
	Eigen::MatrixXd demean = cloud;
	demean.rowwise() -= centroid;

	// ����Э�������
	Eigen::MatrixXd covariance = demean.transpose() * demean;

	// ��Э����������SVD�ֽ⣬ֻ�����ݵ�V����
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinV);

	// ��ȡ������������С����ֵ��Ӧ��������������
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
	bound_points = all_line_points[min_index];
	center_new = all_line_centers[min_index];
	cut_plane_area = all_line_areas[min_index];
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
		:param source_point: ƽ�����һ��,xyz
		:param plane_normal: ƽ�淨����, xyz
		:param plane_center: ƽ�淨������ƽ��Ľ���, xyz
		:return : projected_point: ƽ����һ���ڸ�ƽ���ϵ�ͶӰ��,xyz,
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
	vtkSmartPointer<vtkActor> axis_normalX_actor = createLineActorByNormal( axis_origin, axis_normalX, 15.0, 3.0, "Red");
	vtkSmartPointer<vtkActor> axis_normalY_actor = createLineActorByNormal( axis_origin, axis_normalY, 20.0, 3.0, "Green");
	vtkSmartPointer<vtkActor> axis_normalZ_actor = createLineActorByNormal( axis_origin, axis_normalZ, 25.0, 3.0, "Blue");
	std::vector<vtkSmartPointer<vtkActor>> axis_actors = { axis_normalX_actor, axis_normalY_actor, axis_normalZ_actor };
	return axis_actors;
}

void getTheMinCutPlaneArea(float& cut_plane_area_min, std::vector<float>& bound_points_min, std::vector<std::vector<float>>& rotate_matrix_min,
	std::vector<float>& center_min, const std::vector<float>& rotate_normal, const std::vector<float>& target_normal, 
	const std::vector<float>& target_center, vtkSmartPointer<vtkPolyData> target_poly_data, float max_rotate_angle, float dis_threshold)
{
	/*
		func:����������ĳһ����һ����Χ����ת��������С���Ǹ����档��ת����target_normal������ת��rotate_normal��ת
		rotate_normal : ��ת��
		target_normal : ��ת����(��λ����)
		target_center : ��ת������ԭ��
		target_poly_data : ������stl������ƽ�棨target_center, target_normal����stl�е�����
		max_rotate_angle : ���������Ƕȣ� - rotate_angle, rotate_angle��1���������Ƕ�Ϊ1��
		dis_threshold: �������ĵ㣨��������ĵ�ͳ�ʼ�����ĵ㣩֮��������������
	*/
	std::vector<float> cut_plane_areas;
	std::vector<std::vector<std::vector<float>>> rotate_matrixs;
	std::vector<std::vector<float>> bound_points;
	std::vector<std::vector<float>> centers;

	for (int i = int(-max_rotate_angle); i< int(max_rotate_angle); i += 3)
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

		if (cur_bound_points.size() < 10 * 3) { continue; }

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
	func:�˺����Ĺ����ǣ�������ָ������ת����ת�������ɵ���ת�������ҳ˾���point_new = point * Matrix.
	��˾�����ҳ˾���Ķ���:�������ʱ��������һ�࣬�þ���ͳ�Ϊ��Ӧ��ľ���.
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
		func:����������ĳһ���ᣬ��һ����Χ���ƶ���������С���Ǹ����档 

		return:
		cut_plane_area:��С��������
		bound_points_min:��С�����������
		center_min:��С��������ĵ�

		input:
		target_center : �����ʼ�����ĵ�
		target_normal : ����ķ�����
		target_target_poly_data : ������stl
		max_step : ����normal�����ƶ� - max_step��max_step
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
		if (cur_bound_points.size() < 10 * 3) { continue; }
		all_cut_plane_areas.push_back(cur_cut_plane_area);
		all_bound_points.push_back(cur_bound_points);
		all_centers.push_back(cur_center_new);
		cur_step += 0.5; //ÿ��0.5mm����
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
	func:�˺����Ĺ�����׵����ͨ���ķ��򣬾��������ǣ���֪����������ϵ������������ϵ��x�᷽������z����תһ���Ƕȣ��õ���
         �������׵����ͨ���ķ���
	input:
	rotate_angle:��ת�Ƕ�
	axis_normalX:
	axis_normalY:
	axis_normalZ:

	return:
	normal:׵����ͨ��������
	*/
	auto rotate_matrix = createRotateMatrixAroundNormal(axis_normalZ, rotate_angle);
	auto pedicle_pipeline_normal = getVectorDotMatrixValue(axis_normalX, rotate_matrix);
	return pedicle_pipeline_normal;
}

std::vector<std::vector<float>> getIntersectPointsFromLineAndPolyData(std::vector<float>& point0, std::vector<float>& point1, vtkSmartPointer<vtkPolyData> poly_data)
{
	/*
	func:�˺����Ĺ����Ǽ���׵����ͨ����׵���ཻ����������
	author:BigPanda
	date:2025.02.14
	
	input: 
		point0:׵����ͨ�������ϵĵ�0
		point1:׵����ͨ�������ϵĵ�1
		target_poly_data:������polydata

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
	func:�˺����Ĺ����ǣ�����׵����ͨ����׵������н����У���׵����ͨ�����ĵ������������
	author:BigPanda
	date:2025.02.14

	input:
		all_points:���еĽ���
		source_point:Ŀ��㣨׵����ͨ�����ĵ㣩
		reference_normal:�ο���������������������㰴�ո÷����������

	return:
		point0:Ŀ���0
		point1:Ŀ���1
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
		if (cur_dis > min_dis && cur_dis < sec_min_dis)
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
	if (left_angle > 45.0)
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
	if (right_angle > 45.0)
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
	//############### ����������ĵ㣬�滻֮ǰ��4����������ϵ����ĵ�   #############
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
		left_axis_normalX, left_axis_normalY, left_cut_plane_center, spine_poly_data, 45.0, TwoCentersDistanceThreshold);


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
		right_axis_normalX, right_axis_normalY, right_cut_plane_center, spine_poly_data, 45.0, TwoCentersDistanceThreshold);


	if (right_bound_points_min1.size() > 0)
	{
		std::vector<vtkSmartPointer<vtkActor>> right_bound_points_min1_actor = createPointsActor(right_bound_points_min1, 0.2, 1.0, "red");
		all_actors.insert(all_actors.end(), right_bound_points_min1_actor.begin(), right_bound_points_min1_actor.end());
		right_bound_points_min = right_bound_points_min1;
	}

	//############step1_3:  ����׵��������ϵ�����ĵ�  ##############
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
		left_axis_normalZ_new, left_axis_normalY, left_cut_plane_center, spine_poly_data, 45.0, TwoCentersDistanceThreshold);


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
		right_axis_normalZ_new, right_axis_normalY_new, right_cut_plane_center, spine_poly_data, 45.0, TwoCentersDistanceThreshold);


	if (right_bound_points_min2.size() > 0)
	{
		std::vector<vtkSmartPointer<vtkActor>> right_bound_points_min2_actor = createPointsActor(right_bound_points_min2, 0.2, 0.8, "yellow");
		all_actors.insert(all_actors.end(), right_bound_points_min2_actor.begin(), right_bound_points_min2_actor.end());
		right_bound_points_min = right_bound_points_min2;
	}

	//############step2_3:  ����׵��������ϵ�����ĵ�  ##############
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

	//############step3_1: ����y�᷽��������С���� ##############
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

	//############step3_2: ���½������ĵ�   ##############
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
	std::cout << "���� Y��������С������ʱ used:" << double(end4 - end3) / CLOCKS_PER_SEC 
		<<"/"<< double(end4 - start)/ CLOCKS_PER_SEC << std::endl;

	//############## step4: �������׵��������ϵ�ķ���  #############
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

	//step5:������������ϵ����������ϵ��Z�����ǣ�top_points��ϵ�ƽ��ķ�����
	std::vector<float> top_fit_plane_normal;
	vtkSmartPointer<vtkActor> top_fit_plane_actor;
	std::vector<float> top_fit_plane_center;
	top_fit_plane_actor = fitPlaneActorFromPoints(top_fit_plane_center, top_fit_plane_normal, top_points, "Red");

	std::vector<float> spine_axis_normalZ;
	if (getTwoVectorDotValue(top_fit_plane_normal, left_axis_normalZ_new2) < 0.0f)
	{
		spine_axis_normalZ.push_back((-top_fit_plane_normal[0] + (left_axis_normalZ_new2[0] + right_axis_normalZ_new2[0]) / 2.0) / 2.0);
		spine_axis_normalZ.push_back((-top_fit_plane_normal[1] + (left_axis_normalZ_new2[1] + right_axis_normalZ_new2[1]) / 2.0) / 2.0);
		spine_axis_normalZ.push_back((-top_fit_plane_normal[2] + (left_axis_normalZ_new2[2] + right_axis_normalZ_new2[2]) / 2.0) / 2.0);
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
	std::cout << "������������ϵ��ʱ:" << double(end5 - end4) / CLOCKS_PER_SEC
		<< "/" << double(end5 - start) / CLOCKS_PER_SEC << std::endl;

	//############ step6_1:����׵����ͨ������  #############
	float rot_angle = 15;

	left_angle = getAngleOfCutPlaneNormalAndSpineAxisNormalX(left_cut_plane_center, left_cut_plane_normal,
		spine_axis_center, spine_axis_normalZ, spine_axis_normalX);

	std::cout << "cal left angle:" << left_angle << std::endl;
	if (left_angle < rot_angle) {left_angle = rot_angle;}

	right_angle = getAngleOfCutPlaneNormalAndSpineAxisNormalX(right_cut_plane_center, left_cut_plane_normal,
		spine_axis_center, spine_axis_normalZ, spine_axis_normalX);
	std::cout << "cal right angle:" << right_angle << std::endl;

	if (right_angle < rot_angle) { right_angle = rot_angle; }

	auto pedicle_pipeline_L_normal = createPediclePipelineNormal(left_angle, spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ);
	auto pedicle_pipeline_R_normal = createPediclePipelineNormal(-right_angle, spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ);

	std::vector<float> left_point0 = { float(left_cut_plane_center[0] - 30.0*pedicle_pipeline_L_normal[0]),
		float(left_cut_plane_center[1] - 30.0*pedicle_pipeline_L_normal[1]),
		float(left_cut_plane_center[2] - 30.0*pedicle_pipeline_L_normal[2]) };
	std::vector<float> left_point1 = { float(left_cut_plane_center[0] + 70.0*pedicle_pipeline_L_normal[0]),
	float(left_cut_plane_center[1] + 70.0*pedicle_pipeline_L_normal[1]),
	float(left_cut_plane_center[2] + 70.0*pedicle_pipeline_L_normal[2]) };
	auto pedicle_pipeline_actor_L = createLineActorByPoints(left_point0, left_point1, 4.0, "magenta");

	std::vector<float> right_point0 = { float(right_cut_plane_center[0] - 30.0*pedicle_pipeline_R_normal[0]),
		float(right_cut_plane_center[1] - 30.0*pedicle_pipeline_R_normal[1]),
		float(right_cut_plane_center[2] - 30.0*pedicle_pipeline_R_normal[2]) };

	std::vector<float> right_point1 = { float(right_cut_plane_center[0] + 70.0*pedicle_pipeline_R_normal[0]),
		float(right_cut_plane_center[1] + 70.0*pedicle_pipeline_R_normal[1]),
		float(right_cut_plane_center[2] + 70.0*pedicle_pipeline_R_normal[2]) };
	auto pedicle_pipeline_actor_R = createLineActorByPoints(right_point0, right_point1, 4.0, "yellow");

	clock_t end6 = clock();
	std::cout << "����׵����ͨ��������ʱ:" << double(end6 - end5) / CLOCKS_PER_SEC
		<< "/" << double(end6 - start) / CLOCKS_PER_SEC << std::endl;

	//############ step6_2:����׵����ͨ��������׵�彻��  #############

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
	std::cout << "׵����ͨ��������׵�彻����ʱ:" << double(end7 - end6) / CLOCKS_PER_SEC
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
	std::cout << "����ͼƬ��ʱ:" << double(end8 - end7) / CLOCKS_PER_SEC
		<< "/" << double(end8 - start) / CLOCKS_PER_SEC << std::endl;

}
