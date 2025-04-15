#pragma once
#include <open3d/Open3D.h>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>

#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkNamedColors.h>
#include <vtkSphereSource.h>


#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dirent.h>
#endif
#include <stdexcept>

//#define PI 3.1415926
//#define TwoCentersDistanceThreshold 5.0
//#define CrossAngleThreshold 60.0  //切面法向与参考法向之间的夹角阈值
//#define SearchRotateAngle 60.0    //最小截面搜索的时候，搜索的角度范围-SearchRotateAngle ~ SearchRotateAngle
//#define CutPlaneAreaThreshold 10.0 //椎弓根通道最小截面的面积阈值 10mm^2
//#define CutBoundPointsThreshold 20 //椎弓根通道截面的交点个数的阈值 20个
//#define SearchAlongAxisYStep 5.0   //沿着切面法向法向最小截面搜索的最大范围，默认与TwoCenterDistanceThreshold相等
//#define PolyDataDownSampleNumPolyThreshold 50000 //对spine poly 进行降采样，当poly的个数超过50000的时候进行处理

#define PI 3.1415926
#define TwoCentersDistanceThreshold 5.0
#define CrossAngleThreshold 60.0  //切面法向与参考法向之间的夹角阈值
#define SearchRotateAngle 60.0    //最小截面搜索的时候，搜索的角度范围-SearchRotateAngle ~ SearchRotateAngle
#define CutPlaneAreaThreshold 10.0 //椎弓根通道最小截面的面积阈值 10mm^2
#define CutBoundPointsThreshold 20 //椎弓根通道截面的交点个数的阈值 20个
#define SearchAlongAxisYStep 5.0   //沿着切面法向法向最小截面搜索的最大范围，默认与TwoCenterDistanceThreshold相等
#define PolyDataDownSampleNumPolyThreshold 50000 //对spine poly 进行降采样，当poly的个数超过50000的时候进行处理
#define PedicleCrossAngle 15.0 //椎弓根通道的夹角
#define PediclePiplelineRate 0.8  //螺钉长度占整个椎弓根通道的比例
#define PedicleScrewRadius 1.5  //螺钉的半径，单位mm
#define POINT_NUM 5000

enum SPINE_POINT_LABEL {TOP=1, LEFT=2, RIGHT=3};

std::vector<std::string> list_directory(const std::string& dirPath, bool recursive=false);
std::vector<float> pointCloudNormalize(const std::vector<float>& points);

std::vector<float> randomChoice(const std::vector<float>& spine_points, int num_points, bool replace = false);

std::vector<float> matrixToVector(const Eigen::MatrixXd& matrix);

Eigen::MatrixXd getPointsFromSTL(std::string stl_file, int num_points = 5000);
Eigen::MatrixXd getPointsFromPLY(std::string ply_file, int num_points = 5000);

vtkSmartPointer<vtkPolyData> createPolyDataFromSTL(const std::string& stl_file);

vtkSmartPointer<vtkActor> createCirclePlaneActor(const std::vector<float>& plane_center, 
	const std::vector<float>& plane_normal, float radius = 1.0, float opacity=1.0, 
	const vtkStdString &color = "Red");

vtkSmartPointer<vtkActor> createActorFromSTL(const std::string& stl_file, const vtkStdString &color ="Cornsilk", double opacity=1.0);

vtkSmartPointer<vtkActor> createActorFromPolyData(vtkSmartPointer<vtkPolyData> polydata, 
	const vtkStdString &color ="Cornsilk", double opacity = 1.0);

void showActors(std::vector<vtkSmartPointer<vtkActor> > actors, const std::string& window_name = "show spine");

vtkSmartPointer<vtkActor> createSphereActor(std::vector<float>& point, float radius, float opacity = 0.5, const vtkStdString &color = "Cornsilk");
std::vector<vtkSmartPointer<vtkActor>> createPointsActor(const std::vector<float>& points, float radius = 1.0, float opacity = 0.5, const vtkStdString &color = "Cornsilk");
std::vector<float> getAimPoints(const std::vector<float>& points, const std::vector<int>& labels, SPINE_POINT_LABEL aim_label);

std::vector<float> getPointsMean(const std::vector<float>& points);

void fitPlaneFromPointsBySVD(std::vector<float>& fit_plane_center, std::vector<float>& fit_plane_normal ,const std::vector<float>& points);


vtkSmartPointer<vtkActor> fitPlaneActorFromPoints(std::vector<float>& fit_plane_center, std::vector<float>& fit_plane_normal,
	std::vector<float> points, const vtkStdString &color = "Green", float radius=10.0);

void pedicleSurgeryPlanning(std::vector<float>& top_points, std::vector<float>& left_points, std::vector<float>& right_points,
	vtkSmartPointer<vtkPolyData> spine_poly_data, std::string window_name, std::string save_axis_dir, std::string save_png_dir);

std::vector<float> normalizeVector(const std::vector<float>& point0, const std::vector<float>& point1);
std::vector<float> normalizeVector(const std::vector<float>& point);

float getTwoVectorDotValue(const std::vector<float>& normal0, const std::vector<float>& normal1);

float calculateAngle(const std::vector<float>& normal0, const std::vector<float>& normal1);
vtkSmartPointer<vtkActor> createLineActorByNormal(const std::vector<float>& point0, const std::vector<float>& normal, float length=20.0, float line_width = 4.0, const std::string& color = "red");
vtkSmartPointer<vtkActor> createLineActorByPoints(const std::vector<float>& point0, const std::vector<float>& point1, float line_width = 4.0, const std::string& color = "red");

void getClipedCenterPoints(std::vector<float>& bound_points, std::vector<float>& center_new, float& cut_plane_area, 
	const std::vector<float>& center, const std::vector<float>& normal, vtkSmartPointer<vtkPolyData> target_poly_data);

float getAreaOfClosedCurvePoints(std::vector<float>& points);

std::vector<float> getDistanceOfPoints2AimPoint(const std::vector<std::vector<float>>& points, const std::vector<float>& aim_point);
float getDistanceOfTwoPoints(const std::vector<float>& point0, const std::vector<float>& point1);
std::vector<float> getProjectedPointOnPlane(const std::vector<float>& source_point, const std::vector<float>& plane_normal, const std::vector<float>& plane_center);
std::vector<float> getTwoVectorCrossValue(const std::vector<float>& vector0, const std::vector<float>& vector1);
std::vector<vtkSmartPointer<vtkActor>> createAxisActors(const std::vector<float>& axis_origin, const std::vector<float>& axis_normalX,
	const std::vector<float>& axis_normalY, const std::vector<float>& axis_normalZ, float len=30.0);

void getTheMinCutPlaneArea(float& cut_plane_area_min, std::vector<float>& bound_points_min, std::vector<std::vector<float>>& rotate_matrix_min, 
	std::vector<float>& center_min, const std::vector<float>& rotate_normal, const std::vector<float>& target_normal, 
	const std::vector<float>& target_center, vtkSmartPointer<vtkPolyData> target_poly_data, 
	float max_rotate_angle= SearchRotateAngle, float dis_threshold= TwoCentersDistanceThreshold, 
	float area_threshold= CutPlaneAreaThreshold, int bound_points_threshold=CutBoundPointsThreshold);

std::vector<std::vector<float>> createRotateMatrixAroundNormal(const std::vector<float>& rotate_normal, float rotate_angle=0.0);
//Eigen::Matrix3f createRotateMatrixAroundNormal2(Eigen::Vector3f normal, float angle);
std::vector<float> getVectorDotMatrixValue(const std::vector<float>& normal, const std::vector<std::vector<float>>& rotate_matrix);
void getTheMinCutPlaneAreaAlongAxisY(float& cut_plane_area_min, std::vector<float>& bound_points_min, std::vector<float>& center_min,
	const std::vector<float>& target_center, const std::vector<float>& target_normal, vtkSmartPointer<vtkPolyData> target_poly_data, 
	float max_step = SearchAlongAxisYStep);

std::vector<float> createPediclePipelineNormal(float rotate_angle, const std::vector<float>& axis_normalX, 
	const std::vector<float>& aixs_normalY, const std::vector<float>& axis_normalZ);
std::vector<std::vector<float>> getIntersectPointsFromLineAndPolyData(std::vector<float>& point0, std::vector<float>& point1, 
	vtkSmartPointer<vtkPolyData> poly_data);
void getTheClosedTwoPoints(std::vector<float>& point0, std::vector<float>& point1, const std::vector<std::vector<float>>& all_points,
	const std::vector<float>& source_point, const std::vector<float>& reference_normal);
vtkSmartPointer<vtkActor> createPediclePipelineCylinderActor(const std::vector<float>& point0, const std::vector<float>& point1, 
	float pedicle_pipleline_rate = 0.8, float radius = 3.5 / 2.0, const std::string& color = "magenta");

vtkSmartPointer<vtkActor> createCylinderActor(const std::vector<float>& point0, const std::vector<float>& point1, 
	const std::string& color = "Green", float opacity = 1.0, float radius = 1.5);
void saveSpineSurgicalPlanning2Png(std::vector<vtkSmartPointer<vtkActor>> all_actors, const std::vector<float>& axis_origin, 
	const std::vector<float>& axis_normalX, const std::vector<float>& axis_normalY, const std::vector<float>& axis_normalZ,std::string& save_png_file);


float getAngleOfCutPlaneNormalAndSpineAxisNormalX(const std::vector<float>& cut_plane_center, const std::vector<float>& cut_plane_normal,
	const std::vector<float>& axis_origin, const std::vector<float>& axis_normalZ, const std::vector<float>& axis_normalX);

void registrationPolydata(const std::string& label_name, const std::string& template_stl_dir, const vtkSmartPointer<vtkPolyData> target_polydata, 
	const std::vector<float>& target_points, std::vector<float>& left_points,std::vector<float>& right_points, std::vector<float>& top_points);
bool loadLandmarksFromFile(const std::string& landmark_file, std::vector<float>& top_points, std::vector<float>& left_points,
	std::vector<float>& right_points);

vtkSmartPointer<vtkMatrix4x4> preAlignedTwoPointClouds(const std::vector<float>& source_points, const std::vector<float>& target_points,
	std::vector<std::vector<float>>& source_eigen_vectors, std::vector<std::vector<float>>& target_eigen_vectors);

void PCA(const std::vector<float> points, std::vector<float>& eigen_values, std::vector<std::vector<float>>& eigen_vectors, 
	std::vector<float>& points_center);

std::vector<float> pointsDecenter(const std::vector<float>& points, const std::vector<float>& center);

std::vector<float> getTheMaxAxisPoint(const std::vector<float>& points);
std::vector<float> getTheMinAxisPoint(const std::vector<float>& points);
std::vector<float> getPointsDotMatrix(const std::vector<float>& points, const std::vector<std::vector<float>>& matrix);
std::vector<std::vector<float>> matrix4DotMatrix4(const std::vector<std::vector<float>>& matrix1, const std::vector<std::vector<float>>& matrix2);
void printMatrix(const std::vector<std::vector<float>>& matrix);

vtkSmartPointer<vtkMatrix4x4> convertVectorMatrix3x3TovtkMatrix4x4(const std::vector<std::vector<float>>& matrix);
void vectorPointsDotvtkMatrix4x4(const std::vector<float>& points, const vtkSmartPointer<vtkMatrix4x4> left_matrix, std::vector<float>& points_new);

void getTheMinDistanceOfR(const std::vector<float>& source_points_decenter, const std::vector<std::vector<float>>& source_eigen_vectors,
	const std::vector<float>& target_points_decenter, const std::vector<std::vector<float>>& target_eigen_vectors,
	const std::vector<std::vector<std::vector<float>>>& R, int& min_index, float& min_distance, bool use_icp=false);

void showAllActors(std::vector<std::vector<vtkSmartPointer<vtkActor>>>& all_actors, const std::string& window_name);
void showAllActors3(std::vector<std::vector<vtkSmartPointer<vtkActor>>>& all_actors, const std::string& window_name);
std::vector<float> ICP(const std::vector<float>& source_points, const std::vector<float>& target_points);










