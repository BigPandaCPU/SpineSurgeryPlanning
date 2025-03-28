#pragma once
#include "vtk_tools.h"
#include <iostream>
#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkNamedColors.h>
#include <vtkSphereSource.h>
#include <Eigen/Dense>
#include <open3d/Open3D.h>

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
using namespace std;
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

//enum SPINE_POINT_LABEL { TOP = 1, LEFT = 2, RIGHT = 3 };

using namespace std;
class PedicleSurgeryPlanning
{
public:
	PedicleSurgeryPlanning(vtkSmartPointer<vtkPolyData> spine_poly_data,
		vector<float> top_points, vector<float> left_points,vector<float> right_points);

	static vtkSmartPointer<vtkActor> CreateActorFromSTL(string stl_file, const string &color, float opacity=1.0);
	static vtkSmartPointer<vtkPolyData> CreatePolyDataFromSTL(string stl_file);
	static vtkSmartPointer<vtkActor> CreateActorFromPolyData(vtkSmartPointer<vtkPolyData>poly_data, const string &color, double opacity=1.0);
	static vector<float> NormalizeVector(const vector<float>& point);
	static vector<float> NormalizeVector(const vector<float>& point0, const vector<float>& point1);
	static float GetTwoVectorDotValue(const vector<float>& normal0, const vector<float>& normal1);
	static float CalculateAngle(const vector<float>& normal0, const vector<float>& normal1);
	static vtkSmartPointer<vtkActor> CreatePediclePipelineCylinderActor(const vector<float>& point0, const vector<float>& point1,
		float pedicle_pipleline_rate, float radius, const string& color);
	static vtkSmartPointer<vtkActor> CreateCylinderActor(const vector<float>& point0, const vector<float>& point1,
		const string& color, float opacity, float radius);

	static vtkSmartPointer<vtkActor> CreateSphereActor(vector<float>& point, float radius, float opacity, const string &color);
	static vector<vtkSmartPointer<vtkActor>> CreatePointsActor(const vector<float>& points, float radius, float opacity, const string &color);
	static void ShowActors(vector<vtkSmartPointer<vtkActor>> actors, const string& window_name);
	static void ShowAllActors(vector<vector<vtkSmartPointer<vtkActor>>>& all_actors, const string& window_name);
	static Eigen::MatrixXd GetPointsFromSTL(string stl_file, int num_points = 5000);
	static vector<float> MatrixToVector(const Eigen::MatrixXd& matrix);
	static vector<float> GetTwoVectorCrossValue(const vector<float>& vector0, const vector<float>& vector1);


	
	void Planning();
	void CreateAndShowFinalActors();
	float GetAreaOfClosedCurvePoints(vector<float>& points);
	vector<float> GetDistanceOfPoints2AimPoint(const vector<vector<float>>& points, const vector<float>& aim_point);
	float GetDistanceOfTwoPoints(const vector<float>& point0, const vector<float>& point1);

	void GetClipedCenterPoints(vector<float>& bound_points, vector<float>& center_new, float& cut_plane_area, const vector<float>& center, const vector<float>& normal);
	vector<float> GetPointsMean(const vector<float>& points);
	void FitPlaneFromPointsBySVD(vector<float>& fit_plane_center, vector<float>& fit_plane_normal, const vector<float>& points);
	void GetTheTrueFitPlaneNormal(const vector<float>& points, vector<float>& plane_center, vector<float>& plane_normal);
	void GetTheTrueFitPlaneCenter(vector<float>& cut_plane_center, vector<float>& cut_plane_normal, vector<float>& bound_points_min);
	vector<float> GetProjectedPointOnPlane(const vector<float>& source_point, const vector<float>& plane_normal, const vector<float>& plane_center);
	void GetTheMinCutPlaneArea(float& cut_plane_area_min, vector<float>& bound_points_min, vector<vector<float>>& rotate_matrix_min,
		vector<float>& center_min, const vector<float>& rotate_normal, const vector<float>& target_normal,
		const vector<float>& target_center, float max_rotate_angle, float dis_threshold, float area_threshold,
		int bound_points_threshold);
	vector<vector<float>> CreateRotateMatrixAroundNormal(const vector<float>& rotate_normal, float rotate_angle);
	vector<float> GetVectorDotMatrixValue(const vector<float>& normal, const vector<vector<float>>& rotate_matrix);
	void GetTheMinCutPlaneAreaAlongAxisY(float& cut_plane_min_area, vector<float>& bound_points_min, vector<float>& center_min,
		const vector<float>& target_center, const vector<float>& target_normal,	float max_step = SearchAlongAxisYStep);
	float GetAngleOfCutPlaneNormalAndSpineAxisNormalX(const vector<float>& cut_plane_center, const vector<float>& cut_plane_normal,
		const vector<float>& axis_origin, const vector<float>& axis_normalZ, const vector<float>& axis_normalX);
	vector<float> CreatePediclePipelineNormal(float rotate_angle, const vector<float>& axis_normalX,
		const vector<float>& aixs_normalY, const vector<float>& axis_normalZ);

	void GetTheClosedTwoPoints(vector<float>& point0, vector<float>& point1, const vector<vector<float>>& all_points,
		const vector<float>& source_point, const vector<float>& reference_normal);
	vector<float> GetTheMeanOfTwoPoints(const vector<float>& pointsA, const vector<float>& pointsB);
	vector<float> GetTheMinusOfTwoPoints(const vector<float>& pointsA, const vector<float>& pointsB, float factors = 1.0);
	void ReverseNormal(vector<float>& normal);
	vector<vector<float>> GetIntersectPointsFromLineAndPolyData(vector<float>& point0, vector<float>& point1);

	~PedicleSurgeryPlanning();


private:
	vtkSmartPointer<vtkPolyData> m_spine_poly_data;
	vector<float> m_top_points;
	vector<float> m_left_points;
	vector<float> m_right_points;

	vector<float> m_top_points_center;
	vector<float> m_left_points_center;
	vector<float> m_right_points_center;

	vector<float> m_top_plane_center;
	vector<float> m_top_plane_normal;

	vector<float> m_left_cut_plane_center;
	vector<float> m_left_cut_plane_normal;
	vector<float> m_right_cut_plane_center;
	vector<float> m_right_cut_plane_normal;
	vector<float> m_left_bound_points_min;
	vector<float> m_right_bound_points_min;

	vector<float> m_left_axis_normalX;
	vector<float> m_left_axis_normalY;
	vector<float> m_left_axis_normalZ;

	vector<float> m_right_axis_normalX;
	vector<float> m_right_axis_normalY;
	vector<float> m_right_axis_normalZ;

	vector<float> m_spine_axis_normalX;
	vector<float> m_spine_axis_normalY;
	vector<float> m_spine_axis_normalZ;
	vector<float> m_spine_axis_center;

	vector<float> m_pedicle_pipeline_L_normal;
	vector<float> m_pedicle_pipeline_R_normal;

	vector<float> m_left_pedicle_start_point;
	vector<float> m_left_pedicle_end_point;
	vector<float> m_right_pedicle_start_point;
	vector<float> m_right_pedicle_end_point;
	bool m_plan_result;

};