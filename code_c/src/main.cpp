#include "vtk_tools.h"
using namespace std;
int main()
{
	string source_ply_file = 
		"D:/project/PyProject_tmp/Fast-Global-Registration-main/dataset/spine/source_24_bak.ply";
	string target_ply_file = 
		"D:/project/PyProject_tmp/Fast-Global-Registration-main/dataset/spine/target_24.ply";
	
	auto source_points_eigen = getPointsFromPLY(source_ply_file, 1000);
	auto target_points_eigen = getPointsFromPLY(target_ply_file, 1000);

	vector<float> source_points_vector = matrixToVector(source_points_eigen);
	vector<float> target_points_vector = matrixToVector(target_points_eigen);


	vector<vector<float>> source_eigen_vectors;
	
	vector<vector<float>> target_eigen_vectors;
	
	auto pre_matrix = preAlignedTwoPointClouds(source_points_vector, target_points_vector,
		source_eigen_vectors, target_eigen_vectors);

	vector<float> source_points_new;
	vectorPointsDotvtkMatrix4x4(source_points_vector, pre_matrix, source_points_new);
	auto source_points_actors = createPointsActor(source_points_vector, 1.0, 0.5, "red");
	auto source_points_new_actors = createPointsActor(source_points_new, 1.0, 0.5, "red");
	auto target_points_actors = createPointsActor(target_points_vector, 1.0, 0.5, "LightSteelBlue");
	vector<vtkSmartPointer<vtkActor>> all_actors;// = source_points_actors;
	std::cout << "source points num:" << source_points_actors.size() << std::endl;
	all_actors.insert(all_actors.begin(), source_points_new_actors.begin(), source_points_new_actors.end());
	all_actors.insert(all_actors.begin(), target_points_actors.begin(), target_points_actors.end());
	showActors(all_actors, "show result");

	//method1: ICP get landmark points
	//registrationPolydata(const std::string& label_name, const std::string& template_stl_dir,
	//	const vtkSmartPointer<vtkPolyData> target_polydata, const std::vector<float>& target_points,
	//	std::vector<float>& left_points, std::vector<float>& right_points, std::vector<float>& top_points)

	//method2:PointNet get landmark points
	//std::vector<int> classfier(std::vector<float>& points, bool use_cuda = false);

	//planning
	//pedicleSurgeryPlanning(std::vector<float>& top_points, std::vector<float>& left_points, std::vector<float>& right_points,
	//vtkSmartPointer<vtkPolyData> spine_poly_data, std::string window_name, std::string save_axis_dir, std::string save_png_dir)
	return 0;
}