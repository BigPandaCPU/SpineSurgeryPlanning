#include <open3d/Open3D.h>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>
#include <fstream>
#include "stdio.h"
#include "loadonnx.h"
#include "vtk_tools.h"
#include <ctime>
#include <iostream>


bool fileExists(const std::string& fileName) 
{
	std::ifstream infile(fileName);
	return infile.good();
}


int main(int argc, char* argv[])
{
	int status;
	assert(argc >= 3);
	std::string stl_dir = argv[1];
	std::string save_png_dir = argv[2];
	bool use_cuda = true;

	if (argc > 3)
	{
		std::string engine_mode = argv[3];
		if (engine_mode == "cpu")
		{
			use_cuda = false;
		}
	}
	
	

	std::vector<std::string> stl_names = list_directory(stl_dir);
	std::cout << "stl file count " << stl_names.size() << std::endl;
	int count = 0;
	for (int i = 0; i < stl_names.size(); i++)
	{
		clock_t start0 = clock();
		auto stl_name = stl_names[i];
		auto tmp = stl_names[i];
		std::string save_png_file = save_png_dir + "/" + tmp.replace(tmp.size() - 4, 4, ".png");
		if (fileExists(save_png_file)) { continue; }

		std::string stl_file = stl_dir +std::string("/")+stl_name;
		std::cout << stl_file << std::endl;

		Eigen::MatrixXd spine_points_eigen = getPointsFromSTL(stl_file, POINT_NUM);
		std::vector<float> spine_points_vector = matrixToVector(spine_points_eigen);


		std::vector<int> output_label;
		std::vector<float> spine_points_vector_normal;
		spine_points_vector_normal = pointCloudNormalize(spine_points_vector);
		output_label = classfier(spine_points_vector_normal, use_cuda);
		std::cout << "classfier done!" << std::endl;

		std::vector<float> spine_top_points, spine_left_points, spine_right_points;
		spine_top_points = getAimPoints(spine_points_vector, output_label, SPINE_POINT_LABEL::TOP);
		spine_left_points = getAimPoints(spine_points_vector, output_label, SPINE_POINT_LABEL::LEFT);
		spine_right_points = getAimPoints(spine_points_vector, output_label, SPINE_POINT_LABEL::RIGHT);




		vtkSmartPointer<vtkPolyData> spine_poly_data = vtkSmartPointer<vtkPolyData>::New();
		spine_poly_data = createPolyDataFromSTL(stl_file);


		//std::string color = "LightSteelBlue";
		//vtkSmartPointer<vtkActor> spine_actor = createActorFromSTL(stl_file, color);
		//std::vector<vtkSmartPointer<vtkActor>> top_points_actor = createPointsActor(spine_top_points, 0.5, 1.0, "red");
		//std::vector<vtkSmartPointer<vtkActor>> left_points_actor = createPointsActor(spine_left_points, 0.5, 1.0, "green");
		//std::vector<vtkSmartPointer<vtkActor>> right_points_actor = createPointsActor(spine_right_points, 0.5, 1.0, "blue");


		//std::vector<vtkSmartPointer<vtkActor> > all_actors;
		//all_actors.push_back(spine_actor);
		//all_actors.insert(all_actors.end(), top_points_actor.begin(), top_points_actor.end());
		//all_actors.insert(all_actors.end(), left_points_actor.begin(), left_points_actor.end());
		//all_actors.insert(all_actors.end(), right_points_actor.begin(), right_points_actor.end());

		//showActors(all_actors);
		clock_t start = clock();
		pedicleSurgeryPlanning(spine_top_points, spine_left_points, spine_right_points, spine_poly_data, stl_name, "./data/axis", save_png_dir);
		clock_t end = clock();
		double duration = double(end - start) / CLOCKS_PER_SEC;
		std::cout << "planing used " << duration << std::endl;
		double sum_time = double(end - start0) / CLOCKS_PER_SEC;
		std::cout << "sum used " << sum_time << std::endl;
		std::cout << std::endl;
		count += 1;
		//if (count > 5)
		//	break;
	}

	



	//std::cout << "predict done!" << std::endl;

	//char *save_file_name = "./data/point_clouds_predict.txt";

	//FILE *fp;
	//fp = fopen(save_file_name, "w");
	//for (int i = 0; i < POINT_NUM; i++)
	//{
	//	fprintf(fp, "%.3f %.3f %.3f %d\n",spine_points_vector[i*3+0], spine_points_vector[i*3+1], spine_points_vector[i*3+2], output_label[i]);
	//}
	//fclose(fp);
 //   std::cout << "predict label write done!" << std::endl;

    return 0;
}
