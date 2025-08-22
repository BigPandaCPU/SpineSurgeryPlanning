#include <open3d/Open3D.h>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>
#include <fstream>
#include "stdio.h"
#include <ctime>
#include <iostream>
#include "PedicleSurgeryPlanning.h"
#include "SpinePointCloudSeg.h"
#include "SpineRegistrationICP.h"

bool fileExists(const std::string &file_name) 
{
	std::ifstream file(file_name.c_str());
	return file.good();
}


int main(int argc, char* argv[])
{
	int status;
	//assert(argc >= 3);
	std::string stl_dir = argv[1];
	std::string save_png_dir = argv[2];
	//std::string aim_label = argv[3];
	bool use_cuda = true;

	/*if (argc > 3)
	{
		std::string engine_mode = argv[3];
		if (engine_mode == "cpu")
		{
			use_cuda = false;
		}
	}*/
	
	std::vector<std::string> png_names = list_directory(stl_dir); /*{"sub-verse072_seg-vert_msk_label_24.stl",
										  /*"sub-verse070_seg-vert_msk_label_24.stl",
										  "sub-verse146_seg-vert_msk_label_24.stl",};*///list_directory(stl_dir);*/
	std::vector<std::string> stl_names;

	for (int i = 0; i < png_names.size(); i++)
	{
		auto png_name = png_names[i];
		string cur_label = png_name.substr(png_name.size() - 6, 2);
		//if (cur_label != aim_label) { continue; }
		stl_names.push_back(png_name.replace(png_name.size() - 4, 4, ".stl"));
	}

	std::cout << "stl file count " << stl_names.size() << std::endl;
	int count = 0;
	string method = string("ICP");

	for (int i = 0; i < stl_names.size(); i++)
	{
		clock_t start0 = clock();
		auto stl_name = stl_names[i];
		auto tmp = stl_names[i];
		string save_png_file = save_png_dir + "/" + tmp.replace(tmp.size() - 4, 4, ".png");
		//if (fileExists(save_png_file)) { continue; }

		//if (stl_name == "sub-verse004_seg-vert_msk_label_24.stl")
		//{
		//	cout << "aim stl." << endl;
		//}
	/*	auto m = stl_name.find("1.3.6.1.4.1.9328.50.4.0019_seg_label_20");
		if (m == string::npos)
		{
			continue;
		}*/
		auto n = stl_name.find("label_");
		
		string cur_label = stl_name.substr(n+6, 2);
		//std::cout << "cur label:" << cur_label << std::endl;
		//if (cur_label != "09") { continue; }
	
		if (cur_label == "25") { cur_label = "24"; }
		if (cur_label == "28") { cur_label = "19"; }

		string stl_file = stl_dir +string("/")+stl_name;
		cout << stl_file << endl;

		vector<float> spine_top_points, spine_left_points, spine_right_points;

		auto spine_poly_data = PedicleSurgeryPlanning::CreatePolyDataFromSTL(stl_file);
		clock_t start1 = clock();
		if (method == "PointNet")
		{
			Eigen::MatrixXd spine_points_eigen = PedicleSurgeryPlanning::GetPointsFromSTL(stl_file, POINT_NUM);
			std::vector<float> spine_points = PedicleSurgeryPlanning::MatrixToVector(spine_points_eigen);

			SpinePointCloudSeg *p = SpinePointCloudSeg::GetInstance(string("./checkpoints"), true);
			auto output_label = p->Classfier(spine_points);

			spine_top_points = SpinePointCloudSeg::GetAimPoints(spine_points, output_label, SpinePointCloudSeg::SPINE_POINT_LABEL::TOP);
			spine_left_points = SpinePointCloudSeg::GetAimPoints(spine_points, output_label, SpinePointCloudSeg::SPINE_POINT_LABEL::LEFT);
			spine_right_points = SpinePointCloudSeg::GetAimPoints(spine_points, output_label, SpinePointCloudSeg::SPINE_POINT_LABEL::RIGHT);

		}
		else
		{
			SpineRegistrationICP* icp = new SpineRegistrationICP(stl_file, cur_label, "./data/template_stl", 1000);
			icp->Registration();
			spine_top_points = icp->GetTargetTopPoints();
			spine_left_points = icp->GetTargetLeftPoints();
			spine_right_points = icp->GetTargetRightPoints();
			//icp->CreateFinalActors();
			//icp->ShowRegistrationResult();
			//icp->SaveRegistrationResult2Png(save_png_file);
			delete icp;
		}
		
		clock_t start2 = clock();

		if (spine_top_points.size() < 3 * 3 | spine_left_points.size() < 3 * 3 | spine_right_points.size() < 3 * 3) { continue; }

		//pedicleSurgeryPlanning(spine_top_points, spine_left_points, spine_right_points, spine_poly_data, stl_name, "./data/axis", save_png_dir);
		PedicleSurgeryPlanning* pedicle_plan = new PedicleSurgeryPlanning(spine_poly_data, spine_top_points, spine_left_points, spine_right_points);
		pedicle_plan->Planning();
		pedicle_plan->CreateFinalActors();
		pedicle_plan->ShowPlanResult();
		pedicle_plan->SaveSpineSurgeryPlanning2Png(save_png_file);
		delete pedicle_plan;

		//
		clock_t end = clock();
		double duration = double(end - start2) / CLOCKS_PER_SEC;
		std::cout << "part2,planing used " << duration << std::endl;

		duration = double(start2 - start1) / CLOCKS_PER_SEC;
		std::cout << "part1,get landmark points used " << duration << std::endl;

		double sum_time = double(end - start0) / CLOCKS_PER_SEC;
		std::cout << "sum used " << sum_time << std::endl;
		std::cout << std::endl;
		count += 1;
		////if (count > 5)
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

