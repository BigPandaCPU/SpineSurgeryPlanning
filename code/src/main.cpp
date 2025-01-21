#include <iostream>
#include <vector>
#include <fstream>
#include "loadonnx.h"
#include "stdio.h"

int main()
{
    std::vector<float> points;
	std::vector<int> output_label;

    float x, y, z, label;
    char *file_name = "./0001_label_22.txt";
    FILE *fp = fopen(file_name, "r");
    while(!feof(fp))
    {
        fscanf(fp, "%f %f %f %f\n",&x,&y,&z,&label);
        points.push_back(x);
        points.push_back(y);
        points.push_back(z);
    }
    fclose(fp);

    //for(int i=0; i< 10; i++)
    //{
    //    printf("%d=%f\n",i,res[i]);
    //}
    pc_normalize(points);

    output_label = classfier(points);

	char *save_file_name = "./0001_label_22_predict.txt";

	fp = fopen(save_file_name, "w");
	for (int i = 0; i < point_num; i++)
	{
		fprintf(fp, "%d\n", output_label[i]);
	}
	fclose(fp);
    //std::cout << res << std::endl;

    return 0;
}
