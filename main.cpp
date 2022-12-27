/*
 * @Author: Aokiji996 1300833135@qq.com
 * @Date: 2022-11-23 16:49:02
 * @LastEditors: Aokiji996 1300833135@qq.com
 * @LastEditTime: 2022-11-24 10:31:25
 * @FilePath: /RC_DNN_V0.1.0/main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "yolo.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <time.h>
#include <opencv2//opencv.hpp>
#include <math.h>

#define USE_CUDA true //use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;


int main()
{

#if(defined YOLOV5 && YOLOV5==true)
	string model_path = "models/yolov5s.onnx";
#else
	string model_path = "/home/aokiji/Desktop/RC_DNN_V0.1.0/models/V0.1.0.onnx";
#endif


	Yolo test;
	Net net;
	if (test.readModel(net, model_path, USE_CUDA)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}

	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<Output> result;
	VideoCapture cap(0);
	Mat img;
	clock_t start,end;
	while(1){
		start = clock();
		cap >> img;
		if (test.Detect(img, net, result)) {
			test.drawPred(img, result, color);
			result.clear();
		}
		else {
			imshow("video", img);
			waitKey(1);
		}
		end = clock();
		cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
	}

	return 0;
}
