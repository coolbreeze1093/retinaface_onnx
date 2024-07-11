#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "FaceDetector.h"

using namespace std;

int main(int argc, char** argv)
{

    string imgPath="./image_detection.jpg";
    
    string param = "./FaceDetector.onnx";
    string bin = "../model/face.bin";
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, true);
    // retinaface
    // Detector detector(param, bin, true);
    Timer timer;
    for	(int i = 0; i < 10; i++){


        cv::Mat img = cv::imread(imgPath.c_str());
        cv::Mat _omat = img;
        // scale
        float long_side = std::max(img.cols, img.rows);
        float scale = 1.0;//max_side/long_side;
        //cv::Mat img_scale;
        //cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
        //cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

        if (img.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
            return -1;
        }
        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(img, boxes);
        timer.toc("----total timer:");

        
        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(_omat, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);

            cv::putText(_omat, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::circle(_omat, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(_omat, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(_omat, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(_omat, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(_omat, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
        }
        cv::imwrite("test.png", _omat);
    }
    return 0;
}

