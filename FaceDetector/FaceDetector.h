//
// Created by dl on 19-7-19.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include "OnnxModel.h"
#include <chrono>
using namespace std::chrono;

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};
struct Point {
	float _x;
	float _y;
};

struct bbox {
	float x1;
	float y1;
	float x2;
	float y2;
	float s;
	Point point[5];
};

struct box {
	float cx;
	float cy;
	float sx;
	float sy;
};

class Detector:public OnnxModel
{
public:
    Detector();
    ~Detector();

    void Init(const std::string &model_param, const std::string &model_bin);

    Detector(const std::string &model_param, const std::string &model_bin, bool retinaface = false);

    inline void Release();

    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);

    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    void postProcess(const std::vector<MemoryMan*>&resultVec, std::vector<bbox>& boxes
        , int orgW, int orgH, int dstW, int dstH);

public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;
    std::vector<box> m_anchor;

    int m_lastDstWidth = 0;
    int m_lastDstHeight = 0;
};
#endif //
