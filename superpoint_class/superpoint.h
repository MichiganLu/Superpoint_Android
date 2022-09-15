#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#endif
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "timer.h"

class Superpoint
{
    private:
        char *param_path;
        char *bin_path;
        ncnn::Net superpoint;
    public:
        Superpoint(char *param, char *bin);
        ~Superpoint(){};
        void preprocess(cv::Mat &img, ncnn::Mat &in);
        void extract_points(std::vector<std::vector<float>> &kps, const ncnn::Mat &out1, const float threshold);
        void nms(std::vector<std::vector<float>> &kps, std::vector<std::vector<float>> &nms_kps, const int &nms_radius);
        void extract_descriptor(const ncnn::Mat &out2, const std::vector<std::vector<float>> &nms_kps, std::vector<std::vector<float>> &descriptor);
        void get_subpixel_coordinate(const ncnn::Mat &out1, const std::vector<std::vector<float>> &nms_kps, std::vector<cv::Point2f> &final_kps);
        void detect_and_compute(cv::Mat &img, std::vector<cv::Point2f> &final_kps, std::vector<std::vector<float>> &descriptor);
};