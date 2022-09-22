#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"

#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "timer.h"

#include <cstring>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>

#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "LoadUDOPackage.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#ifdef ENABLE_GL_BUFFER
#include <GLES2/gl2.h>
#include "CreateGLBuffer.hpp"
#endif

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "timer.h"

class Superpoint
{
    private:
        char *dlc_path;
        std::string bufferTypeStr = "ITENSOR";
        std::unique_ptr<zdl::DlContainer::IDlContainer> container;
        #ifdef USE_ANDROID
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
        #else
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
        #endif
        zdl::DlSystem::RuntimeList runtimeList;
        zdl::DlSystem::PlatformConfig platformConfig;
        std::unique_ptr<zdl::SNPE::SNPE> snpe;
        bool ret = runtimeList.add(runtime);
        bool runtimeSpecified = false;
        bool execStatus = false;
        bool usingInitCaching = false;
        bool useUserSuppliedBuffers = false;
        std::string UdoPackagePath = "";    //UDO stands for user defined operation
        std::string OutputDir = "./output/";
        MultiEntryTimer timer;

    public:
        Superpoint(char *dlc_path);
        ~Superpoint(){};
        void preprocess(std::unique_ptr<zdl::SNPE::SNPE>& snpe, cv::Mat &img, zdl::DlSystem::TensorMap &inputTensorMap);
        void extract_points(std::vector<std::vector<float>> &kps, const std::vector<float> outputHeatmap, const float threshold, const int &hm_height, const int &hm_width, const int &hm_channel);
        void nms(std::vector<std::vector<float>> &kps, std::vector<std::vector<float>> &nms_kps, const int &nms_radius);
        void extract_descriptor(const std::vector<float> &outputDesc, const std::vector<std::vector<float>> &nms_kps, std::vector<std::vector<float>> &descriptor, const int &desc_height, const int &desc_width, const int &desc_channel);
        void get_subpixel_coordinate(const std::vector<float> outputHeatmap, const std::vector<std::vector<float>> &nms_kps, std::vector<cv::Point2f> &final_kps, const int &hm_width);
        void detect_and_compute(cv::Mat &img, std::vector<cv::Point2f> &final_kps, std::vector<std::vector<float>> &descriptor);
};