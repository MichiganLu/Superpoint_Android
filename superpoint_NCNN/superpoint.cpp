#include "superpoint.h"

MultiEntryTimer timer;

Superpoint::Superpoint(char *param, char *bin)
{
    param_path = param;
    bin_path = bin;
    superpoint.opt.use_vulkan_compute = false;
    if (superpoint.load_param(param))
        exit(-1);
    if (superpoint.load_model(bin))
        exit(-1);
}

void Superpoint::preprocess(cv::Mat &img, ncnn::Mat &in)
{
    int w = img.cols;
    int h = img.rows;
    in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 320, 240);
    const float mean_vals[1] = {0.f};
    const float norm_vals[1] = {1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
}

void Superpoint::extract_points(std::vector<std::vector<float>> &kps, const ncnn::Mat &out1, const float threshold)
{
    //each keypoint has three elements, width, height, confidence
    for (int h=0; h<out1.h; h++)
    {
        for (int w=0; w<out1.w; w++)
        {
            for (int c=0; c<out1.c; c++)
            {
                if (out1[w+h*out1.w+c*out1.h*out1.w] > threshold)  
                {
                    std::vector<float> temp = {float(w), float(h), out1[w+h*out1.w+c*out1.h*out1.w]};
                    kps.push_back(temp);
                }
            }
        }
    }
    //sort kps by confidence in descending order
    int idx = 2;
    std::sort(kps.begin(), kps.end(), [idx](const std::vector<float>& a, const std::vector<float>& b) {
        return a[idx] > b[idx];
    });
    //print for debug
    // for (size_t i = 0; i < kps.size(); i++) {
    //     for (size_t j = 0; j < kps[i].size(); j++) {
    //         std::cout << kps[i][j] << (j + 1 < kps[i].size() ? ' ' : '\n');
    //     }
    // }
}

void Superpoint::nms(std::vector<std::vector<float>> &kps, std::vector<std::vector<float>> &nms_kps, const int &nms_radius)
{
    //nms by confidence
    //kps is in width(column), height(row), notice that you should access element by matrix[height][width]
    //within nms_map: 1 is occupied, 0 is free to occupy, -1 is suppressed
    //nms radius is 4 pixels
    cv::Mat nms_map(240+2*nms_radius, 320+2*nms_radius, CV_32SC1, cv::Scalar::all(0));    //the +2*nms_radius accounts for padding on the left, right, above, below
    for (size_t i=0; i<kps.size(); i++)    //iterate over points in descending order based on confidence
    {
        int temp_w = int(kps[i][0])+nms_radius;     //temp_w +nms_radius: offset for padding, same for temp_h
        int temp_h = int(kps[i][1])+nms_radius;
        if (nms_map.ptr<int>(temp_h)[temp_w] == 0)
        {
            //suppresing
            for (int x=-nms_radius; x<=nms_radius; x++)
            {
                for (int y=-nms_radius; y<=nms_radius; y++)
                {
                    nms_map.ptr<int>(temp_h+x)[temp_w+y] = -1;
                }
            }
            //occupying the corner
            nms_map.ptr<int>(temp_h)[temp_w] = 1;
        }
    }
    //extract unsuppressed points
    for (int r=2*nms_radius; r<=nms_map.rows-2*nms_radius; r++)     //-2*nms_radius to get rid of padding and corner point
    {
        for (int c=2*nms_radius; c<=nms_map.cols-2*nms_radius; c++)
        {
            if (nms_map.ptr<int>(r)[c] == 1)
            {
                int width = c-nms_radius;   //you have offseted width and height, you need to cancel the offset when extract points
                int height = r-nms_radius;
                std::vector<float> temp = {float(width),float(height)};  //nms_kps also in width, height
                nms_kps.push_back(temp);
            }
        }
    }
    //print for debug
    // int idx = 0;
    // std::sort(nms_kps.begin(), nms_kps.end(), [idx](const std::vector<float>& a, const std::vector<float>& b) {
    //     return a.at(idx) > b.at(idx);
    // });
    // for (size_t i = 0; i < nms_kps.size(); i++) {
    //     for (size_t j = 0; j < nms_kps[i].size(); j++) {
    //         std::cout << nms_kps[i][j] << (j + 1 < nms_kps[i].size() ? ' ' : '\n');
    //     }
    // }
}

void Superpoint::extract_descriptor(const ncnn::Mat &out2, const std::vector<std::vector<float>> &nms_kps, std::vector<std::vector<float>> &descriptor)
{
    for (size_t i=0; i<nms_kps.size(); i++)
    {
        std::vector<float> temp_descriptor(256);
        int width = int(nms_kps[i][0]);
        int height = int(nms_kps[i][1]);
        for (int c=0; c<256; c++)    //iterate over channel to extract descriptor, 256 is the channel length
        {
            temp_descriptor[c] = out2[width+height*out2.w+c*out2.w*out2.h];
        }
        descriptor.push_back(temp_descriptor);
    }
}

void Superpoint::get_subpixel_coordinate(const ncnn::Mat &out1, const std::vector<std::vector<float>> &nms_kps, std::vector<cv::Point2f> &final_kps)
{
    for (size_t i=0; i<nms_kps.size(); i++)
    {
        int width = int(nms_kps[i][0]);
        int height = int(nms_kps[i][1]);
        float final_width=0;
        float final_height=0;
        float total_weight=0;

        //the next two for loops do weighted addition
        for (int x=-2; x<=2; x++)    //I define x to be the width
        {
            for (int y=-2; y<=2; y++)   //I define y to be the height
            {
                total_weight = total_weight + out1[(width+x)+(height+y)*out1.w];
            }
        }
        for (int x=-2; x<=2; x++)    //I define x to be the width
        {
            for (int y=-2; y<=2; y++)   //I define y to be the height
            {
                final_width = final_width + out1[(width+x)+(height+y)*out1.w]/total_weight*(width+x);
                final_height = final_height + out1[(width+x)+(height+y)*out1.w]/total_weight*(height+y);
            }
        }
        cv::Point2f final_point(final_width*2, final_height*2);    //multiple 2 to scale back to 640,480
        final_kps.push_back(final_point);
    }
}

void Superpoint::detect_and_compute(cv::Mat &img, std::vector<cv::Point2f> &final_kps, std::vector<std::vector<float>> &descriptor)
{
    //preprocess image
    timer.Start("preprocess");
    ncnn::Mat in;
    preprocess(img, in);
    timer.StopAndCount("preprocess");

    //forward
    timer.Start("forward");
    ncnn::Extractor ex = superpoint.create_extractor();
    ex.set_num_threads(4);
    ncnn::Mat out1;   //out1 for keypoints heatmap, of dim [1,1,240,320]
    ncnn::Mat out2;   //out2 for keypoints descriptor, of dim [1,256,240,320]
    ex.input("input.1", in);
    ex.extract("output.1", out1);
    ex.extract("output.2", out2);
    timer.StopAndCount("forward");

    //extract keypoints
    timer.Start("extract_kps");
    float threshold = 0.003;
    std::vector<std::vector<float>> kps;    //kps of dim [N,3], 3 for width, height, confidence
    extract_points(kps, out1, threshold);
    timer.StopAndCount("extract_kps");

    //nms
    timer.Start("nms");
    int nms_radius = 4;
    std::vector<std::vector<float>> nms_kps; //nms_kps of dim [N,2], 2 for width, height; because confidence is not needed anymore
    nms(kps, nms_kps, nms_radius);
    timer.StopAndCount("nms");

    //extract descriptor
    timer.Start("extract_des");
    extract_descriptor(out2, nms_kps, descriptor);    //descriptor of dim [N, 256], N for number of nmsed keypoints, 256 for descriptor length
    timer.StopAndCount("extract_des");

    //find subpixel position of kps
    timer.Start("subpixel");
    get_subpixel_coordinate(out1, nms_kps, final_kps);
    timer.StopAndCount("subpixel");

    timer.PrintMilliSeconds();
}




