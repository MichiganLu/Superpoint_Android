#include "superpoint.h"

void vec2DToMat(cv::Mat &desc, std::vector<std::vector<float>> &descriptor)
{
    for (unsigned int i = 0; i < descriptor.size(); ++i)
    {
        // Make a temporary cv::Mat row and add to NewSamples _without_ data copy
        cv::Mat Sample(1, descriptor[0].size(), cv::DataType<float>::type, descriptor[i].data());
        desc.push_back(Sample);
    }
}

void point2fToKeyPoint(std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &final_kps)
{
    for (size_t i=0; i<final_kps.size(); i++)
    {
        cv::KeyPoint temp_kp(final_kps[i], 1.f);
        kps.push_back(temp_kp);
    }
}

int main()
{
    MultiEntryTimer timer2;
    //initialize model
    #ifdef USE_ANDROID
    char *dlc_path = "../models/coco_manuall2_unshaped_prob_sim.dlc";
    #else
    char *dlc_path = "/home/cvte-vm/Deep_Feature_Extract/pytorch-superpoint/implementation/superpoint_SNPE/coco_manuall2_unshaped_prob_sim.dlc";
    #endif
    Superpoint model(dlc_path);

    //input image
    #ifdef USE_ANDROID
    const char* inputFile1 = "../img/1.ppm";
    const char* inputFile2 = "../img/2.ppm";
    #else
    // const char* inputFile1 = "/media/cvte-vm/C4CE54D9CE54C4F8/3D_Datasets/HPatches/hpatches-sequences-release/i_pool/1.ppm";
    // const char* inputFile2 = "/media/cvte-vm/C4CE54D9CE54C4F8/3D_Datasets/HPatches/hpatches-sequences-release/i_pool/2.ppm";
    const char* inputFile1 = "/media/cvte-vm/C4CE54D9CE54C4F8/3D_Datasets/HPatches/hpatches-sequences-release/v_dogman/1.ppm";
    const char* inputFile2 = "/media/cvte-vm/C4CE54D9CE54C4F8/3D_Datasets/HPatches/hpatches-sequences-release/v_dogman/5.ppm";
    #endif
    std::ifstream inputList(inputFile1);
    if (!inputList) {
        std::cout << "Input image not valid. Please ensure that you have provided a valid input image for processing." << std::endl;
        return EXIT_FAILURE;
    }
    cv::Mat img1 = cv::imread(inputFile1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(inputFile2, cv::IMREAD_GRAYSCALE);
    //the following is only for imshow
    cv::Mat img1rgb = cv::imread(inputFile1);
    cv::resize(img1rgb,img1rgb,cv::Size(640,480),cv::INTER_LINEAR);
    cv::Mat img2rgb = cv::imread(inputFile2);
    cv::resize(img2rgb,img2rgb,cv::Size(640,480),cv::INTER_LINEAR);

    //computation
    std::vector<cv::Point2f> final_kps1;
    std::vector<std::vector<float>> descriptor1;
    std::vector<cv::Point2f> final_kps2;
    std::vector<std::vector<float>> descriptor2;

    for (int t=0; t<10; t++)
    {
    timer2.Start("first_img_total_processing_time");
    model.detect_and_compute(img1,final_kps1,descriptor1);
    timer2.StopAndCount("first_img_total_processing_time");
    
    timer2.Start("second_img_total_processing_time");
    model.detect_and_compute(img2,final_kps2,descriptor2);
    timer2.StopAndCount("second_img_total_processing_time");
    }
    timer2.PrintMilliSeconds();

    //show match quality
    //convert descriptor to cv::Mat
    cv::Mat desc1(0, descriptor1[0].size(), cv::DataType<float>::type);
    cv::Mat desc2(0, descriptor2[0].size(), cv::DataType<float>::type);
    vec2DToMat(desc1, descriptor1);
    vec2DToMat(desc2, descriptor2);
    //convert keypoints to std::vector<cv::KeyPoint>
    std::vector<cv::KeyPoint> kps1;
    std::vector<cv::KeyPoint> kps2;
    point2fToKeyPoint(kps1, final_kps1);
    point2fToKeyPoint(kps2, final_kps2);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SuperPoint is a floating-point descriptor NORM_L2 is used
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( desc1, desc2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.80f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    cv::Mat img_matches;
    drawMatches( img1rgb, kps1, img2rgb, kps2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    cv::imwrite("DSP_pool_0.005_0.80.jpg",img_matches);
    // cv::imshow("Good Matches", img_matches );
    // cv::waitKey(0);
    // cv::destroyAllWindows;

    // std::cout<<"mat is "<<desc1.at<float>(15,200)<<". vector is "<<descriptor1[15][200]<<std::endl;

    return 0;

}