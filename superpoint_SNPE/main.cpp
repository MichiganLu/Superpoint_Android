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
    char *dlc_path = "../models/coco_manualL2_unshapedProb_uninterpolateDesc_sim.dlc";
    #else
    char *dlc_path = "../../../coco_manualL2_unshapedProb_uninterpolateDesc_sim.dlc";
    #endif
    Superpoint model(dlc_path);

    //input image
    #ifdef USE_ANDROID
    const char* inputFile1 = "../img/3.ppm";
    const char* inputFile2 = "../img/4.ppm";
    #else
    const char* inputFile1 = "../../../../test_img/i_pool/1.ppm";
    const char* inputFile2 = "../../../../test_img/i_pool/2.ppm";
    // const char* inputFile1 = "../../../../test_img/v_dogman/1.ppm";
    // const char* inputFile2 = "../../../../test_img/v_dogman/5.ppm";
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

    //final_kps is of dimension N, N for number of keypoints
    //descriptor is of dimension N*256, N for number of keypoints, 256 for descriptor length
    for (int t=0; t<1; t++)
    {
    // std::vector<cv::Point2f> final_kps1;
    // std::vector<std::vector<float>> descriptor1;
    // std::vector<cv::Point2f> final_kps2;
    // std::vector<std::vector<float>> descriptor2;
    // cv::Mat img11 = img1.clone();
    // cv::Mat img22 = img2.clone();
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
    //KNN
    // std::vector< std::vector<cv::DMatch> > knn_matches;
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // matcher->knnMatch( desc1, desc2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.82f;
    // std::vector<cv::DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }

    //BFMatcher
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(4, true);
    matcher.match(desc1,desc2, matches);
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 0.77)
        {
        good_matches.push_back(matches[i]);
        }
    }
    //-- Draw matches
    cv::Mat img_matches;
    drawMatches( img1rgb, kps1, img2rgb, kps2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // -- Show detected matches
    cv::imwrite("x64_ipool_0.008_0.77_top1k.jpg",img_matches);
    cv::imshow("Good Matches", img_matches );
    cv::waitKey(0);
    cv::destroyAllWindows;

    return 0;

}