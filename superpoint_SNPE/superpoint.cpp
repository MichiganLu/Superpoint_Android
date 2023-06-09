#include "superpoint.h"

Superpoint::Superpoint(char *dlc_path)
{
    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc_path);
    if (!dlcFile) {
        std::cout << "Input dlc file not valid. Please ensure that you have provided a valid dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        exit(-1);
    }

    // Check runtime
    runtime = checkRuntime(runtime);

    // Load container
    container = loadContainerFromFile(dlc_path);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       exit(-1);
    }

    snpe = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, platformConfig, usingInitCaching);   //trace in to set input and output config
    
    if (snpe == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       exit(-1);
    }
    if (usingInitCaching)
    {
       if (container->save(dlc_path))
       {
          std::cout << "Saved container into archive successfully" << std::endl;
       }
       else
       {
          std::cout << "Failed to save container into archive" << std::endl;
       }
    }

    // Configure logging output and start logging. The snpe-diagview
    // executable can be used to read the content of this diagnostics file
    // auto logger_opt = snpe->getDiagLogInterface();
    // if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
    // auto logger = *logger_opt;
    // auto opts = logger->getOptions();

    // opts.LogFileDirectory = OutputDir;
    // if(!logger->setOptions(opts)) {
    //     std::cerr << "Failed to set options" << std::endl;
    //     exit(-1);
    // }
    // if (!logger->start()) {
    //     std::cerr << "Failed to start logger" << std::endl;
    //     exit(-1);
    // }
}

void Superpoint::preprocess(std::unique_ptr<zdl::SNPE::SNPE>& snpe, cv::Mat &img, zdl::DlSystem::TensorMap &inputTensorMap)
{
    timer.Start("preprocess_convert_resize");
    img.convertTo(img, CV_32FC1);
    cv::resize(img, img, cv::Size(320, 240), cv::INTER_LINEAR);
    timer.StopAndCount("preprocess_convert_resize");

    //normalize image
    // for (int i = 0; i < img.rows; i++)
	// {
	// 	for (int j = 0; j < img.cols; j++)
	// 	{
	// 		img.ptr<float>(i)[j] = img.ptr<float>(i)[j]/255.0;    //for superpoint, normalized to [0,1]
	// 	}
	// }
    timer.Start("preprocess_normalize");
    img = img/255.0;
    timer.StopAndCount("preprocess_normalize");

    //convert cv mat to float vector
    // timer.Start("preprocess_inputvec");
    // std::vector<float> inputVec = img.reshape(1,1);
    // timer.StopAndCount("preprocess_inputvec");


    //get input tensor name
    timer.Start("preprocess_create tensor");
    const auto& inputTensorNamesRef = snpe->getInputTensorNames();
    if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &inputTensorNames = *inputTensorNamesRef;
    std::string inputName(inputTensorNames.at(0));
    //create Itensor
    const auto &inputShape_opt = snpe->getInputDimensions(inputTensorNames.at(0));
    const auto &inputShape = *inputShape_opt;
    std::unique_ptr<zdl::DlSystem::ITensor> input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    timer.StopAndCount("preprocess_create tensor");
    // if (input->getSize() != inputVec.size()) {
    //     std::cerr << "Size of input does not match network.\n"
    //                 << "Expecting: " << input->getSize() << "\n"
    //                 << "Got: " << inputVec.size() << "\n";
    //     exit(-1);
    // }

    //place data to Itensor
    timer.Start("preprocess_place_to_itensor");
    float *begin_ptr = img.ptr<float>(0);
    float *end_ptr = begin_ptr+img.cols*img.rows;
    std::copy(begin_ptr, end_ptr, input->begin());
    inputTensorMap.add(inputName.c_str(), input.release());
    timer.StopAndCount("preprocess_place_to_itensor");
    //return input tensor map
    std::cout << "Finished processing inputs for current inference \n";
}

void Superpoint::extract_points(std::vector<std::vector<float>> &kps, zdl::DlSystem::ITensor *outputHeatmap, const float threshold, const int &hm_height, const int &hm_width, const int &hm_channel, std::vector<std::vector<float>> &reshape_hm)
{
    zdl::DlSystem::ITensor::iterator outputHeatmap_ptr = outputHeatmap->begin();
    //each keypoint has three elements, hm_width, hm_height, confidence
    int block_size = int(sqrt(hm_channel));
    for (int h=0; h<hm_height; h++)
    {
        for (int w=0; w<hm_width; w++)
        {
            for (int c=0; c<hm_channel; c++)
            {
                int reshape_h = h*block_size+int(c/block_size);
                int reshape_w = w*block_size+(c%block_size);
                int index = c+w*hm_channel+h*hm_channel*hm_width;
                reshape_hm[reshape_h][reshape_w] = outputHeatmap_ptr[index];
                if (outputHeatmap_ptr[index] > threshold)  
                {
                    std::vector<float> temp = {float(reshape_w), float(reshape_h), outputHeatmap_ptr[index]};
                    kps.push_back(temp);
                }
            }
        }
    }
    timer.Start("sort");
    //sort kps by confidence in descending order
    int idx = 2;
    std::sort(kps.begin(), kps.end(), [idx](const std::vector<float>& a, const std::vector<float>& b) {  //this is lambda function, inside [] is the arg you pass
        return a[idx] > b[idx];
    });
    timer.StopAndCount("sort");
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
    int count = 0;
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
            //occupying the keypoint
            nms_map.ptr<int>(temp_h)[temp_w] = 1;
            //record kps and jump out if more than 1000 kps
            if ((temp_h-2*nms_radius)>-1 && (temp_h+2*nms_radius)<nms_map.rows)
            {
                if ((temp_w-2*nms_radius)>-1 && (temp_w+2*nms_radius)<nms_map.cols)
                {
                    std::vector<float> temp = {float(temp_w-nms_radius),float(temp_h-nms_radius)};  //nms_kps also in width, height
                    nms_kps.push_back(temp);
                    count++;
                    if (count > 999) goto enough;
                }
                
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
enough:
    std::cout<<"total amounf of keypoint extracted: "<<nms_kps.size()<<std::endl;
}

void Superpoint::extract_descriptor(zdl::DlSystem::ITensor* outputDesc, const std::vector<cv::Point2f> &final_kps, std::vector<std::vector<float>> &descriptor, const int &desc_height, const int &desc_width, const int &desc_channel, const int &hm_channel)
{
    int block_size = int(sqrt(hm_channel));
    zdl::DlSystem::ITensor::iterator outputDesc_ptr = outputDesc->begin();
    for (size_t i=0; i<final_kps.size(); i++)
    {
        //extract points and perform bilinear interpolation
        std::vector<float> temp_descriptor(256);
        float w = final_kps[i].x / (block_size*2);   //divided by block_size*2 to map it back to output desc dimension, remember you multiply 2 in "get_subpixel_coordinate" to map 320,240 to 640,480
        float h = final_kps[i].y / (block_size*2);   //w corresponds to x, h coorespond to y
        //p1
        int w1 = int(w);    //correspond to x1
        int h1 = int(h);    //correspond to y1
        int w2 = w1+1;      //correspond to x2
        int h2 = h1+1;      //correspond to y2
        if (h >= desc_height-1)    //do this to avoid pointer pointing out of range
        {
            h1 = 28;
            h2 = 29;
        }
        if (w >= desc_width-1)    //do this to avoid pointer pointing out of range
        {
            w1 = 38;
            w2 = 39;
        }
        float w2w = w2-w;
        float h2h = h2-h;
        float ww1 = w-w1;
        float hh1 = h-h1;
        //print for debug
        // if (w1<0 || w2<0 || h1<0 || h2<0)
        // {
        //     std::cout<<"less"<<std::endl;
        // }
        // if (w1>39 || w2>39 || h1>29 || h2>29)
        // {
        //     std::cout<<"more"<<std::endl;
        //     std::cout<<"w is "<<w<<std::endl;
        //     std::cout<<"h is "<<h<<std::endl;
        // }

        for (int c=0; c<256; c++)    //iterate over channel to extract descriptor, 256 is the channel length
        {
            //bilinear interpolation
            temp_descriptor[c] = (w2w*h2h)*outputDesc_ptr[c+w1*desc_channel+h1*desc_width*desc_channel] + \
            (ww1*h2h)*outputDesc_ptr[c+w2*desc_channel+h1*desc_width*desc_channel] + \
            (w2w*hh1)*outputDesc_ptr[c+w1*desc_channel+h2*desc_width*desc_channel] + \
            (ww1*hh1)*outputDesc_ptr[c+w2*desc_channel+h2*desc_width*desc_channel];
        }
        descriptor.push_back(temp_descriptor);
    }
}

void Superpoint::get_subpixel_coordinate(const std::vector<std::vector<float>> &reshape_hm, const std::vector<std::vector<float>> &nms_kps, std::vector<cv::Point2f> &final_kps)
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
                total_weight = total_weight + reshape_hm[height+y][width+x];
            }
        }
        for (int x=-2; x<=2; x++)    //I define x to be the width
        {
            for (int y=-2; y<=2; y++)   //I define y to be the height
            {
                final_width = final_width + reshape_hm[height+y][width+x]/total_weight*(width+x);
                final_height = final_height + reshape_hm[height+y][width+x]/total_weight*(height+y);
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
    zdl::DlSystem::TensorMap input;
    preprocess(snpe, img, input);
    timer.StopAndCount("preprocess");

    //forward
    timer.Start("forward");
    zdl::DlSystem::TensorMap outputTensorMap;
    execStatus = snpe->execute(input, outputTensorMap);
    timer.StopAndCount("forward");
    if (execStatus == true)
    {
        std::cout << "executed successfully."  << std::endl;
    } else {
        std::cerr << "Error while executing the network." << std::endl;
    }

    //extract two output tensors, that is, keypoint heatmap and descriptor
    // Get all output tensor names from the network
    timer.Start("move output to std vector");
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    // get heatmap tensor
    // get heatmap tensor from outputTensorMap by name
    auto outputHeatmap = outputTensorMap.getTensor(tensorNames.at(1));
    // get heatmap tensor dimension
    zdl::DlSystem::TensorShape tensorShape1 = outputHeatmap->getShape();
    const size_t* dims1 = tensorShape1.getDimensions(); //dims[0] batch, dims[1] channel, dims[2] height, dims[3] width
    int hm_height = int(dims1[2]);
    int hm_width = int(dims1[3]);
    int hm_channel = int(dims1[1]);
    // get descriptor tensor
    // get descriptor tensor from outputTensorMap by name
    auto outputDesc = outputTensorMap.getTensor(tensorNames.at(0));
    // get descriptor tensor dimension
    zdl::DlSystem::TensorShape tensorShape2 = outputDesc->getShape();
    const size_t* dims2 = tensorShape2.getDimensions(); //dims[0] batch, dims[1] height, dims[2] width, dims[3] channel
    int desc_height = int(dims2[1]);
    int desc_width = int(dims2[2]);
    int desc_channel = int(dims2[3]);
    timer.StopAndCount("move output to std vector");

    //extract keypoints
    timer.Start("extract_kps");
    float threshold = 0.008;
    std::vector<std::vector<float>> kps;    //kps of dim [N,3], 3 for width, height, confidence
    int rows = int(hm_height*sqrt(hm_channel));
    int cols = int(hm_width*sqrt(hm_channel));
    std::vector<std::vector<float>> reshape_hm(rows,std::vector<float>(cols,0));
    extract_points(kps, outputHeatmap, threshold, hm_height, hm_width, hm_channel, reshape_hm);
    timer.StopAndCount("extract_kps");

    //nms
    timer.Start("nms");
    int nms_radius = 4;
    std::vector<std::vector<float>> nms_kps; //nms_kps of dim [N,2], 2 for width, height; because confidence is not needed anymore
    nms(kps, nms_kps, nms_radius);
    timer.StopAndCount("nms");

    //find subpixel position of kps
    timer.Start("subpixel");
    get_subpixel_coordinate(reshape_hm, nms_kps, final_kps);
    timer.StopAndCount("subpixel");

    //extract descriptor
    timer.Start("extract_des");
    extract_descriptor(outputDesc, final_kps, descriptor, desc_height, desc_width, desc_channel, hm_channel);    //descriptor of dim [N, 256], N for number of nmsed keypoints, 256 for descriptor length
    timer.StopAndCount("extract_des");
    timer.PrintMilliSeconds();
}




