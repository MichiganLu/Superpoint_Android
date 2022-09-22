//==============================================================================
//
//  Copyright (c) 2017-2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

#include "LoadInputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"


// Load a batched single input tensor for a network which requires a single input
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE>& snpe , std::vector<std::string>& fileLines)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert (strList.size() == 1);

    // If the network has a single input, each line represents the input file to be loaded for that input
    std::vector<float> inputVec;
    for(size_t i=0; i<fileLines.size(); i++) {
        std::string filePath(fileLines[i]);
        std::cout << "Processing DNN Input: " << filePath << "\n";
        std::vector<float> loadedFile = loadFloatDataFile(filePath);
        inputVec.insert(inputVec.end(), loadedFile.begin(), loadedFile.end());
    }

    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;

    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains the expected number of elements.
       With the input dimensions computed create a tensor to convey the input into the network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    //Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    zdl::DlSystem::TensorShape tensorShape= snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    if(fileLines.size()<batchSize) {
           for(size_t j=0; j<batchSize-fileLines.size(); j++) {       //输入数量不够batch的数量就补padding
                std::vector<float> padding(input->getSize()/batchSize,0);  //input->getSize()是整个batch的长度， input->getSize()/batchSize是单个input的长度， 后面数字0是把padding设为0
                inputVec.insert(inputVec.end(),padding.begin(),padding.end());
           }
    }

    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }

    /* Copy the loaded input file contents into the networks input tensor. SNPE's ITensor supports C++ STL functions like std::copy() */
    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}

// Load multiple input tensors for a network which require multiple inputs
std::tuple<zdl::DlSystem::TensorMap, bool> loadMultipleInput (std::unique_ptr<zdl::SNPE::SNPE>& snpe , std::vector<std::string>& fileLines)
{
    zdl::DlSystem::TensorMap dummy; // dummy map for returning on failure
    const auto& inputTensorNamesRef = snpe->getInputTensorNames();
    if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &inputTensorNames = *inputTensorNamesRef;
    // Make sure the network requires multiple inputs
    assert (inputTensorNames.size() > 1);

    if (inputTensorNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputs(inputTensorNames.size());    //inputs is a vector of itensor pointer
    zdl::DlSystem::TensorMap  inputTensorMap;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);       //用fileLines中的一个元素来初始化fileLine
        // Treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');        //把fileLine一行的几个输入分开放到filePaths里，fileLine的输入用空格隔开

        for (size_t j = 0; j<inputTensorNames.size(); j++) {      //输入一个一个处理

            // print out which file is being processed
            std::string filePath(filePaths[j]);   //把filesPath的文件一个一个拿出来，用来初始化filePath
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            std::string inputName(inputTensorNames.at(j));
            std::vector<float> inputVec = loadFloatDataFile(filePath);

            const auto &inputShape_opt = snpe->getInputDimensions(inputTensorNames.at(j));
            const auto &inputShape = *inputShape_opt;
            inputs[j] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);    //把创建的tensor放到vector of Itensor pointer中， input的第j项对应第j个输入

            if (inputs[j]->getSize() != inputVec.size()) {
                std::cerr << "Size of input does not match network.\n"
                          << "Expecting: " << inputs[j]->getSize() << "\n"
                          << "Got: " << inputVec.size() << "\n";
                return std::make_tuple(dummy, false);
            }

            std::copy(inputVec.begin(), inputVec.end(), inputs[j]->begin());
            inputTensorMap.add(inputName.c_str(), inputs[j].release());
        }
    }
    std::cout << "Finished processing inputs for current inference \n";
    return std::make_tuple(inputTensorMap, true);
}

bool loadInputUserBufferTfN(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines,
                         zdl::DlSystem::UserBufferMap& inputMap,
                         int bitWidth)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        for (size_t j = 0; j < inputNames.size(); j++) {
            const char *name = inputNames.at(j);
            std::string filePath(filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            unsigned char stepEquivalentTo0;
            float quantizedStepSize;
            if(!loadByteDataFileBatchedTfN(filePath, applicationBuffers.at(name), i, stepEquivalentTo0, quantizedStepSize, bitWidth))
            {
                return false;
            }
            auto userBufferEncoding = dynamic_cast<zdl::DlSystem::UserBufferEncodingTfN *>(&inputMap.getUserBuffer(name)->getEncoding());
            userBufferEncoding->setStepExactly0(stepEquivalentTo0);
            userBufferEncoding->setQuantizedStepSize(quantizedStepSize);

        }
    }
    return true;
}

// Load multiple batched input user buffers
bool loadInputUserBufferFloat(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        for (size_t j = 0; j < inputNames.size(); j++) {
            const char *name = inputNames.at(j);
            std::string filePath(filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            if(!loadByteDataFileBatched(filePath, applicationBuffers.at(name), i))
            {
                return false;
            }
        }
    }
    return true;
}

void loadInputUserBuffer(std::unordered_map<std::string, GLuint>& applicationBuffers,
                               std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                               const GLuint inputglbuffer)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    for (size_t i = 0; i < inputNames.size(); i++) {
        const char* name = inputNames.at(i);
        applicationBuffers.at(name) = inputglbuffer;
    };
}
