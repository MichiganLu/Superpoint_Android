//==============================================================================
//
//  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <algorithm>
#include <sstream>
#include <unordered_map>

#include "SaveOutputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

// Print the results to raw files
// ITensor
bool saveOutput (zdl::DlSystem::TensorMap outputTensorMap,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize)
{
    // Get all output tensor names from the network
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

    // Iterate through the output Tensor map, and print each output layer name to a raw file
    for( auto& name : tensorNames)
    {
        // Split the batched output tensor and save the results
        for(size_t i=0; i<batchSize; i++) {
            std::ostringstream path;
            path << outputDir << "/"
                 << "Result_" << num + i << "/"
                 << name << ".raw";
            auto tensorPtr = outputTensorMap.getTensor(name);    //it is saving result img by img, not batch by batch; if the network has two output, tensorPtr points to one of the output
            size_t batchChunk = tensorPtr->getSize() / batchSize;

            if(!SaveITensorBatched(path.str(), tensorPtr, i, batchChunk))   //i is iterating within batch, batchChunk is datasize for each batch
            {
                return false;
            }
        }
    }
    return true;
}

// Execute the network on an input user buffer map and print results to raw files
bool saveOutput (zdl::DlSystem::UserBufferMap& outputMap,
                 std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize,
                 bool isTfNBuffer,
                 int bitWidth)
{
   // Get all output buffer names from the network
   const zdl::DlSystem::StringList& outputBufferNames = outputMap.getUserBufferNames();

   int elementSize = bitWidth / 8;

   // Iterate through output buffers and print each output to a raw file
   for(auto& name : outputBufferNames)
   {
       for(size_t i=0; i<batchSize; i++) {
           std::ostringstream path;
           path << outputDir << "/"
                << "Result_" << num + i << "/"
                << name << ".raw";
           auto bufferPtr = outputMap.getUserBuffer(name);
           size_t batchChunk = bufferPtr->getSize() / batchSize;
           size_t dataChunk = bufferPtr->getOutputSize() / batchSize;
           if(batchChunk != dataChunk) {
              std::cout << "\tUserBuffer size is " << bufferPtr->getSize() << " bytes, but "
                                                 << bufferPtr->getOutputSize() << " bytes of data was found." << std::endl;
              if( dataChunk > batchChunk )
                 std::cout << "\tAssign a larger buffer using a bigger -z argument" << std::endl;
              batchChunk = std::min(batchChunk,dataChunk);
           }
           if (isTfNBuffer)
           {
              std::vector<uint8_t> output;
              zdl::DlSystem::UserBufferEncodingTfN ubetfN = dynamic_cast<zdl::DlSystem::UserBufferEncodingTfN &>(outputMap.getUserBuffer(name)->getEncoding());
              output.resize(applicationOutputBuffers.at(name).size() * sizeof(float) / elementSize);
              TfNToFloat(reinterpret_cast<float *>(&output[0]),applicationOutputBuffers.at(name).data(),
                         ubetfN.getStepExactly0(),ubetfN.getQuantizedStepSize(),
                        applicationOutputBuffers.at(name).size() / elementSize, bitWidth);
              if(!SaveUserBufferBatched(path.str(), output, i, batchChunk * sizeof(float) / elementSize))
              {
                  return false;
              }
           }
           else
           {
              if(!SaveUserBufferBatched(path.str(), applicationOutputBuffers.at(name), i, batchChunk))
              {
                  return false;
              }
           }
       }
   }
   return true;
}
