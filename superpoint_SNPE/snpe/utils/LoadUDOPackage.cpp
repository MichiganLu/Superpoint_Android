//==============================================================================
//
//  Copyright (c) 2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <string>

#include "LoadUDOPackage.hpp"

#include "SNPE/SNPEFactory.hpp"
#include "Util.hpp"

bool loadUDOPackage(const std::string& UdoPackagePath)
{
    std::vector<std::string> udoPkgPathsList;
    split(udoPkgPathsList, UdoPackagePath, ',');
    for (const auto &u : udoPkgPathsList)
    {
       if (false == zdl::SNPE::SNPEFactory::addOpPackage(u))
       {
          std::cerr << "Error while loading UDO package: "<< u << std::endl;
          return false;
       }
    }
    return true;
}