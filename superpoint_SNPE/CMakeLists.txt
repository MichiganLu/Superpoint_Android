#首先设置cmake version
cmake_minimum_required(VERSION 3.5)

#设置project名
project (superpoint_snpe)

message("compiler is ${CMAKE_CXX_COMPILER}")
message("system_name=${CMAKE_SYSTEM_NAME}, host_name=${CMAKE_HOST_SYSTEM_NAME}")
#debug模式,只有在x64里设置debug模式
if (CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME)
    SET(CMAKE_BUILD_TYPE "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()
message("cmake build type is ${CMAKE_BUILD_TYPE}")

#传参到C++文件控制宏
if (CMAKE_SYSTEM_NAME STREQUAL "Android")  #x64 compilation
    add_definitions( "-DUSE_ANDROID" )
endif()

#创建变量，分别存放外部library，内部library和头文件地址
set(EXTERNAL_LIBS)
set(INTERNAL_LIBS)
set(INCLUDE_DIR)

#设置library输出文件和executable输出文件地址
if (CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME)  #x64 compilation
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/x64/library)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/x64/bin)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")    #android compilation
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/android/library)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/android/bin)
endif()

#添加外部库
#x64编译
if (CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME)     #CMAKE_SYSTEM_NAME是内部参数，x64编译
    message( STATUS "comipling to system = ${CMAKE_HOST_SYSTEM_NAME}")
    #添加系统路径内的库
    #opencv
    FIND_PACKAGE ( OpenCV REQUIRED COMPONENTS core highgui imgcodecs features2d)
    if(OpenCV_FOUND)
        message( STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}"  )
        message( STATUS "OpenCV_LIBS = ${OpenCV_LIBS}"  )
        message( STATUS "OpenCV_DIR = ${OpenCV_DIR}"  )
    else()
        message (FATAL_ERROR "Cannot find OpenCV")
    endif()
    #glog
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(glog REQUIRED libglog)
    message( STATUS "Glog_INCLUDE_DIRS = ${glog_INCLUDE_DIRS}"  )
    message( STATUS "Glog_LIBS = ${glog_LIBRARIES}"  )
    message( STATUS "Glog_DIR = ${GLOG_DIR}"  )

    list(APPEND EXTERNAL_LIBS ${OpenCV_LIBS} ${glog_LIBRARIES})
    list(APPEND INCLUDE_DIR ${OpenCV_INCLUDE_DIRS} ${glog_INCLUDE_DIRS})

    #添加不在系统路径下的外部库
    #SNPE
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/snpe/lib/x64")  
    list(APPEND EXTERNAL_LIBS HtpPrepare omp SNPE)
    list(APPEND INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/snpe/include")



#添加外部库
#android编译
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
    message( STATUS "compiling to system = Android")
    #add android libs
    list(APPEND EXTERNAL_LIBS android log jnigraphics)
    #glog
    link_directories("/home/cvte-vm/Downloads/unify_ndk-master/SrGroup/glog/lib/arm64-v8a")
    list(APPEND EXTERNAL_LIBS glog gflags gflags_nothreads)
    list(APPEND INCLUDE_DIR "/home/cvte-vm/Downloads/unify_ndk-master/SrGroup/glog/include")
    #opencv
    link_directories("/home/cvte-vm/Downloads/unify_ndk-master/SrGroup/opencv/lib/arm64-v8a")
    list(APPEND EXTERNAL_LIBS opencv_world)
    list(APPEND INCLUDE_DIR "/home/cvte-vm/Downloads/unify_ndk-master/SrGroup/opencv/include")
    #SNPE
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/snpe/lib/arm64-v8a")
    list(APPEND EXTERNAL_LIBS c++_shared SNPE)
    list(APPEND INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/snpe/include")


endif()

#添加内部库
file(GLOB SOURCES1 "${CMAKE_CURRENT_SOURCE_DIR}/snpe/utils/*.cpp" )
message("source1 is ${SOURCES1}")
add_library(sp_SNPE_lib SHARED
"${CMAKE_CURRENT_SOURCE_DIR}/superpoint.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/timer.cpp"
${SOURCES1})
list(APPEND INTERNAL_LIBS sp_SNPE_lib)
list(APPEND INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")


#所有INCLUDE_DIR编写完成，现在include_directories
include_directories(${INCLUDE_DIR})

# #链接内部库和外部库
target_link_libraries(${INTERNAL_LIBS} ${EXTERNAL_LIBS})

#print directories for debug
message( STATUS "External lib = ${EXTERNAL_LIBS}"  )
message( STATUS "Internal lib = ${INTERNAL_LIBS}"  )
message( STATUS "Include dir = ${INCLUDE_DIR}"  )

#创建executable并链接内部库和外部库
set(EXEC  
"${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)
add_executable(SNPE_superpoint ${EXEC} )

#link library和可执行文件
target_link_libraries( SNPE_superpoint
    ${EXTERNAL_LIBS}
)
target_link_libraries( SNPE_superpoint
	${INTERNAL_LIBS}
)