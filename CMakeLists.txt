cmake_minimum_required(VERSION 3.8.0)

OPTION(convert_rtmodel "build tensorrt convertion project" ON)
OPTION(attri_inference "build tensorrt inference project" ON)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
SET(CMAKE_INSTALL_PREFIX install)
SET(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
SET(THIRD_PARTY_PATH ${PROJECT_SOURCE_DIR}/thirdparty)

FILE(GLOB_RECURSE common_file ${PROJECT_SOURCE_DIR}/src/modules/common/*.*)
FILE(GLOB_RECURSE convert_module_file ${PROJECT_SOURCE_DIR}/src/modules/convert/*.*)
FILE(GLOB_RECURSE infer_module_file ${PROJECT_SOURCE_DIR}/src/modules/inference/*.*)

INCLUDE_DIRECTORIES(/usr/local/cuda-9.0/include)
INCLUDE_DIRECTORIES(/home/xieyi/libs/cudnn_7_5/7.5.0/include)
INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/opencv/include)
INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/tensorrt-5.1.2/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/modules/common)


IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_DEFINITIONS(-O3)
ELSE()
    ADD_DEFINITIONS(-O0)
ENDIF()


if (convert_rtmodel)
    INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/modules/convert)
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src/servers/convert)
endif(convert_rtmodel)


if (attri_inference)
    INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/modules/inference)
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src/servers/inference)
endif(attri_inference)
