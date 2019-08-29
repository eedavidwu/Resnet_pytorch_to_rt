#ifndef _COMMON_H_
#define _COMMON_H_
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace std;

/* TensorRT realated */
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif

#define CHECK_RT(status)								\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

/* YoloV3 Structures */

enum YOLO_CLASS
{
    HEAD = 0,
    BODY = 1
};

typedef struct tag_ImageRatio
{
    int w{0};              // width
    int h{0};              // height
    float r{0.0f};         // ratio for minimum scale
    float shiftw{0.0f};    // for minimum scale and move Net image to centre
    float shifth{0.0f};    // for minimum scale and move Net image to centre
    int im_w{0};           // origin image width
    int im_h{0};           // origin image height
    float ratiow{0.0f};
    float ratioh{0.0f};
} ImageRatio;

/* Attribute Structures */
typedef struct tag_Attribute
{
    vector<float> hat;
    vector<float> cloth;
} Attribute;

#endif //_COMMON_H_
