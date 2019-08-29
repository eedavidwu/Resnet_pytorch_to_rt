
#ifndef TENSORRT_BATCH_STREAM_H
#define TENSORRT_BATCH_STREAM_H

#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "utils.h"
#include "common.h"

class BatchStream {
public:
    BatchStream(int batchSize, int maxBatches, nvinfer1::DimsCHW &dims, string &image_dir, vector<float> &mean_value, vector<float> &std_value);

    bool next();

    float *getOneBatch();

    int getCalBatchSize() const;

    nvinfer1::DimsCHW getDims() const;

private:

    bool update();

    int mCalBatchSize{0};
    int mNbCalBatch{0};
    int mCalBatchCnt{0};

    int mImageSize{0};

    string mImageDir;
    std::vector<std::string> impaths;

    nvinfer1::DimsCHW mDims;
    std::vector<float> mBatch;

    vector<float> Mean_Value;
    vector<float> Std_Value;
};

#endif //TENSORRT_BATCH_STREAM_H
