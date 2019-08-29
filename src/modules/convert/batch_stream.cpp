#include <algorithm>
#include "batch_stream.h"

BatchStream::BatchStream(int cal_batchsize, int nb_cal_batch, nvinfer1::DimsCHW &dims, string &image_dir,
                         vector<float> &mean_value, vector<float> &std_value) :
        mCalBatchSize(cal_batchsize), mNbCalBatch(nb_cal_batch), mDims(dims), mImageDir(image_dir)
{
    mImageSize = mDims.c() * mDims.h() * mDims.w();
    mBatch.resize(mCalBatchSize * mImageSize, 0);
    impaths.clear();
    mCalBatchCnt = 0;
    Mean_Value.clear();
    Std_Value.clear();
    Mean_Value.push_back(mean_value[0]);
    Mean_Value.push_back(mean_value[1]);
    Mean_Value.push_back(mean_value[2]);
    Std_Value.push_back(std_value[0]);
    Std_Value.push_back(std_value[1]);
    Std_Value.push_back(std_value[2]);
    randProduce(mImageDir, impaths, mCalBatchSize * mNbCalBatch);
}

bool BatchStream::next() {
    if (mCalBatchCnt == mNbCalBatch)
        return false;
    std::cout << "calibration using " << mCalBatchCnt << " batch" << std::endl;
    if (!update())
        return false;
    mCalBatchCnt++;
    return true;
}

float *BatchStream::getOneBatch() { return &mBatch[0]; }

int BatchStream::getCalBatchSize() const { return mCalBatchSize; }

nvinfer1::DimsCHW BatchStream::getDims() const { return mDims; }

bool BatchStream::update() {
    for (int i = 0; i < mCalBatchSize; ++i) {
        cv::Mat mat = cv::imread(impaths[mCalBatchCnt * mCalBatchSize + i]);
        cv::Mat resized, dst;
        cv::resize(mat, resized, cv::Size(mDims.w(), mDims.h()*3/2), (0, 0), (0, 0), cv::INTER_CUBIC);
        dst = resized(cv::Rect(0, 0, mDims.w(), mDims.h()));
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
        mat2pic(dst, getOneBatch(), i * mDims.c() * mDims.h() * mDims.w(), Mean_Value, Std_Value);
        mat.release();
        dst.release();
        resized.release();
    }
    return true;
}
