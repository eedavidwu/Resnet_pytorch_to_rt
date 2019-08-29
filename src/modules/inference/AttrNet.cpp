
#include "AttrNet.h"

// constructor
AttrNet::AttrNet() : BaseNet() {}

// set default parameters for ResNet34
void AttrNet::init(int width, int height, int maxBatchSize)
{
    input_w_ = width;
    input_h_ = height;
    channel_ = 3;
    max_batchsize_ = maxBatchSize;

    input_blob_num_ = 1;
    output_blob_num_ = 2;

    input_blob_name_ = new string[input_blob_num_];
    output_blob_name_ = new string[output_blob_num_];
    input_size_ = new int[input_blob_num_];
    output_size_ = new int[output_blob_num_];

    input_blob_name_[0] = "data";
    output_blob_name_[0] = "prob_hat";
    output_blob_name_[1] = "prob_cloth";

    input_size_[0] = input_w_ * input_h_ * channel_;
    output_size_[0] = 2;
    output_size_[1] = 2;

    inputIndex_ = new int[input_blob_num_];
    input_data_ = new float*[input_blob_num_];
    for (int i=0; i<input_blob_num_; i++)
    {
        CHECK_RT(cudaHostAlloc((void **)&input_data_[i], max_batchsize_ * input_size_[i] * sizeof(float), cudaHostAllocDefault));
    }

    outputIndex_ = new int[output_blob_num_];
    output_data_ = new float*[output_blob_num_];
    for (int i=0; i<output_blob_num_; i++)
    {
        CHECK_RT(cudaHostAlloc((void **)&output_data_[i], max_batchsize_ * output_size_[i] * sizeof(float), cudaHostAllocDefault));
    }

    clean();
    initCudaStream();
}

void AttrNet::clean()
{
    for(int i=0;i<input_blob_num_;i++)
        memset(input_data_[i], 0, max_batchsize_*input_size_[i]*sizeof(float));
    for(int i=0;i<output_blob_num_;i++)
        memset(output_data_[i], 0, max_batchsize_*output_size_[i]*sizeof(float));
}

AttrNet::~AttrNet()
{

}

void AttrNet::convertData(cv::Mat &img, int i)
{
    image2RGBMatrixBatch(img, input_data_[0], i * input_size_[0]);
}

Attribute AttrNet::run(cv::Mat &img)
{
    cv::Mat copy = img.clone();

    if (img.cols != input_w_ || img.rows != input_h_)
        resize(copy, copy, cv::Size(input_w_, input_h_));

    int batch_size = 1;
    convertData(copy, 0);
    doInference(input_data_, output_data_, batch_size);

    Attribute hat_cloth_prob;
    for (int i=0; i<2; i++)
    {
        hat_cloth_prob.hat.push_back(getOutputData(0)[i]);
        hat_cloth_prob.cloth.push_back(getOutputData(1)[i]);
    }
    return hat_cloth_prob;
}

void AttrNet::image2RGBMatrixBatch(const cv::Mat &image, float *input, int startIndex)
{
    if ((image.data == NULL) || (image.type() != CV_8UC3)){
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    if (input == NULL){
        return;
    }

    float *p = input + startIndex;
    for (int rowI = 0; rowI < image.rows; rowI++)
    {
        for (int colK = 0; colK < image.cols; colK++)
        {
            *p = (float(image.at<cv::Vec3b>(rowI, colK)[2]) / 255.0 - mean_value[0]) / std_value[0];
            *(p + image.rows*image.cols) = (float(image.at<cv::Vec3b>(rowI, colK)[1]) / 255.0 - mean_value[1]) / std_value[1];
            *(p + 2*image.rows*image.cols) = (float(image.at<cv::Vec3b>(rowI, colK)[0]) / 255.0 - mean_value[2]) / std_value[2];
            p++;
//            cout<<*p<<endl;
        }
    }
}
