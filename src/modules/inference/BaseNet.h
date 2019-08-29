#ifndef _BASE_NET_H_
#define _BASE_NET_H_

#include "common.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;

class BaseNet
{
    public:
        BaseNet();
        virtual void init(int width, int height, int maxBacthSize);
        virtual ~BaseNet();
        int loadNetFromCaffeModel(const string & protoPath, const string & caffePath);
        int loadNetFromRTModel(const string & rtPath, int device=0);
        int loadNetFromRTModel(const string & rtPath, nvinfer1::IPluginFactory* pluginFactory, int device=0);
        void doInference(float ** input, float ** output, int batchSize);
        void gpuInference(float ** output, int batchSize);

        int getOutputBlobNum(){return output_blob_num_;}
        int getOutputSize(int i){return output_size_[i];}
        float* getOutputData(int i){return output_data_[i];}
        void setInputData(float* data){input_data_[0]=data;}
        int getNbModelLayers();
        string getLayerNameByIndex(int index);
        int getLayerDimByIndex(int index);

        int getInputWidth() {return input_w_;}
        int getInputHeight() {return input_h_;}
        int getInputChannel() {return channel_;}
        int max_batchsize_;

    protected:
        void* buffers_[4];
        cudaStream_t stream_;
        ICudaEngine* engine_;
        IExecutionContext* cuda_context_;
        Logger gLogger;

        // network input and output
        int input_w_, input_h_, channel_;
        int *input_size_, *output_size_;
        int input_blob_num_, output_blob_num_;
        string *input_blob_name_, *output_blob_name_;
        int *inputIndex_, *outputIndex_;
        float **input_data_, **output_data_;

        void initCudaStream();
        void releaseCudaStream();
};

#endif
