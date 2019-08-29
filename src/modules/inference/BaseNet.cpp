#include "BaseNet.h"

// constructor
BaseNet::BaseNet()
{
    stream_ = NULL;
    cuda_context_ = NULL;

    max_batchsize_   = 0;
    input_w_         = 0;
    input_h_         = 0;
    channel_         = 0;
    input_blob_num_  = 0;
    output_blob_num_ = 0;
    input_size_      = NULL;
    output_size_     = NULL;
    input_blob_name_ = NULL;
    output_blob_name_= NULL;
    inputIndex_      = NULL;
    outputIndex_     = NULL;
    input_data_      = NULL;
    output_data_     = NULL;
}

// Specific Net should set its own initial parameters
void BaseNet::init(int width, int height, int maxBacthSize)
{
    input_w_ = width;
    input_h_ = height;
    channel_ = 3;
    max_batchsize_ = maxBacthSize;
}

BaseNet::~BaseNet()
{
    if(cuda_context_)
    {
        cuda_context_->destroy();
        cuda_context_ = NULL;
    }

    if(engine_)
    {
        engine_->destroy();
        engine_ = NULL;
    }

    delete [] input_size_;
    delete [] output_size_;
    delete [] input_blob_name_;
    delete [] output_blob_name_;

    for (int i = 0; i < input_blob_num_; ++i) {
        cudaFreeHost(input_data_[i]);
    }

    for (int i = 0; i < output_blob_num_; ++i) {
        cudaFreeHost(output_data_[i]);
    }

    releaseCudaStream();

    delete [] inputIndex_;
    delete [] outputIndex_;
}

int BaseNet::loadNetFromCaffeModel(const string & protoPath, const string & caffePath)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    bool useFp16 = builder->platformHasFastFp16();


    nvinfer1::DataType modelDataType = useFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
    
    const IBlobNameToTensor *blobNameToTensor =
            parser->parse(protoPath.c_str(),               // caffe deploy file
                          caffePath.c_str(),               // caffe model file
                          *network,                        // network definition that the parser will populate
                          modelDataType);

    if (!blobNameToTensor)
    {
        cout<<"Failed to parse caffe network."<<endl;
        return 0;
    }

    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate
    for (int i=0; i<output_blob_num_; i++)
    {
        network->markOutput(*blobNameToTensor->find(output_blob_name_[i].c_str()));
    }

    // Build the engine
    builder->setMaxBatchSize(max_batchsize_);
    builder->setMaxWorkspaceSize(1 << 25);

    // set up the network for paired-fp16 format if available
    if(useFp16)
        builder->setHalf2Mode(true);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    IExecutionContext* context = engine->createExecutionContext();
    cuda_context_ = context;
    engine_ = engine;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

int BaseNet::loadNetFromRTModel(const string & rtPath, int device)
{
    // deserialized the engine from disk
    ifstream cache(rtPath.c_str(), std::ios::in | std::ios::binary);
    if (!cache)
    {
        cout<<"fail to open tensorRT model file!"<<endl;
        return 0;
    }
    streampos begin, end;
    begin = cache.tellg();
    cache.seekg(0, std::ios::end);
    end = cache.tellg();
    const int modelSize = end-begin;
    cache.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    cache.read((char*)modelMem, modelSize);
    cache.close();

    // set GPU ID for tensorRT reference
    cudaError_t st = cudaSetDevice(device);
    if (st != cudaSuccess)
        throw std::invalid_argument("could not set CUDA device");
    
    IRuntime* infer = createInferRuntime(gLogger);
    ICudaEngine* engine = infer->deserializeCudaEngine((const void*)modelMem, modelSize, nullptr);
    free(modelMem);
    IExecutionContext* context = engine->createExecutionContext();
    cuda_context_ = context;
    engine_ = engine;

    infer->destroy();
    return 1;
}

int BaseNet::getNbModelLayers()
{
    return cuda_context_->getEngine().getNbLayers();
}

string BaseNet::getLayerNameByIndex(int index)
{
    string layer_name;
    layer_name = cuda_context_->getEngine().getBindingName(index);
    return layer_name;
}

int BaseNet::getLayerDimByIndex(int index)
{
    int dims = cuda_context_->getEngine().getBindingDimensions(index).nbDims;
    if (dims == 3)
    {
        cout<<"dim0:"<<cuda_context_->getEngine().getBindingDimensions(index).d[0]<<endl;
        cout<<"dim1:"<<cuda_context_->getEngine().getBindingDimensions(index).d[1]<<endl;
        cout<<"dim2:"<<cuda_context_->getEngine().getBindingDimensions(index).d[2]<<endl;
        cout<<"datatype:"<<int(cuda_context_->getEngine().getBindingDataType(index))<<endl;
    }
}

int BaseNet::loadNetFromRTModel(const string & rtPath, nvinfer1::IPluginFactory* pluginFactory, int device)
{
    // deserialized the engine from disk
    ifstream cache(rtPath.c_str(), std::ios::in | std::ios::binary);
    if (!cache)
    {
        cout<<"fail to open tensorRT model file!"<<endl;
        return 0;
    }
    streampos begin, end;
    begin = cache.tellg();
    cache.seekg(0, std::ios::end);
    end = cache.tellg();
    const int modelSize = end-begin;
    cache.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    cache.read((char*)modelMem, modelSize);
    cache.close();

    // set GPU ID for tensorRT reference
    cudaError_t st = cudaSetDevice(device);
    if (st != cudaSuccess)
        throw std::invalid_argument("could not set CUDA device");

    IRuntime* infer = createInferRuntime(gLogger);
    ICudaEngine* engine = infer->deserializeCudaEngine((const void*)modelMem, modelSize, pluginFactory);
    free(modelMem);
    IExecutionContext* context = engine->createExecutionContext();
    cuda_context_ = context;
    engine_ = engine;

    infer->destroy();
    return 1;
}

void BaseNet::doInference(float ** input, float ** output, int batchSize)
{
    for(int i=0;i<input_blob_num_;i++)
    {
        CHECK_RT(cudaMemcpyAsync(buffers_[inputIndex_[i]],input[i],input_size_[i]*batchSize*sizeof(float),cudaMemcpyHostToDevice,stream_));
    }

    cuda_context_->enqueue(batchSize, buffers_, stream_, nullptr);

    for(int i=0;i<output_blob_num_;i++)
    {
        CHECK_RT(cudaMemcpyAsync(output[i],buffers_[outputIndex_[i]],output_size_[i]*batchSize*sizeof(float),cudaMemcpyDeviceToHost,stream_));
    }
            
    cudaStreamSynchronize(stream_);
}

void BaseNet::gpuInference(float ** output, int batchSize)
{
    cuda_context_->enqueue(batchSize, buffers_, stream_, nullptr);

    for(int i=0;i<output_blob_num_;i++)
    {
        CHECK_RT(cudaMemcpyAsync(output[i],buffers_[outputIndex_[i]],output_size_[i]*batchSize*sizeof(float),cudaMemcpyDeviceToHost,stream_));
    }

    cudaStreamSynchronize(stream_);
}

void BaseNet::initCudaStream()
{
    const ICudaEngine& engine = cuda_context_->getEngine();
    for(int i=0;i<input_blob_num_;i++)
    {
        inputIndex_[i] = engine.getBindingIndex(input_blob_name_[i].c_str());
        CHECK_RT(cudaMalloc(&buffers_[inputIndex_[i]], max_batchsize_*input_size_[i]*sizeof(float)));
    }
    for(int i=0;i<output_blob_num_;i++)
    {
        outputIndex_[i] = engine.getBindingIndex(output_blob_name_[i].c_str());
        CHECK_RT(cudaMalloc(&buffers_[outputIndex_[i]],max_batchsize_*output_size_[i]*sizeof(float)));
    }
    CHECK_RT(cudaStreamCreate(&stream_));
}

void BaseNet::releaseCudaStream()
{
    cudaStreamDestroy(stream_);
    for(int i=0;i<input_blob_num_;i++)
        CHECK_RT(cudaFree(buffers_[inputIndex_[i]]));
    for(int i=0;i<output_blob_num_;i++)
        CHECK_RT(cudaFree(buffers_[outputIndex_[i]]));
}
