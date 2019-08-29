#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <sstream>
#include <iterator>
#include <map>
#include <fstream>
#include <cassert>
#include <cstring>
#include "utils.h"
#include "common.h"
#include "batch_stream.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static Logger gLogger;

void caffeToTRTModel(const std::string &deployFile,                   // name for caffe prototxt
                     const std::string &modelFile,                    // name for model
                     const std::vector<std::string> &outputs,         // network outputs
                     unsigned int maxBatchSize,                       // batch size - NB must be at least as large as the batch we want to run with
                     nvcaffeparser1::IPluginFactory* pluginFactory,	  // factory for plugin layers
                     IInt8Calibrator *calibrator,
                     IHostMemory *&gieModelStream,
                     std::map<std::string, DimsCHW> &gInputDimensions,
                     std::map<std::string, DimsCHW> &gOutputDimensions,
                     const std::string &rtfilepath)
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);

    std::cout << "Begin parsing model..." << std::endl;

    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              nvinfer1::DataType::kFLOAT);
    std::cout << "End parsing model..." << std::endl;

    // input blob
    for (int i = 0, n = network->getNbInputs(); i < n; i++) {
        DimsCHW dims = static_cast<DimsCHW &&>(network->getInput(i)->getDimensions());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));

        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x"
                  << dims.w() << std::endl;
    }

    // specify which tensors are outputs
    for (auto &s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // output blob
    for (int i = 0, n = network->getNbOutputs(); i < n; i++) {
        DimsCHW dims = static_cast<DimsCHW &&>(network->getOutput(i)->getDimensions());
        gOutputDimensions.insert(std::make_pair(network->getOutput(i)->getName(), dims));

        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x"
                  << dims.w() << std::endl;

    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);    // we need about 6MB of scratch space for the plugin layer for batch size 5
    builder->setInt8Mode(true);
    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);
    builder->setDebugSync(true);
    builder->setInt8Calibrator(calibrator);

    std::cout << "\nBegin building engine...\n" << std::endl;
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout << "\nEnd building engine...\n" << std::endl;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();

    // save
    std::ofstream outfile(rtfilepath.c_str(), std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "fail to open file to write" << std::endl;
    }
    unsigned char *streamdata = (unsigned char *) gieModelStream->data();
    outfile.write((char *) streamdata, gieModelStream->size());
    outfile.close();

    std::cout << "Write Done." << std::endl;

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void onnxToTRTModel( const std::string& modelFile,        // name of the onnx model
                     unsigned int maxBatchSize,           // batch size - NB must be at least as large as the batch we want to run with
                     IHostMemory *&trtModelStream,        // output buffer for the TensorRT model
                     const std::string& rtfilepath,
                     bool convertToInt8,
                     IInt8Calibrator *calibrator)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder);

    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    auto parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.reportableSeverity)))
    {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    vector<ITensor*> tensors;
    int nb_out = network->getNbOutputs();
    for(int i = 0; i < nb_out; i++)
    {
        ITensor* tensor = network->getOutput(i);
        cout<<tensor->getName()<<endl;
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    if (convertToInt8)
    {
        builder->setInt8Mode(true);
        builder->setAverageFindIterations(8);
        builder->setMinFindIterations(1);
        builder->setDebugSync(true);
        builder->setInt8Calibrator(calibrator);
    }

    std::cout << "\nBegin building engine...\n" << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout << "\nEnd building engine...\n" << std::endl;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();

    // save
    std::ofstream outfile(rtfilepath.c_str(), std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "fail to open file to write" << std::endl;
    }
    unsigned char *streamdata = (unsigned char *) trtModelStream->data();
    outfile.write((char *) streamdata, trtModelStream->size());
    outfile.close();

    std::cout << "Write Done." << std::endl;

    engine->destroy();
    builder->destroy();
}

class Int8EntropyCalibrator : public IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(BatchStream &stream, int firstBatch, string &networkName, string &input_blob_name,
                          bool readCache = false) :
            mStream(stream), mReadCache(readCache), mNetworkName(networkName), mInputBlobName(input_blob_name) {
        DimsCHW dims = mStream.getDims();
        mInputCount = mStream.getCalBatchSize() * dims.c() * dims.h() * dims.w();
        CHECK_RT(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    }

    virtual ~Int8EntropyCalibrator() {
        CHECK_RT(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getCalBatchSize(); }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) override {
        if (!mStream.next())
            return false;

        CHECK_RT(cudaMemcpy(mDeviceInput, mStream.getOneBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], mInputBlobName.c_str()));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void *readCalibrationCache(size_t &length) override {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName().c_str(), std::ios::binary);

        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        std::cout << "readCalibrationCache" << std::endl;
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void *cache, size_t length) override {
        std::cout << "writeCalibrationCache" << std::endl;
        std::ofstream output(calibrationTableName().c_str(), std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }

private:
    std::string calibrationTableName() {
        return std::string("CalibrationTable") + mNetworkName;
    }

    BatchStream mStream;
    bool mReadCache;

    size_t mInputCount;
    void *mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;

    string mNetworkName;
    string &mInputBlobName;
};


int main(int argc, char **argv)
{
    Logger logger;
    IHostMemory *gieModelStream{nullptr};

    int cal_batchsize = 16;
    int nb_cal_batches = 100;

    string image_dir = "../calibration_images/";
    nvinfer1::DimsCHW dims{3, 128, 96};
    vector<float> mean_value{0.485, 0.456, 0.406};
    vector<float> std_value{0.229, 0.224, 0.225};
    BatchStream calibrationStream(cal_batchsize, nb_cal_batches, dims, image_dir, mean_value, std_value);

    int first_cal_batch = 0;
    string network_name = "AttrNet";
    string input_blob_name = "data";
    Int8EntropyCalibrator calibrator(calibrationStream, first_cal_batch, network_name, input_blob_name);

    std::string onnxFile = "../model/resnet34_softmax.onnx";           // path for onnx model
    std::string rtfilepath = "../model/attr_metro_int8.rt";  // path for tensorrt model

    /* int8 model */
    onnxToTRTModel(onnxFile, cal_batchsize, gieModelStream, rtfilepath, true, &calibrator);

    /* fp32 model */
    //onnxToTRTModel(onnxFile, cal_batchsize, gieModelStream, rtfilepath, false, nullptr);

    return 0;
}

