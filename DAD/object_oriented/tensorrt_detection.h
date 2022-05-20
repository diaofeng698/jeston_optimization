#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleEngines.h"
#include "util/npp_common.h"
#include "util/process.h"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <sys/time.h>
#include <typeinfo>
#include <unistd.h>

struct TensorRTInferenceResult
{
    int class_idx;
    float probability;
};

class TensorRTInference
{
    typedef bool (*PreprocessingCallbackFun)(const cv::Mat &, std::shared_ptr<TensorRTInference>);
    typedef bool (*PostprocessingCallbackFun)(std::shared_ptr<TensorRTInference>);

  public:
    TensorRTInferenceResult inference_result;
    TensorRTInference(const string model_path, const string input_tensor_name, const string output_tensor_name);
    ~TensorRTInference();
    bool TensorRTBuild();
    bool TensorRTInfer(PreprocessingCallbackFun preprocessing_call_fun,
                       PostprocessingCallbackFun postprocessing_call_fun, const cv::Mat &img,
                       std::shared_ptr<TensorRTInference> temp);
    static bool ProcessInputOpenCV(const cv::Mat &img, std::shared_ptr<TensorRTInference> temp);
    static bool ProcessInputNPPI(const cv::Mat &img, std::shared_ptr<TensorRTInference> temp);
    static bool VerifyOutput(std::shared_ptr<TensorRTInference> temp);

  private:
    samplesCommon::OnnxSampleParams params_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    int input_channel_{0};
    int input_height_{0};
    int input_width_{0};
    int input_batch_{0};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::shared_ptr<samplesCommon::BufferManager> buffers_;
};