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
  public:
    TensorRTInferenceResult inference_result;
    TensorRTInference(const string model_path, const string input_tensor_name, const string output_tensor_name,
                      const int preprocessing_method);
    ~TensorRTInference();
    bool TensorRTBuild();
    bool TensorRTInfer(const cv::Mat &img);

  private:
    samplesCommon::OnnxSampleParams params_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    int class_number_{0};
    int input_channel_{0};
    int input_height_{0};
    int input_width_{0};
    int input_batch_{0};
    int preprocessing_method_{0};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::shared_ptr<samplesCommon::BufferManager> buffers_;
    bool ProcessInputTest();
    bool ProcessInputOpenCV(const cv::Mat &img);
    bool ProcessInputNPPI(const cv::Mat &img);
    bool VerifyOutput();
};