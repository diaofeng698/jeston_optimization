#include "tensorrt_detection.h"

TensorRTInference::TensorRTInference(const string model_path, const string input_tensor_name,
                                     const string output_tensor_name, const int preprocessing_method)
{
    context_ = nullptr;
    engine_ = nullptr;
    preprocessing_method_ = preprocessing_method;
    params_.onnxFileName = model_path;
    params_.inputTensorNames.push_back(input_tensor_name);
    params_.outputTensorNames.push_back(output_tensor_name);
    TensorRTBuild();
}

TensorRTInference::~TensorRTInference()
{
    std::cout << "TensorRTInference Destructor Finished! " << std::endl;
}

bool TensorRTInference::TensorRTBuild()
{

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return EXIT_FAILURE;
    }
    std::string model_file = params_.onnxFileName;

    if (access(model_file.c_str(), F_OK) != -1)
    {
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            sample::loadEngine(model_file, params_.dlaCore, sample::gLogError), samplesCommon::InferDeleter());
    }
    else
    {
        std::cout << "Model File Missed, Please check ! " << std::endl;
        return EXIT_FAILURE;
    }

    if (!engine_)
    {
        return EXIT_FAILURE;
    }

    if (params_.inputTensorNames.size() != 1)
    {
        std::cout << "Please Give Input Tensor Name " << std::endl;
        return EXIT_FAILURE;
    }

    int index = engine_->getBindingIndex(params_.inputTensorNames.at(0).c_str());
    if (index == -1)
    {
        return EXIT_FAILURE;
    }
    input_dims_ = engine_->getBindingDimensions(index);

    input_batch_ = input_dims_.d[0];
    input_height_ = input_dims_.d[1];
    input_width_ = input_dims_.d[2];
    input_channel_ = input_dims_.d[3];

    if (params_.outputTensorNames.size() != 1)
    {
        std::cout << "Please Give Output Tensor Name " << std::endl;
        return EXIT_FAILURE;
    }

    index = engine_->getBindingIndex(params_.outputTensorNames.at(0).c_str());
    if (index == -1)
    {
        return EXIT_FAILURE;
    }

    output_dims_ = engine_->getBindingDimensions(index);

    context_ = SampleUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

    return EXIT_SUCCESS;
}

bool TensorRTInference::TensorRTInfer(const cv::Mat &img)
{

    if (!context_)
    {
        std::cout << "TensorRT Initial Failed " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "TensorRT Start Inference " << std::endl;
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    samplesCommon::BufferManager buffers(engine_);

    if (preprocessing_method_ == 0)
    {
        if (ProcessInputTest(buffers))
        {
            return EXIT_FAILURE;
        }

        buffers.copyInputToDevice();
    }
    else if (preprocessing_method_ == 1)
    {
        if (ProcessInputOpenCV(img, buffers))
        {
            return EXIT_FAILURE;
        }
        buffers.copyInputToDevice();
    }
    else if (preprocessing_method_ == 2)
    {
        if (ProcessInputNPPI(img, buffers))
        {
            return EXIT_FAILURE;
        }
    }
    else
    {
        std::cout << "Specify Correct Preprocess Method " << std::endl;
        return EXIT_FAILURE;
    }

    // Memcpy from host input buffers to device input buffers
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return EXIT_FAILURE;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    if (VerifyOutput(buffers))
    {
        return EXIT_FAILURE;
    }
    std::cout << "TensorRT End Inference " << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "TensorRT Whole Infer Time [ms] : " << diff_time << std::endl;

    return EXIT_SUCCESS;
}

bool TensorRTInference::ProcessInputTest(const samplesCommon::BufferManager &buffers)
{
    std::cout << "Choose Input Test " << std::endl;
    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(params_.inputTensorNames[0]));
    for (int b = 0, volImg = input_channel_ * input_height_ * input_width_; b < input_batch_; b++)
    {
        for (int c = 0, volChl = input_height_ * input_width_; c < input_channel_; c++)
        {
            for (int idx = 0; idx < volChl; idx++)
            {
                hostDataBuffer[b * volImg + c * volChl + idx] = 1.0;
            }
        }
    }
    return EXIT_SUCCESS;
}

bool TensorRTInference::ProcessInputOpenCV(const cv::Mat &img, const samplesCommon::BufferManager &buffers)
{
    std::cout << "Choose Input OpenCV " << std::endl;

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(params_.inputTensorNames[0]));

    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat resized_img;
    cv::resize(gray_img, resized_img, cv::Size(input_height_, input_width_));

    for (int b = 0, volImg = input_channel_ * input_height_ * input_width_; b < input_batch_; b++)
    {
        for (int idx = 0, volChl = input_height_ * input_width_; idx < volChl; idx++)
        {

            for (int c = 0; c < input_channel_; ++c)
            {
                hostDataBuffer[b * volImg + idx * input_channel_ + c] = resized_img.data[idx];
            }
        }
    }

    return EXIT_SUCCESS;
}

bool TensorRTInference::ProcessInputNPPI(const cv::Mat &img, const samplesCommon::BufferManager &buffers)
{
    std::cout << "Choose Input NPPI " << std::endl;

    float *deviceDataBuffer = static_cast<float *>(buffers.getDeviceBuffer(params_.inputTensorNames[0]));

    unsigned char *org_device_data = nullptr;
    CHECK(cudaMalloc(&org_device_data, img.rows * img.cols * img.channels()));
    CHECK(cudaMemcpy(org_device_data, img.data, img.rows * img.cols * img.channels(), cudaMemcpyHostToDevice));

    int resized_wid = input_width_;
    int resized_height = input_height_;
    bool bret = true;

    // step1:resize
    unsigned char *resize_device_data = nullptr;
    CHECK(cudaMalloc(&resize_device_data, resized_wid * resized_height * img.channels()));

    bret = resize_image(org_device_data, img.cols, img.rows, img.cols * img.channels(), resize_device_data, resized_wid,
                        resized_height);
    if (!bret)
    {
        sample::gLogError << "NPPI Resize Image Failed " << std::endl;
        return EXIT_FAILURE;
    }

    // step2:gray
    unsigned char *gray_device_data = nullptr;
    CHECK(cudaMalloc(&gray_device_data, resized_wid * resized_height * 1));
    bret = gray_image(resize_device_data, resized_wid, resized_height, 3, gray_device_data);

    if (!bret)
    {
        sample::gLogError << "NPPI Gray Image Failed " << std::endl;
        return EXIT_FAILURE;
    }

    // step3:repeat
    unsigned char *repeat_device_data = nullptr;
    CHECK(cudaMalloc(&repeat_device_data, resized_wid * resized_height * 3));
    bret = repeat_gray_image(gray_device_data, resized_wid, resized_height, 3, repeat_device_data);
    if (!bret)
    {
        sample::gLogError << "NPPI Repeat Image Failed " << std::endl;
        return EXIT_FAILURE;
    }

    NppiSize dstsize = {resized_wid, resized_height};
    NPP_CHECK(nppiConvert_8u32f_C3R((Npp8u *)resize_device_data, resized_wid * 3 * sizeof(uchar),
                                    (Npp32f *)deviceDataBuffer, resized_wid * 3 * sizeof(float), dstsize));

    cudaFree(org_device_data);
    cudaFree(resize_device_data);
    cudaFree(gray_device_data);
    cudaFree(repeat_device_data);

    return EXIT_SUCCESS;
}

bool TensorRTInference::VerifyOutput(const samplesCommon::BufferManager &buffers)
{
    const int outputSize = output_dims_.d[1];
    float *output = static_cast<float *>(buffers.getHostBuffer(params_.outputTensorNames[0]));

    int max_idx = -1;
    float max_prob = 0.0;

    std::cout << "output: ";
    for (int i = 0; i < outputSize; i++)
    {
        std::cout << std::fixed << output[i] << ", ";
        if (max_prob < output[i])
        {
            max_prob = output[i];
            max_idx = i;
        }
    }
    std::cout << std::endl;

    sample::gLogInfo << " Prob " << std::fixed << std::setw(5) << std::setprecision(4) << output[max_idx] << " "
                     << "Class " << max_idx << std::endl;

    inference_result.class_idx = max_idx;
    inference_result.probability = output[max_idx];
    return EXIT_SUCCESS;
}