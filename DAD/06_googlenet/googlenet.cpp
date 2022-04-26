#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleEngines.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include "util/npp_common.h"
#include "util/process.h"

#include <cstdlib>
#include <fstream>
#include <io.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>

const std::string gSampleName = "TensorRT.googlenet";
double total_fDiffTime;
double total_pre_fDiffTime;
double total_infer_fDiffTime;
double total_togpu_fDiffTime;
double total_tocpu_fDiffTime;
class GoogleNet
{
    template <typename T> using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

  public:
    GoogleNet(const samplesCommon::OnnxSampleParams &params) : mParams(params), mEngine(nullptr)
    {
    }

    bool build();

    bool infer(std::string file_name);

  private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    int inputChannel{0};
    int inputH{0};
    int inputW{0};
    int inputB{0};

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    bool processInputTest(const samplesCommon::BufferManager &buffers);
    bool processInputOpenCV(const samplesCommon::BufferManager &buffers, std::string file_name);
    bool processInputNPPI(const samplesCommon::BufferManager &buffers, std::string file_name);

    bool verifyOutput(const samplesCommon::BufferManager &buffers);
};
/**
 * @brief
 * @note
 * @retval
 */
bool GoogleNet::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    std::string onnx_file = locateFile(mParams.onnxFileName, mParams.dataDirs);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(sample::loadEngine(onnx_file, mParams.dlaCore, sample::gLogError),
                                                     samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    int index = mEngine->getBindingIndex(mParams.inputTensorNames.at(0).c_str());
    if (index == -1)
        return false;

    mInputDims = mEngine->getBindingDimensions(index);
    assert(mInputDims.nbDims == 4);
    std::cout << "mInputDims: ";
    for (size_t i = 0; i < mInputDims.nbDims; i++)
    {
        std::cout << mInputDims.d[i] << ", ";
    }
    std::cout << std::endl;

    index = mEngine->getBindingIndex(mParams.outputTensorNames.at(0).c_str());
    if (index == -1)
        return false;
    mOutputDims = mEngine->getBindingDimensions(index);
    assert(mOutputDims.nbDims == 2);
    std::cout << "mOutputDims: ";
    for (size_t i = 0; i < mOutputDims.nbDims; i++)
    {
        std::cout << mOutputDims.d[i] << ", ";
    }
    std::cout << std::endl;

    inputB = mInputDims.d[0];
    inputH = mInputDims.d[1];
    inputW = mInputDims.d[2];
    inputChannel = mInputDims.d[3];
    return true;
}

bool GoogleNet::infer(std::string file_name)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    struct timeval engine_tv1, engine_tv2;
    /*Start timer*/
    gettimeofday(&engine_tv1, NULL);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    /*End timer*/
    gettimeofday(&engine_tv2, NULL);

    double engine_fDiffTime = ((double)(engine_tv2.tv_usec - engine_tv1.tv_usec) / 1000.0) +
                              ((double)(engine_tv2.tv_sec - engine_tv1.tv_sec) * 1000.0);
    std::cout << "===============> createExecutionContext: " << engine_fDiffTime << std::endl;

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    bool bret = true;

    if (mParams.preprocess_device == -1)
    {
        bret = processInputTest(buffers);
        buffers.copyInputToDevice();
    }
    else if (mParams.preprocess_device == 0)
    {
        struct timeval pre_tv1, pre_tv2;
        /*Start timer*/
        gettimeofday(&pre_tv1, NULL);

        bret = processInputOpenCV(buffers, file_name);
        /*End timer*/
        gettimeofday(&pre_tv2, NULL);

        double pre_fDiffTime = ((double)(pre_tv2.tv_usec - pre_tv1.tv_usec) / 1000.0) +
                               ((double)(pre_tv2.tv_sec - pre_tv1.tv_sec) * 1000.0);
        std::cout << "===============> Pre-processing time of DaD: " << pre_fDiffTime << std::endl;
        total_pre_fDiffTime += pre_fDiffTime;

        struct timeval togpu_tv1, togpu_tv2;
        /*Start timer*/
        gettimeofday(&togpu_tv1, NULL);
        buffers.copyInputToDevice();
        /*End timer*/
        gettimeofday(&togpu_tv2, NULL);

        double togpu_fDiffTime = ((double)(togpu_tv2.tv_usec - togpu_tv1.tv_usec) / 1000.0) +
                                 ((double)(togpu_tv2.tv_sec - togpu_tv1.tv_sec) * 1000.0);
        std::cout << "===============> copyInputToDevice: " << togpu_fDiffTime << std::endl;
    }
    else
    {
        struct timeval pre_npp_tv1, pre_npp_tv2;
        /*Start timer*/
        gettimeofday(&pre_npp_tv1, NULL);

        bret = processInputNPPI(buffers, file_name);

        /*End timer*/
        gettimeofday(&pre_npp_tv2, NULL);

        double pre_npp_fDiffTime = ((double)(pre_npp_tv2.tv_usec - pre_npp_tv1.tv_usec) / 1000.0) +
                                   ((double)(pre_npp_tv2.tv_sec - pre_npp_tv1.tv_sec) * 1000.0);
        std::cout << "===============> Pre-processing time of DaD: " << pre_npp_fDiffTime << std::endl;
        total_pre_fDiffTime += pre_npp_fDiffTime;
    }
    if (!bret)
        return false;

    struct timeval infer_tv1, infer_tv2;
    /*Start timer*/
    gettimeofday(&infer_tv1, NULL);

    // Memcpy from host input buffers to device input buffers
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    /*End timer*/
    gettimeofday(&infer_tv2, NULL);

    double infer_fDiffTime = ((double)(infer_tv2.tv_usec - infer_tv1.tv_usec) / 1000.0) +
                             ((double)(infer_tv2.tv_sec - infer_tv1.tv_sec) * 1000.0);
    std::cout << "===============> Infer in GPU: " << infer_fDiffTime << std::endl;
    total_infer_fDiffTime += infer_fDiffTime;

    struct timeval tocpu_tv1, tocpu_tv2;
    /*Start timer*/
    gettimeofday(&tocpu_tv1, NULL);
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    /*End timer*/
    gettimeofday(&tocpu_tv2, NULL);

    double tocpu_fDiffTime = ((double)(tocpu_tv2.tv_usec - tocpu_tv1.tv_usec) / 1000.0) +
                             ((double)(tocpu_tv2.tv_sec - tocpu_tv1.tv_sec) * 1000.0);
    std::cout << "===============> copyOutputToHost: " << tocpu_fDiffTime << std::endl;
    total_tocpu_fDiffTime += tocpu_fDiffTime;

    struct timeval out_tv1, out_tv2;
    /*Start timer*/
    gettimeofday(&out_tv1, NULL);
    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    /*End timer*/
    gettimeofday(&out_tv2, NULL);

    double out_fDiffTime =
        ((double)(out_tv2.tv_usec - out_tv1.tv_usec) / 1000.0) + ((double)(out_tv2.tv_sec - out_tv1.tv_sec) * 1000.0);
    std::cout << "===============> verifyOutput: " << out_fDiffTime << std::endl;

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool GoogleNet::processInputTest(const samplesCommon::BufferManager &buffers)
{
    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    std::cout << "inputH " << inputH << " inputW " << inputW << " inputChannel " << inputChannel << std::endl;

    for (int i = 0; i < inputChannel * inputH * inputW; i++)
    {
        hostDataBuffer[i] = 255.0;
    }

    return true;
}

bool GoogleNet::processInputOpenCV(const samplesCommon::BufferManager &buffers, std::string file_name)
{
    std::cout << "*************** OpenCV ***************" << std::endl;

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    std::string img_name = locateFile(file_name, mParams.dataDirs);
    cv::Mat img = cv::imread(img_name);
    std::cout << "raw inputH " << img.rows << " inputW " << img.cols << " inputChannel " << img.channels() << std::endl;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    std::cout << "resized inputH " << resized_img.rows << " inputW " << resized_img.cols << " inputChannel "
              << resized_img.channels() << std::endl;

    cv::Mat gray_img;
    cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);
    std::cout << "gray inputH " << gray_img.rows << " inputW " << gray_img.cols << " inputChannel "
              << gray_img.channels() << std::endl;
    //  //std::cout << "img:" << img.at<cv::Vec3b>(1, 1) << std::endl;
    //  // resize to 256
    //  double fx = 256.0 / std::min(img.rows, img.cols);
    //  cv::Mat resized_img;
    //  cv::resize(img, resized_img, cv::Size(), fx, fx);
    //  std::cout << "resized_img:" << resized_img.at<cv::Vec3b>(1, 1) << std::endl;
    //  // cv::imwrite("resized_img.jpg", resized_img);
    //  // center crop
    //  int crop_x = std::max((resized_img.cols - inputW) / 2, 0);
    //  int crop_y = std::max((resized_img.rows - inputH) / 2, 0);
    //  std::cout << "crop_x:" << crop_x << ", crop_y:" << crop_y << std::endl;
    //  cv::Mat croped_mat = resized_img(cv::Rect(crop_x, crop_y, crop_x + inputW, crop_y + inputH));
    //  // cv::imwrite("croped_mat.jpg", croped_mat);
    //  std::cout << croped_mat.at<cv::Vec3b>(0, 0) << std::endl;
    //
    //  // normalized
    //  std::vector<float> mean_value{0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0};
    //  std::vector<float> std_value{0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0};
    // float *ptr_chn0 = hostDataBuffer;
    //  float *ptr_chn1 = ptr_chn0 + inputH * inputW;
    //  float *ptr_chn2 = ptr_chn1 + inputH * inputW;
    // for (size_t i = 0; i < inputH; i++) {
    //    for (size_t j = 0; j < inputW; j++) {
    //  *(ptr_chn0+inputChannel*i*j) = gray_img.at<unchar>(i,j);
    //  *(ptr_chn0+inputChannel*i*j+1) = gray_img.at<unchar>(i,j);
    //  *(ptr_chn0+inputChannel*i*j+2) = gray_img.at<unchar>(i,j);
    for (int b = 0, volImg = inputChannel * inputH * inputW; b < inputB; b++)
    {
        for (int idx = 0, volChl = inputH * inputW; idx < volChl; idx++)
        {

            for (int c = 0; c < inputChannel; ++c)
            {
                hostDataBuffer[b * volImg + idx * inputChannel + c] = gray_img.data[idx];
            }
        }
    }

    // if (i == 0) {
    //   std::cout << *(ptr_chn0 + i * inputW + j) << ", ";
    // }

    // if (i == 0) {
    //   std::cout << std::endl;
    // }

    // std::cout << *ptr_chn0 << ", " << *ptr_chn1 << ", " << *ptr_chn2 << ", " << std::endl;
    return true;
}

bool GoogleNet::processInputNPPI(const samplesCommon::BufferManager &buffers, std::string file_name)
{
    float *deviceDataBuffer = static_cast<float *>(buffers.getDeviceBuffer(mParams.inputTensorNames[0]));
    std::cout << "*************** NPP ***************" << std::endl;

    // float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    //  Npp32f *rPlane = (Npp32f *)(deviceDataBuffer);
    //  Npp32f *gPlane = (Npp32f *)(deviceDataBuffer + inputW * inputH);
    //  Npp32f *bPlane = (Npp32f *)(deviceDataBuffer + inputW * inputH * 2);
    //  Npp32f *rgbPlanes[3] = {bPlane, gPlane, rPlane};

    struct timeval npp_read_tv1, npp_read_tv2;
    /*Start timer*/
    gettimeofday(&npp_read_tv1, NULL);

    std::string img_name = locateFile(file_name, mParams.dataDirs);
    cv::Mat img = cv::imread(img_name);
    /*End timer*/
    gettimeofday(&npp_read_tv2, NULL);

    double npp_read_fDiffTime = ((double)(npp_read_tv2.tv_usec - npp_read_tv1.tv_usec) / 1000.0) +
                                ((double)(npp_read_tv2.tv_sec - npp_read_tv1.tv_sec) * 1000.0);
    std::cout << "===============> npp read : " << npp_read_fDiffTime << std::endl;

    struct timeval npp_togpu_tv1, npp_togpu_tv2;
    /*Start timer*/
    gettimeofday(&npp_togpu_tv1, NULL);
    unsigned char *org_device_data = nullptr;
    CHECK(cudaMalloc(&org_device_data, img.rows * img.cols * img.channels()));
    CHECK(cudaMemcpy(org_device_data, img.data, img.rows * img.cols * img.channels(), cudaMemcpyHostToDevice));
    /*End timer*/
    gettimeofday(&npp_togpu_tv2, NULL);

    double npp_togpu_fDiffTime = ((double)(npp_togpu_tv2.tv_usec - npp_togpu_tv1.tv_usec) / 1000.0) +
                                 ((double)(npp_togpu_tv2.tv_sec - npp_togpu_tv1.tv_sec) * 1000.0);
    std::cout << "===============> npp togpu : " << npp_togpu_fDiffTime << std::endl;
    total_togpu_fDiffTime += npp_togpu_fDiffTime;
    // double fx = 256.0 / std::min(img.rows, img.cols);
    // int resized_wid = img.cols * fx;
    // int resized_height = img.rows * fx;

    // step1:resize

    struct timeval npp_resize_tv1, npp_resize_tv2;
    /*Start timer*/
    gettimeofday(&npp_resize_tv1, NULL);

    int resized_wid = 224;
    int resized_height = 224;
    unsigned char *resize_device_data = nullptr;
    CHECK(cudaMalloc(&resize_device_data, resized_wid * resized_height * img.channels()));
    bool bret = true;
    bret = resize_image(org_device_data, img.cols, img.rows, img.cols * img.channels(), resize_device_data, resized_wid,
                        resized_height);
    if (!bret)
    {
        sample::gLogError << "resize image failed!" << std::endl;
        return false;
    }

    /*End timer*/
    gettimeofday(&npp_resize_tv2, NULL);

    double npp_resize_fDiffTime = ((double)(npp_resize_tv2.tv_usec - npp_resize_tv1.tv_usec) / 1000.0) +
                                  ((double)(npp_resize_tv2.tv_sec - npp_resize_tv1.tv_sec) * 1000.0);
    std::cout << "===============> npp resize : " << npp_resize_fDiffTime << std::endl;

    // step2:gray

    struct timeval npp_gray_tv1, npp_gray_tv2;
    /*Start timer*/
    gettimeofday(&npp_gray_tv1, NULL);

    unsigned char *gray_device_data = nullptr;
    CHECK(cudaMalloc(&gray_device_data, resized_wid * resized_height * 1));
    bret = gray_image(resize_device_data, resized_wid, resized_height, 3, gray_device_data);
    if (!bret)
    {
        sample::gLogError << "gray image failed!" << std::endl;
        return false;
    }

    /*End timer*/
    gettimeofday(&npp_gray_tv2, NULL);

    double npp_gray_fDiffTime = ((double)(npp_gray_tv2.tv_usec - npp_gray_tv1.tv_usec) / 1000.0) +
                                ((double)(npp_gray_tv2.tv_sec - npp_gray_tv1.tv_sec) * 1000.0);
    std::cout << "===============> npp gray : " << npp_gray_fDiffTime << std::endl;

    // step3:repeat

    struct timeval npp_repeat_tv1, npp_repeat_tv2;
    /*Start timer*/
    gettimeofday(&npp_repeat_tv1, NULL);

    unsigned char *repeat_device_data = nullptr;
    CHECK(cudaMalloc(&repeat_device_data, resized_wid * resized_height * 3));
    bret = repeat_gray_image(gray_device_data, resized_wid, resized_height, 3, repeat_device_data);
    if (!bret)
    {
        sample::gLogError << "repeat image failed!" << std::endl;
        return false;
    }

    /*End timer*/
    gettimeofday(&npp_repeat_tv2, NULL);

    double npp_repeat_fDiffTime = ((double)(npp_repeat_tv2.tv_usec - npp_repeat_tv1.tv_usec) / 1000.0) +
                                  ((double)(npp_repeat_tv2.tv_sec - npp_repeat_tv1.tv_sec) * 1000.0);
    std::cout << "===============> npp repeat : " << npp_repeat_fDiffTime << std::endl;

    // //  crop_image
    // unsigned char *crop_device_data = nullptr;
    // CHECK(cudaMalloc(&crop_device_data, inputW * inputH * img.channels()));
    // CHECK(cudaMemset(crop_device_data, 0, inputW * inputH * img.channels()));
    // int crop_x = std::max((resized_wid - inputW) / 2, 0);
    // int crop_y = std::max((resized_height - inputH) / 2, 0);
    // cv::Rect roi(crop_x, crop_y, inputW, inputH);
    // bret = crop_image(resize_device_data, resized_wid, resized_height, img.channels(), resized_wid *img.channels(),
    //                   roi, crop_device_data);
    // if (!bret)
    // {
    //     sample::gLogError << "crop_image failed!" << std::endl;
    //     return false;
    // }

    // convert to float32
    // float *norm_data = nullptr;
    // CHECK(cudaMalloc(&norm_data, inputW * inputH * img.channels() * sizeof(float)));

    struct timeval npp_tofloat_tv1, npp_tofloat_tv2;
    /*Start timer*/
    gettimeofday(&npp_tofloat_tv1, NULL);

    NppiSize dstsize = {resized_wid, resized_height};
    NPP_CHECK(nppiConvert_8u32f_C3R((Npp8u *)resize_device_data, resized_wid * 3 * sizeof(uchar),
                                    (Npp32f *)deviceDataBuffer, resized_wid * 3 * sizeof(float), dstsize));
    /*End timer*/
    gettimeofday(&npp_tofloat_tv2, NULL);

    double npp_tofloat_fDiffTime = ((double)(npp_tofloat_tv2.tv_usec - npp_tofloat_tv1.tv_usec) / 1000.0) +
                                   ((double)(npp_tofloat_tv2.tv_sec - npp_tofloat_tv1.tv_sec) * 1000.0);
    std::cout << "===============> npp tofloat : " << npp_tofloat_fDiffTime << std::endl;
    // float mean[3] = {0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0};
    // float std_val[3] = {0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0};
    // bret = normlize_image(norm_data, inputW, inputH, img.channels(), mean, std_val);
    // if (!bret)
    // {
    //     sample::gLogError << "normlize_image failed!" << std::endl;
    //     return false;
    // }
    // // HWC -> CHW and BGR -> RGB
    // nppiCopy_32f_C3P3R((Npp32f *)norm_data, inputW * 3 * sizeof(float), rgbPlanes, inputW * sizeof(float),dstsize);

#if 0
      {
        float *host_data = new float[inputH * inputW * 3];
        CHECK(cudaMemcpy(host_data, deviceDataBuffer, inputW * inputH * img.channels() * sizeof(float),
                         cudaMemcpyDeviceToHost));
        fstream file("npp_res.txt", std::ios::out);
        for (size_t c = 0; c < img.channels(); c++) {
          float *ptr_chn = host_data + c * inputH * inputW;
          for (size_t r = 0; r < inputH; r++) {
            file << "[" << c << "," << r << "]:";
            for (size_t col = 0; col < inputW; col++) {
              file << std::setprecision(4) << ptr_chn + r * inputW + col << ", ";
            }
            file << std::endl;
          }
        }
        file.close();
      }
#endif
    cudaFree(org_device_data);
    cudaFree(resize_device_data);
    // cudaFree(crop_device_data);
    // cudaFree(norm_data);
    cudaFree(gray_device_data);
    cudaFree(repeat_device_data);
    std::cout << "*************** NPP out***************" << std::endl;

    return true;
}

bool GoogleNet::verifyOutput(const samplesCommon::BufferManager &buffers)
{
    const int outputSize = mOutputDims.d[1];
    float *output = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};
    if (1)
    {
        std::cout << "output: ";
        for (int i = 0; i < outputSize; i++)
        {
            std::cout << std::fixed << output[i] << ", ";
        }
        std::cout << std::endl;
        return true;
    }
    //  // Calculate Softmax
    //  float sum{0.0f};
    //  int max_idx = -1;
    //  float max_prob = 0.0;
    //  for (int i = 0; i < outputSize; i++) {
    //    output[i] = exp(output[i]);
    //    sum += output[i];
    //  }
    //
    //  sample::gLogInfo << "Output:" << std::endl;
    //  for (int i = 0; i < outputSize; i++) {
    //    output[i] /= sum;
    //    val = std::max(val, output[i]);
    //    if (val == output[i]) {
    //      idx = i;
    //    }
    //    if (output[i] > 0.1) {
    //      std::cout << i << "-->" << output[i] << std::endl;
    //    }
    //    if (max_prob < output[i]) {
    //      max_prob = output[i];
    //      max_idx = i;
    //    }
    //  }
    //  // if (max_prob > 0.9f) {
    //  sample::gLogInfo << " Prob " << max_idx << "  " << std::fixed << std::setw(5)
    //                   << std::setprecision(4) << output[max_idx] << " "
    //                   << "Class " << max_idx << ": "
    //                   << std::string(int(std::floor(output[max_idx] * 10 + 0.5f)), '*') << std::endl;
    //  // }
    //  sample::gLogInfo << std::endl;
    //
    //  return max_prob > 0.8f;
}

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("./../model");
        // params.dataDirs.push_back("data/samples/mnist/");
    }
    else
    {
        params.dataDirs = args.dataDirs;
    }
    // default
    // params.onnxFileName = "dad.trt";
    // params.inputTensorNames.push_back("input_1");
    // params.outputTensorNames.push_back("dense");
    // int
    params.onnxFileName = "mobilenet_int8.plan";
    params.inputTensorNames.push_back("input_1:0");
    params.outputTensorNames.push_back("Identity:0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.preprocess_device = 1;

    return params;
}

void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or "
                 "--datadir=<path to data directory>] [--useDLACore=<int>] [--preprocess=<int>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding "
                 "the default. This option can be used "
                 "multiple times to add multiple directories. If no data "
                 "directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
                 "DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--preprocess=N or -p=N  "
                 "指定前处理所有设备，-1表示使用测试数据，0表示使用CPU做前处理，1表示使用GPU做前处理"
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char **argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    GoogleNet sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    std::cout << "*********** Start infer ***********" << std::endl;
    int num_img = 1000;
    string file_name = "img.jpg";
    for (int i = 0; i < num_img; i++)
    {
        struct timeval tv1, tv2;
        /*Start timer*/
        gettimeofday(&tv1, NULL);
        if (!sample.infer(file_name))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
        /*End timer*/
        gettimeofday(&tv2, NULL);

        double fDiffTime =
            ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
        std::cout << "===============> Inference time of DaD: " << fDiffTime << std::endl;
        total_fDiffTime += fDiffTime;
    }

    std::cout << "+++++++++++++++> " << num_img
              << " frame average whole inference time of DaD : " << total_fDiffTime / num_img << std::endl;
    std::cout << "   +++++++++++++++> " << num_img
              << " frame average pre processing time of DaD : " << total_pre_fDiffTime / num_img << std::endl;
    std::cout << "      +++++++++++++++> " << num_img
              << " frame average copy to cpu time of DaD : " << total_tocpu_fDiffTime / num_img << std::endl;
    std::cout << "   +++++++++++++++> " << num_img
              << " frame average only infer time of DaD : " << total_infer_fDiffTime / num_img << std::endl;
    std::cout << "   +++++++++++++++> " << num_img
              << " frame average copy to gpu time of DaD : " << total_togpu_fDiffTime / num_img << std::endl;

    return sample::gLogger.reportPass(sampleTest);
}
