/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!
#include "argsParser.h"
#include "buffers.h"
#include "calibrator.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

using samplesCommon::SampleUniquePtr;
struct SampleINT8Params : public samplesCommon::OnnxSampleParams
{
    int nbCalBatches;        //!< The number of batches for calibration
    int calBatchSize;        //!< The calibration batch size
    std::string networkName; //!< The name of the network
    int classesNum;
};

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
  public:
    SampleOnnxMNIST(const SampleINT8Params &params) : mParams(params), mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(DataType dataType);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(std::string file_name);

  private:
    SampleINT8Params mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    int inputChannel{0};
    int inputH{0};
    int inputW{0};
    int inputB{0};

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                          SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                          SampleUniquePtr<nvonnxparser::IParser> &parser, DataType dataType, int classesNum);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers, std::string file_name);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build(DataType dataType)
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8()) ||
        (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser =
        SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser, dataType, mParams.classesNum);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // Input shape
    Dims32 dim;
    dim.nbDims = 4;
    dim.d[0] = 1;
    dim.d[1] = 224;
    dim.d[2] = 224;
    dim.d[3] = 3;
    network->getInput(0)->setDimensions(dim);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    std::ofstream planFile("/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/sampleINT8/"
                           "model/model_cpp_int8.plan",
                           std::ios::binary);
    planFile.write(static_cast<char *>(plan->data()), plan->size());

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()),
                                                     samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    inputB = mInputDims.d[0];
    inputH = mInputDims.d[1];
    inputW = mInputDims.d[2];
    inputChannel = mInputDims.d[3];

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                                       SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                                       SampleUniquePtr<nvonnxparser::IParser> &parser, DataType dataType,
                                       int classesNum)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    network->markOutput(*network->getOutput(0));

    // Calibrator life time needs to last until after the engine is built.

    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
    config->setMaxWorkspaceSize(1_GiB);
    if (dataType == DataType::kINT8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    builder->setMaxBatchSize(mParams.batchSize);

    if (dataType == DataType::kINT8)
    {

        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(
            mParams.batchSize, inputW, inputH,
            "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/SF_dataset/train/c",
            "../model/mobilenetint8.cache", "input_1:0", false, classesNum);
        config->setInt8Calibrator(calibrator);
    }

    if (mParams.dlaCore >= 0)
    {
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        if (mParams.batchSize > builder->getMaxDLABatchSize())
        {
            sample::gLogError << "Requested batch size " << mParams.batchSize
                              << " is greater than the max DLA batch size of " << builder->getMaxDLABatchSize()
                              << ". Reducing batch size accordingly." << std::endl;
            return false;
        }
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer(std::string file_name)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, file_name))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager &buffers, std::string file_name)
{
    // const int inputH = mInputDims.d[2];
    // const int inputW = mInputDims.d[3];

    // // Read a random digit file
    // srand(unsigned(time(nullptr)));
    // std::vector<uint8_t> fileData(inputH * inputW);
    // mNumber = rand() % 10;
    // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // // Print an ascii representation
    // sample::gLogInfo << "Input:" << std::endl;
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    // }
    // sample::gLogInfo << std::endl;

    // float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    // }

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

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
// bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager &buffers)
// {
//     const int outputSize = mOutputDims.d[1];
//     float *output = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
//     float val{0.0f};
//     int idx{0};

//     // Calculate Softmax
//     float sum{0.0f};
//     for (int i = 0; i < outputSize; i++)
//     {
//         output[i] = exp(output[i]);
//         sum += output[i];
//     }

//     sample::gLogInfo << "Output:" << std::endl;
//     for (int i = 0; i < outputSize; i++)
//     {
//         output[i] /= sum;
//         val = std::max(val, output[i]);
//         if (val == output[i])
//         {
//             idx = i;
//         }

//         sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
//                          << " "
//                          << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
//                          << std::endl;
//     }
//     sample::gLogInfo << std::endl;

//     return idx == mNumber && val > 0.9f;
// }

bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager &buffers)
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
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleINT8Params initializeSampleParams(const samplesCommon::Args &args)
{
    SampleINT8Params params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
        params.dataDirs.push_back(
            "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/sampleINT8/model/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    // params.onnxFileName = "mnist.onnx";
    // params.inputTensorNames.push_back("Input3");
    // params.outputTensorNames.push_back("Plus214_Output_0");
    params.onnxFileName = "mobilenet.onnx";
    // params.inputTensorNames.push_back("Input3");
    // params.outputTensorNames.push_back("Plus214_Output_0");
    params.inputTensorNames.push_back("input_1:0");
    params.outputTensorNames.push_back("Identity:0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.batchSize = 1;
    params.calBatchSize = 1;
    params.classesNum = 10;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
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

    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;
    std::vector<DataType> dataTypes = {DataType::kINT8};

    if (!sample.build(dataTypes[0]))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    string file_name = "img.png";
    if (!sample.infer(file_name))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
