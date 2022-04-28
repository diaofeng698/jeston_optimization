#include "calibrator.h"
#include "cuda_utils.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/dnn/dnn.hpp>

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char *img_dir,
                                               const char *calib_table_name, const char *input_blob_name,
                                               bool read_cache, int classes_num)
    : batchsize_(batchsize), input_w_(input_w), input_h_(input_h), img_idx_(0), img_dir_(img_dir),
      calib_table_name_(calib_table_name), input_blob_name_(input_blob_name), read_cache_(read_cache),
      classes_num(classes_num)
{
    input_count_ = 3 * input_w * input_h * batchsize;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir(img_dir, img_files_, img_labels_, classes_num);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return batchsize_;
}

bool Int8EntropyCalibrator2::getBatch(void *bindings[], const char *names[], int nbBindings) noexcept
{
    if (img_idx_ + batchsize_ > (int)img_files_.size())
    {
        return false;
    }
    std::cout << "---------- Calibrator File From : " << img_idx_ << " to " << img_idx_ + batchsize_ - 1
              << " ----------" << std::endl;

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++)
    {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_files_[i]);
        if (temp.empty())
        {
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img(temp, input_w_, input_h_);
        input_imgs_.push_back(pr_img);
    }

    img_idx_ += batchsize_;
    cv::Mat blob =
        cv::dnn::blobFromImages(input_imgs_, 1.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);

    CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void *Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *cache, size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char *>(cache), length);
}

int Int8EntropyCalibrator2::getImageNum()
{
    return img_files_.size();
}

std::vector<std::string> Int8EntropyCalibrator2::getImageFiles()
{
    return img_files_;
}

std::vector<int> Int8EntropyCalibrator2::getImageLabels()
{
    return img_labels_;
}
