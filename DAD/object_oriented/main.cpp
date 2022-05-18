#include "tensorrt_detection.h"

int main()
{
    std::shared_ptr<TensorRTInference> DaD_net;

    std::string input_image_path =
        "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/06_googlenet/model/img.png";

    std::string DaDModelPath = std::string("/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/"
                                           "06_googlenet/model/model_cpp_int8.plan");

    DaD_net = std::make_shared<TensorRTInference>(DaDModelPath, "input_1:0", "Identity:0", 2);

    cv::Mat input_frame = cv::imread(input_image_path);

    if (DaD_net->TensorRTInfer(input_frame))
    {
        std::cout << "Inference Failed " << std::endl;
        return EXIT_FAILURE;
    }
}