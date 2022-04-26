#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cgi import test
from pytools import F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import random
import cv2
from torch import classes

# For our custom calibrator
from calibrator import MNISTEntropyCalibrator,load_train_image

# For ../common.py
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common
import time


TRT_LOGGER = trt.Logger()
onnx_model_path = "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/weights/mobilenet.onnx"
      
def build_engine(onnx_file_path, calib, batch_size=32,engine_file_path=""):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        builder.max_batch_size = batch_size
        config.max_workspace_size = common.GiB(1)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please irst to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
            

        network.get_input(0).shape = [1, 224, 224, 3]
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)

        # profile =builder.create_optimization_profile()
        # profile.set_shape("input_1:0",(1,224,224,3),(1,224,224,3),(1,224,224,3))
        # config.add_optimization_profile(profile)
        # engine = builder.build_engine(network, config)    # after deserialize for inference
        # plan = engine.serialize()


        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine



def inference_image(engine):
    # input & output
    image_path = "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/06_googlenet/model/img.png"
    image_cv = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    image_resize = cv2.resize(image_gray, (224, 224))
    image = np.array(image_resize, dtype=np.float32)
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)

    input = np.array(image, dtype=np.float32)
    
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    # np.copyto(inputs[0].host, input)
    # trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    inputs[0].host = input
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    prediction = np.argmax(trt_outputs[0])

    print("outputs:", trt_outputs)
    print("prediction:", prediction)



def check_accuracy(engine, classes):


    cal_dir = "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/SF_dataset/train"
    diff_time = 0.0
    correct_num = 0
    total_num = 0
    
    for folder_num in range(classes):
        folder = 'c' + str(folder_num)
        for file in os.listdir(os.path.join(cal_dir, folder)):
            image_path = os.path.join(cal_dir, folder, file)

            start_time = time.time()
            image_cv = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            image_resize = cv2.resize(image_gray, (224, 224))
            image = np.array(image_resize, dtype=np.float32)
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)

            test = np.array(image, dtype=np.float32)

            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            context = engine.create_execution_context()

            inputs[0].host = test
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,batch_size=1)
            prediction = np.argmax(trt_outputs[0])
            end_time = time.time()

            diff_time += end_time-start_time
            
            if prediction == folder_num:
                correct_num +=1
            total_num += 1
    percent_correct = 100 * correct_num / float(total_num)
    average_time = 1000 * diff_time / float(total_num)
    print("Total NUm : {:}".format(total_num))
    #print("Average Inference Time : {:}ms".format(average_time))
    print("Total Accuracy: {:}%".format(percent_correct))


def main():
    # _, data_files = common.find_sample_data(description="Runs a Caffe MNIST network in Int8 mode", subfolder="mnist", find_files=["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "train-images-idx3-ubyte", ModelData.DEPLOY_PATH, ModelData.MODEL_PATH], err_msg="Please follow the README to download the MNIST dataset")
    # [test_set, test_labels, train_set, deploy_file, model_file] = data_files

    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    data = load_train_image()
    test_set = data[0]
    test_label = data[1]

    calibration_cache = "mobilenet_calibration.cache"
    engine_file_path = "/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/DAD/06_googlenet/model/mobilenet_int8.plan"
    

    batch_size = 8 
    calib = MNISTEntropyCalibrator(test_set,cache_file=calibration_cache,batch_size=batch_size)

    # Inference batch size can be different from calibration batch size.
    #inference_image(build_engine(onnx_model_path, calib, batch_size,engine_file_path))


    print("Start build engine ! ")
    with build_engine(onnx_model_path, calib, batch_size,engine_file_path) as engine:
        # Batch size for inference can be different than batch size used for calibration.
        check_accuracy(engine=engine,classes=10)

if __name__ == '__main__':
    main()
