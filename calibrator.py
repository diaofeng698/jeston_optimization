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

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

from tqdm import tqdm
import pandas as pd
import cv2

dataset_folder = '/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/SF_dataset'
driver_file = '/home/fdiao/Downloads/TensorRT-8.2.1.8/samples/python/int8_caffe_mnist/SF_dataset/driver_imgs_list.csv'
train_image = []
classes=10


def load_train_image():
    
    driver_details = pd.read_csv(os.path.join(driver_file), na_values='na')
    driver_details.set_index('img')
    print('driver list detail: ', driver_details.head(5))
    driver_list = list(driver_details['subject'].unique())
    print('driver list:', driver_list)
    print('total drivers:', len(driver_list))
    print(driver_details.groupby(by=['subject'])['img'].count())
    class_distribution = driver_details.groupby(by=['classname'])['img'].count()
    img_quantity = list(class_distribution.values)
    print('img amount of every class: ', img_quantity)
    
    for folder_index in range(classes):
        class_folder = 'c' + str(folder_index)
        print(f'now we are in the folder {class_folder}')

        imgs_folder_path = os.path.join(dataset_folder, 'train', class_folder)
        imgs = os.listdir(imgs_folder_path)

        for img_index in tqdm(range(len(imgs))):
            img_path = os.path.join(imgs_folder_path, imgs[img_index])
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))

            # ********difference**********
            img = np.array(img, dtype=np.float32)

            img = img[..., np.newaxis]
            img = np.repeat( img,3, -1)
            label = folder_index
            driver = driver_details[driver_details['img'] == imgs[img_index]]['subject'].values[0]

            train_image.append([img, label, driver])

    print('total images:', len(train_image))

    driver_valid_list = {}

    driver_test_list = {}

    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test, y_test = [], []

    for image, label, driver in train_image:
        if driver in driver_test_list:
          X_test.append(image)
          y_test.append(label)
        elif driver in driver_valid_list:
          X_valid.append(image)
          y_valid.append(label)
        else:
            X_train.append(image)
            y_train.append(label)

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)

    # X_train = np.array(X_train).reshape(-1, 224, 224, 3)
    # X_valid = np.array(X_valid).reshape(-1, 224, 224, 3)
    # X_test_array = np.array(X_test).reshape(-1, 224, 224, 3)
    # print(f'X_train shape: {X_train.shape}')
    # print(f'X_valid shape: {X_valid.shape}')
    # print(f'X_test shape: {X_test_array.shape}')

    # y_train = np.array(y_train)
    # y_valid = np.array(y_valid)
    # y_test_array = np.array(y_test)
    # print(f'y_train shape: {y_train.shape}')
    # print(f'y_valid shape: {y_valid.shape}')
    # print(f'y_test shape: {y_test_array.shape}')

    return X_train,y_train

class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = training_data
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

