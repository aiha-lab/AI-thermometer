# Utility functions for performing image inference
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import os
import sys
import time

import tensorrt as trt
import tensorflow as tf
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import utils.engine8_test as engine_utils # TRT Engine creation/save/load utils
import utils.model_test as model_utils # UFF conversion uttils
from utils.paths_test import PATHS # Path management

from calibrator import VOCEntropyCalibrator

# ../../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
        os.pardir
    )
)
import common

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, uff_model_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # If engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            # This function uses supplied .uff file
            # alongside with UffParser to build TensorRT
            # engine. For more details, check implmentation
            
            # Set up for calibration
            calib = None
            if trt_engine_datatype == trt.DataType.INT8:
                with open(PATHS.get_voc_image_set_path(), 'r') as f:
                    voc_image_numbers = f.readlines()
                    voc_image_numbers = [line.strip() for line in voc_image_numbers]
                total_imgs = len(voc_image_numbers)
                voc_names = []

                calibration_cache = "ssd_calibration_eval.cache"
                
                for n in range(total_imgs):
                    voc_names.append(voc_image_numbers[n] + ".jpg")

                _, calib_data = common.find_sample_data(description="Runs a ResNet50 network in Int8 mode", subfolder="JPEGImages", find_files=voc_names)
                calib = VOCEntropyCalibrator(calib_data, total_imgs, cache_file=calibration_cache)
            
            self.trt_engine = engine_utils.build_engine(
                uff_model_path, calib, TRT_LOGGER,
                trt_engine_datatype=trt_engine_datatype,
                batch_size=batch_size)
            # Save the engine to file
            engine_utils.save_engine(self.trt_engine, trt_engine_path)

        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = engine_utils.load_engine(
                self.trt_runtime, trt_engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_utils.allocate_buffers(self.trt_engine)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(model_utils.ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, image_path):
        """Infers model on given image.

        Args:
            image_path (str): image to run object detection model on
        """

        # Load image into CPU
        img = self._load_img(image_path)

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        [detection_out, keepCount_out] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))

        # And return results
        return detection_out, keepCount_out

    def infer_batch(self, image_paths, inf_time):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """

        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(image_paths)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image_paths list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

        # Load all images to CPU...
        imgs = self._load_imgs(image_paths)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, imgs.ravel())
        
        inf_start_time = time.time()

        # ...fetch model outputs...
        detection_out = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size)
        
        inf_time = inf_time + (time.time() - inf_start_time)

        
        '''
        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inf_start_time) * 1000))))
        '''

        # ...and return results.
        return detection_out, inf_time

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, model_utils.ModelData.get_input_channels())
        ).astype(np.uint8)

    def _load_imgs(self, image_paths):
        batch_size = self.trt_engine.max_batch_size
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            self.numpy_array[idx] = img_np
        return self.numpy_array


    def _load_img(self, image_path):
        image = Image.open(image_path)
        model_input_width = model_utils.ModelData.get_input_width()
        model_input_height = model_utils.ModelData.get_input_height()
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np


