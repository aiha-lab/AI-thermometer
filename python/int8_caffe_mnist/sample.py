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

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# For our custom calibrator
from calibrator import load_mnist_data, load_mnist_labels, MNISTEntropyCalibrator

# For ../common.py
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common
import time
import argparse
import logging

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    8: trt.DataType.INT8,
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

class ModelData(object):
    DEPLOY_PATH = "deploy.prototxt"
    MODEL_PATH = "mnist_lenet.caffemodel"
    OUTPUT_NAME = "prob"
    # The original model is a float32 one.
    DTYPE = trt.float32



def parse_commandline_arguments():
    parser = argparse.ArgumentParser(description='Run classification inference on MNIST.')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16, 8], default=32,
        help='desired TensorRT float precision to build an engine with')
    
    # Parse arguments passed
    args = parser.parse_args()
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    
    return args

def get_engine_path(precision, batch_size):
    return os.path.join(
        './engine/',
        'engine_{}_{}.buf'.format(precision, batch_size))

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        logger.error("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))

def check_network(network):
    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])
    max_layer_len = max([len(layer.name) for layer in layers])

    logger.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logger.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, layer in enumerate(layers):
        logger.debug("Layer  {0} | Name: {1:{2}} | Type: {3} | Precision: {4}".format(
            i, layer.name, max_layer_len, layer.type, layer.precision_is_set))
    for i, out in enumerate(outputs):
        logger.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))

# This function builds an engine from a Caffe model.
def build_int8_engine(deploy_file, model_file, batch_size=32, trt_engine_datatype=trt.DataType.FLOAT):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        builder.max_workspace_size = common.GiB(1)
        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
            # builder.strict_type_constraints = True
        elif trt_engine_datatype == trt.DataType.INT8:
            # Now we create a calibrator and give it the location of our calibration data.
            # We also allow it to cache calibration data for faster engine building.
            _, [calib_data] = common.find_sample_data(description="Runs a Caffe MNIST network in Int8 mode", subfolder="mnist", find_files=["t10k-images-idx3-ubyte"])
            calibration_cache = "mnist_calibration.cache"
            builder.int8_mode = True
            # builder.strict_type_constraints = True
            builder.int8_calibrator = MNISTEntropyCalibrator(calib_data, cache_file=calibration_cache)
        # Parse Caffe model
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        
        check_network(network)

        # Build engine and do int8 calibration.
        return builder.build_cuda_engine(network)


def check_accuracy(context, inf_time, batch_size, test_set, test_labels):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)

    num_correct = 0
    num_total = 0

    batch_num = 0
    
    inf_time_step = 0
    # start_time = time.time()

    for start_idx in range(0, test_set.shape[0], batch_size):
        batch_num += 1
        
        '''
        if batch_num % 20 == 0:
            print("Validating batch {:}".format(batch_num))
        '''

        # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
        # This logic is used for handling that case.
        end_idx = min(start_idx + batch_size, test_set.shape[0])
        effective_batch_size = end_idx - start_idx
        
        # Do inference for every batch.
        inputs[0].host = test_set[start_idx:start_idx + effective_batch_size]
        
        start_time = time.time()
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=effective_batch_size)
        inf_time_step = inf_time_step + time.time() - start_time
        # Use argmax to get predictions and then check accuracy
        preds = np.argmax(output.reshape(64, 10)[0:effective_batch_size], axis=1)
        labels = test_labels[start_idx:start_idx + effective_batch_size]
        num_total += effective_batch_size
        num_correct += np.count_nonzero(np.equal(preds, labels))
    
    print("TensorRT inference time: {:.3f} ms".format(
            (inf_time_step) * 1000))
    
    inf_time = inf_time + inf_time_step 
    
    percent_correct = 100 * num_correct / float(num_total)
    # print("Total Accuracy: {:}%".format(percent_correct))
    
    return inf_time

def main():

    # Parse command line arguments
    args = parse_commandline_arguments()
    
    _, data_files = common.find_sample_data(description="Runs a Caffe MNIST network in Int8 mode", subfolder="mnist", find_files=["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", ModelData.DEPLOY_PATH, ModelData.MODEL_PATH])
    [test_set, test_labels, train_set, deploy_file, model_file] = data_files

    # Inference batch size can be different from calibration batch size.
    batch_size = 64
    inf_time = 0
    engine = None
    trt_engine_path = get_engine_path(args.precision, batch_size)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    if not os.path.exists(trt_engine_path):
        # Build a TensorRT engine.
        engine = build_int8_engine(deploy_file, model_file, batch_size, trt_engine_datatype=args.trt_engine_datatype)
        # Save the engine to file
        buf = engine.serialize()
        with open(trt_engine_path, 'wb') as f:
            f.write(buf)

    # If we get here, the file with engine exists, so we can load it
    if not engine:
        print("Loading cached TensorRT engine from {}".format(trt_engine_path))
        with open(trt_engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)    
    
    for i in range(50):
        with engine.create_execution_context() as context:
            # Batch size for inference can be different than batch size used for calibration.
            inf_time = check_accuracy(context, inf_time,  batch_size, test_set=load_mnist_data(test_set), test_labels=load_mnist_labels(test_labels))
    
    print("Average inference time: {:.3f} ms".format(
            (inf_time) * 1000 / 50))



if __name__ == '__main__':
    main()
