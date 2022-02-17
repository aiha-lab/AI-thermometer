# AI-thermometer
This is a project which implement high-speed AI thermometer.

## Key features
### TensorRT
The goal was implementing AI thermometer on embedded system using deep learning-based object detection model. So we optimized SSD(Single-shot multibox detector) via TensorRT, which is the library of NVIDIA for high-performance deep learning inference. Note that the TensorRT version which we used is 6.0.1
### Using thermal imaging camera only (w/o normal RGB camera)
In general, most of the AI-based thermometers are using not only normal RGB camera but also thermal imaging camera. Because the face detection should work on normal images. Therefore, we did transfer learning, retrained SSD on custom dataset of thermal face images(using TensorFlow object detection API). We constructed the dataset by capture images on thermal imaging camera manually.
### Pipeline sturcture
The demo code(demo/demo.py) is operates in pipeline fashion. The entire workload is split into threads, and each thread works separately so overall throughput (FPS) is improved comparing to serial way. I got this idea from JK Jung's code (https://github.com/jkjung-avt/tensorrt_demos/blob/master/trt_ssd_async.py).

## Environment
* Platform: Jetson AGX Xavier Developer kit
* Thermal imaging camera: FLIR Lepton 3.5 + GroupGets LLC Purethermal 2
* Libraries: TensorRT (Release 6.0.1), OpenCV, NumPy, PyCUDA, etc.

## Configuration
### demo/
Final demo of this project. demo.py performs real-time object detection on thermal camera input. The system operates in Wait-Mode if there is no person in front of the camera. When a human face appears, the system enters Measure-Mode, then displays the bounding box as well as the current temperature. The system also displays messages. For example, it shows "Fever" for an alert if the measured temperature exceeds 37.5℃; otherwise, it prints "Normal." When the human face disappears again, the system goes back to Wait-Mode.

<img src="demo/results/1.jpg" width="240px" height="180px"></img>
<img src="demo/results/2.jpg" width="240px" height="180px"></img><br/>
<img src="demo/results/demo_test_1.gif" width="240px" height="180px"></img>
<img src="demo/results/demo_test_2.gif" width="240px" height="180px"></img>

### demo_face_detection/
This demo is only for face recognition on normal images(not for thermal images).

<img src="demo_face_detection/demo_face_detection_demo_result.gif" width="320px" height="180px"></img><br/>

### python/
TensorRT sample python codes we used, and their variations. We applied INT8 quantization and calibration scheme in 'int_caffe_mnist' to 'uff_ssd'. So we were able to implement the backbone network(MobileNet-v2) of SSD in INT8 precision.
* uff_ssd: Object Detection with SSD in Python (Original TensorRT sample)
* uff_ssd_mb2: Chage the backbone network from Inception-v2 to MobileNet-v2. (The backbone is Inception-v2 in original sample codes)
* uff_ssd_mb2_wider: The MobileNet-v2-based SSD model is retrained on WIDER FACE dataset for only face detection. (transfer learning)
* uff_ssd_mb2_thermal: Retrain the model on custom thermal image dataset.
* int8_caffe_mnist: INT8 Calibration In Python (Original TensorRT sample)

## References
* TensorRT demo code: https://github.com/jkjung-avt/tensorrt_demos
* TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
* TensorRT python example codes: https://github.com/NVIDIA/TensorRT/tree/master/samples/python
* TensorFlow object detection API: https://github.com/tensorflow/models/tree/master/research/object_detection
* TensorFlow 1 Detection Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
* WIDER FACE dataset: http://shuoyang1213.me/WIDERFACE/
* SSD paper: https://arxiv.org/abs/1512.02325
* Another AI-thermomerter project: https://github.com/tomek-l/ai-thermometer
