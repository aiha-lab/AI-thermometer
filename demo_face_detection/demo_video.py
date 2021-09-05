"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""


import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
# from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

import time

WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_inception_v2_coco',
    'ssdlite_mobilenet_v2_coco',
]


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    # parser = add_camera_args(parser)
    parser.add_argument('-p', '--precision', type=int,
                        default='32',
                        choices=[32, 16, 8])
    args = parser.parse_args()
    return args


def loop_and_detect(det_time, cam, trt_ssd, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    record = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    full_scrn = False
    fps = 0.0
    tot_frames = 0
    tic = time.time()
    
    start_time = time.time()

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        ret, frame = cam.read()
        if frame is None:
            break
        tot_frames = tot_frames + 1
        boxes, confs, clss, det_time = trt_ssd.detect(frame, det_time, conf_th)
        frame = vis.draw_bboxes(frame, boxes, confs, clss)
        frame = show_fps(frame, fps)
        cv2.imshow(WINDOW_NAME, frame)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('r'):
            record = True
            video = cv2.VideoWriter("./face_video.avi", fourcc, 20, (width, height))
            print("Recording...")
        elif key == ord('e'):
            record = False
            video.release()
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

        if record == True:
            video.write(frame)
    
    
    end_time = time.time()
    
    print("Total frames : {}".format(tot_frames))
    print("Average FPS : {:.2f} s\n".format(
        tot_frames / (end_time - start_time)))
    
    # print(tot_frames/det_time)
    return det_time


def main():
    args = parse_args()
    cam = cv2.VideoCapture('./People.mp4')
    '''
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    '''
    cls_dict = get_cls_dict('wider')
    trt_ssd = TrtSSD(args.precision, INPUT_HW)
    
    width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    open_window(
        WINDOW_NAME, 'Camera TensorRT SSD Demo',
        width, height)
    vis = BBoxVisualization(cls_dict)
    
    det_time = 0
    start_time = time.time()
    det_time = loop_and_detect(det_time, cam, trt_ssd, conf_th=0.3, vis=vis)
    end_time = time.time()
    print("Video play time : {:.2f} s\n".format(
        (end_time - start_time)))
    print(det_time)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
