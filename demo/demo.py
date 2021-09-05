"""trt_ssd_async.py

This is the 'async' version of trt_ssd.py implementation.  It creates
1 dedicated child thread for fetching camera input and do inferencing
with the TensorRT optimized SSD model/engine, while using the main
thread for drawing detection results and displaying video.  Ideally,
the 2 threads work in a pipeline fashion so overall throughput (FPS)
would be improved comparing to the non-async version.
"""


import time
import argparse
import threading

import numpy as np
import cv2
import pycuda.driver as cuda

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps, show_temperature, show_waiting, show_status
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'Thermal Camera Face Detection Demo'
MAIN_THREAD_TIMEOUT = 10.0  # 10 seconds
INPUT_HW = (300, 300)

# These global variables are 'shared' between the main and child
# threads.  The child thread writes new frame and detection result
# into these variables, while the main thread reads from them.
s_img, s_boxes, s_confs, s_clss = None, None, None, None
t_img, t_boxes, i = None, None, 0
temp_mode = False

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-p', '--precision', type=int,
                        default='32',
                        choices=[32, 16, 8])
    '''
    parser.add_argument('-m', '--model', type=str,
                        default='ssd_mobilenet_v1_coco',
                        choices=SUPPORTED_MODELS)
    '''
    args = parser.parse_args()
    return args


class TrtThread(threading.Thread):
    """TrtThread

    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the TrtThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, condition, cam, precision, conf_th):
        """__init__

        # Arguments
            condition: the condition variable used to notify main
                       thread about new frame and detection result
            cam: the camera object for reading input image frames
            model: a string, specifying the TRT SSD model
            conf_th: confidence threshold for detection
        """
        threading.Thread.__init__(self)
        self.condition = condition
        self.cam = cam
        self.precision = precision
        self.conf_th = conf_th
        self.cuda_ctx = None  # to be created when run
        self.trt_ssd = None   # to be created when run
        self.running = False

    def run(self):
        """Run until 'running' flag is set to False by main thread.

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """
        global s_img, s_boxes, s_confs, s_clss

        print('TrtThread: loading the TRT SSD engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_ssd = TrtSSD(self.precision, INPUT_HW)
        
        print('TrtThread: start running...')
        self.running = True
        while self.running:
            
            img = self.cam.read()
            
            if img is None:
                break
            # start = time.time()
            boxes, confs, clss = self.trt_ssd.detect(img, self.conf_th)
            # print("detect")
            # print("detect : {:.6f} ms".format((time.time()-start)*1000))
            with self.condition:
                s_img, s_boxes, s_confs, s_clss = img, boxes, confs, clss
                self.condition.notify()
        del self.trt_ssd
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()



class DisplayThread(threading.Thread):
    def __init__(self, condition, condition2, vis, width, height):
        threading.Thread.__init__(self)
        self.condition = condition
        self.condition2 = condition2
        self.vis = vis
        self.width = width
        self.height = height
        self.running = False

    def run(self):
        global s_img, s_boxes, s_confs, s_clss, t_img, t_boxes, i, temp_mode
        
        print('DisplayThread: start running...')
        self.running = True
        
        full_scrn = False
        fps = 0.0
        tic = time.time()
        sum_fps = 0
        
        while self.running:
            with self.condition:
                if self.condition.wait(timeout=MAIN_THREAD_TIMEOUT):
                    img, confs, clss = s_img, s_confs, s_clss
                    if ((i % 50) == 0):
                        boxes = s_boxes
                else:
                    raise SystemExit('ERROR: timeout waiting for img from child')
            # start = time.time()
            if temp_mode:
                img = self.vis.draw_bboxes(img, boxes, confs, clss)
            
            # print("bbox")
            # print("draw : {:.6f} ms".format((time.time()-start)*1000))
            
            # img = show_fps(img, fps)
            # cv2.imshow(WINDOW_NAME, img)
            t_img = img
            t_boxes = boxes
            with self.condition2:
                self.condition2.notify()
            
            '''
            toc = time.time()
            # curr_fps = 1.0 / (toc - tic)
            fps = 1.0 / (toc - tic)
            sum_fps = sum_fps + fps

            if((t_i % 150) == 149):
                print("\nFPS : {:.2f}".format(sum_fps/t_i))
                # print(t_mode)
            
            # calculate an exponentially decaying average of fps number
            # fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            '''
            
        print('DisplayThread: stopped...')
        
    def stop(self):
        self.running = False
        self.join()

def temperature(condition2):
    
    global t_img, t_boxes, i, temp_mode
    j = 0
    sum_temps = 0
    avg_temp = 0

    full_scrn = False
    fps = 0.0
    tic = time.time()
    sum_fps = 0

    record = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    stat = None
    black = np.zeros((120,160,3), np.uint8)
    
    while True:

        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break
        with condition2:
            if condition2.wait(timeout=MAIN_THREAD_TIMEOUT):
                img, boxes = t_img, t_boxes
            else:
                raise SystemExit('ERROR: temperature() timeout waiting')
        
        temp = 0

        # start = time.time()
        if(len(boxes) != 0):

            temp_mode_past = temp_mode
            temp_mode = True
            if(temp_mode_past != temp_mode):
                i, j, sum_fps, fps, temp, sum_temps, avg_temps = 0, 0, 0, 0, 0, 0, 0

            (x_min, y_min, x_max, y_max) = boxes[0]    
            num_pixels = 0
            for y in range(y_min+1, y_max-1):
                for x in range(x_min+1, x_max-1):
                    b = img.item(y, x, 0)
                    g = img.item(y, x, 1)
                    r = img.item(y, x, 2)

                    if(((b+g+r) > 550) and ((b+g+r) < 570)):
                        temp = (b + g + r)*0.556 + temp
                        num_pixels = num_pixels + 1

            if(num_pixels != 0):
                temp = round((temp / num_pixels - 273.15), 2)
            
            elif(num_pixels == 0):
                temp = 0
            
            if(temp != 0):
                j = j + 1
                sum_temps = sum_temps + temp

            if(((i % 50) == 0) and (j != 0)):
                avg_temp = sum_temps/j
            
            img = show_temperature(img, avg_temp)
            
            if(avg_temp > 37.5):
                stat = 'Fever'
                color = (32, 32, 240)
                img = show_status(img, stat, color)
            
            else:
                stat = 'Normal'
                color = (32, 240, 32)
                img = show_status(img, stat, color)

        elif(len(boxes) == 0):
            temp_mode_past = temp_mode
            temp_mode = False
            if(temp_mode_past != temp_mode):
                i, j, sum_fps, fps, temp, sum_temps, avg_temps = 0, 0, 0, 0, 0, 0, 0

        # print("temp")
        # print("temp : {:.6f} ms".format((time.time()-start)*1000))
        
        # start = time.time()
        if temp_mode:
            cv2.imshow(WINDOW_NAME, img)
        else:
            # img = black
            img = show_waiting(img)
            cv2.imshow(WINDOW_NAME, img)

        
        # print("display : {:.6f} ms".format((time.time()-start)*1000))
        # print("display")

        i = i + 1
        key = cv2.waitKey(1)
            
        toc = time.time()
        # curr_fps = 1.0 / (toc - tic)
        fps = 1.0 / (toc - tic)
        sum_fps = sum_fps + fps

        if((i % 150) == 149):
            print("\nFPS : {:.2f}".format(sum_fps/i))
            # print(t_mode)

        # calculate an exponentially decaying average of fps number
        # fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        
        if key == 27:  # ESC key: quit program
            break

        elif key == ord('r'):
            record = True
            video = cv2.VideoWriter("./ucc_video.avi", fourcc, 100, (160, 120))
            print("Recording...")
        
        elif key == ord('e'):
            record = False
            print("Video saved")
            video.release()

        elif key == ord('c'):
            time_now = time.strftime('%c', time.localtime(time.time()))
            cv2.imwrite('./capture/'+time_now+'.jpg',img)
            print('Captured')

        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        
        elif key == ord('t'): # Toggle temperature mode
            temp_mode = not temp_mode
            i = 0
            j = 0
            sum_fps = 0
            fps = 0
            temp = 0
            sum_temps = 0
            avg_temps = 0

        if record == True:
            video.write(img)


def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cuda.init()  # init pycuda driver

    # cls_dict = get_cls_dict(args.model.split('_')[-1])
    cls_dict = get_cls_dict('wider')

    open_window(WINDOW_NAME, 'Camera TensorRT SSD Demo', 480, 360)

    width = cam.img_width
    height = cam.img_height
    
    start_time = time.time()
    vis = BBoxVisualization(cls_dict)
    
    condition = threading.Condition()
    condition2 = threading.Condition()
    
    trt_thread = TrtThread(condition, cam, args.precision, conf_th=0.7)
    disp_thread = DisplayThread(condition, condition2, vis, width, height)
    
    trt_thread.start()  # start the threads
    disp_thread.start()
    
    temperature(condition2)
    
    trt_thread.stop()   # stop the threads
    disp_thread.stop()
    
    end_time = time.time()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
