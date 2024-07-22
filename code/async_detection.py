import copy
from abc import ABC, abstractmethod
import cv2
import torch.cuda
import json
import numpy as np
from PIL import Image
from torch.utils.collect_env import get_gpu_info
from ultralytics import YOLO
from torch.cuda.amp import autocast
import time
import torch
from ultralytics import YOLO
import subprocess
from multiprocessing import Process, Queue, Lock, shared_memory
import threading
import queue

import torch
import tensorrt as trt
# from torch2trt import torch2trt

import asyncio

class YOLOv8Model:
    def __init__(self):
        pass

    def load_model(self, path_to_model_weights, device):
        # Check for CUDA device and set it
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        current_model = YOLO(path_to_model_weights).to(device=device)
        return current_model

    async def detect_objects(self, frame, model, current_model_conf, image_dislayer, labels_translator=None):
        if labels_translator is not None:
            labels_translator = {int(k): v for k, v in labels_translator.items()}
        details_nn_results = model(frame, imgsz=1920)
        for item in details_nn_results:
            boxes = item.boxes.cpu().numpy()  # get boxes on cpu in numpy
            for obj in boxes.data:
                if obj[4] > current_model_conf:
                    if int(obj[5]) in labels_translator:
                        descr = labels_translator[int(obj[5])]
                        image_dislayer.one_object_display(frame, obj, str(descr + ' ' + str(round(obj[4], 2))))

    def get_gpu_info(self):
        gpu_info = subprocess.check_output("nvidia-smi", shell=True)
        gpu_info = gpu_info.decode("utf-8")
        print(gpu_info)

    async def run_three_models(self, translator_1, weights_1, cur_model_conf_1,
                         translator_2, weights_2, cur_model_conf_2,
                         translator_3, weights_3, cur_model_conf_3,
                         captor, displayer):

        print(f"Torch cuda is available: {torch.cuda.is_available()}")

        start_time = time.time()
        fps_list = []
        if torch.cuda.is_available():
            device1 = torch.device('cuda:0')
            device2 = torch.device('cuda:1')
        yolo_v8_class_obj = YOLOv8Model()
        yolo_v8_class_obj.get_gpu_info()
        palm_model = yolo_v8_class_obj.load_model(weights_1, device1)
        drone_tools_model = yolo_v8_class_obj.load_model(weights_2, device2)
        drone_details_model = yolo_v8_class_obj.load_model(weights_3, device1)
        # video_stream = VideoCaptureAsync().start()
        while True:
            time_1 = time.time()
            ret, frame = captor.get_frame()
            # frame = video_stream.read()
            if frame is None:
                break

            await asyncio.gather(
                self.detect_objects(frame=frame,
                                    model=palm_model,
                                    current_model_conf=cur_model_conf_1,
                                    image_dislayer=displayer,
                                    labels_translator=translator_1),
                # self.detect_objects(frame=frame,
                #                     model=drone_tools_model,
                #                     current_model_conf=cur_model_conf_2,
                #                     image_dislayer=displayer,
                #                     labels_translator=translator_2),
                self.detect_objects(frame=frame,
                                    model=drone_details_model,
                                    current_model_conf=cur_model_conf_3,
                                    image_dislayer=displayer,
                                    labels_translator=translator_3)
            )


            displayer.display_frame_with_labels(frame)

            elapsed_time = time.time() - time_1
            fps = 1 / elapsed_time
            print("FPS", fps)
            fps_list.append(fps)

            if time.time() - start_time >= 60:
                avg_fps = sum(fps_list) / len(fps_list)
                print("Средний FPS за 1 минут: ", avg_fps)
                fps_list = []
                start_time = time.time()

            pressed_key = cv2.waitKey(1)
            if pressed_key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


class ConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = {}
        self.load_config()

    def load_config(self):
        with open(self.file_path, 'r', encoding="UTF-8") as file:
            self.config = json.load(file)

    def get_config(self):
        return self.config


class ImageCaptor(object):
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.vid.set(cv2.CAP_PROP_FPS, 30)
        # Инициализировать объект записи видео
        self.output = cv2.VideoWriter(f'output_video_{video_source}.mp4',
                                      cv2.VideoWriter.fourcc('M', 'P', '4', 'V'), 11,
                                      (1920, 1080))

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                self.output.write(cv2.flip(cv2.flip(frame, 1), 0))
                return ret, cv2.flip(cv2.flip(frame, 1), 0)
            else:
                return ret, None
        else:
            return None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.output.release()

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    return
                self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class ImageDisplayer:
    def __init__(self):
        pass

    def one_object_display(self, array: np.ndarray, cur_obj: np.ndarray, des: str) -> None:
        cv2.rectangle(array, (int(cur_obj[0]), int(cur_obj[1])),
                      (int(cur_obj[2]), int(cur_obj[3])), (255, 255, 0), 2)
        cv2.putText(array, des, (int(cur_obj[0]), int(cur_obj[1])),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        return None

    def display_frame_with_labels(self, frame):
        cv2.namedWindow('monitor', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('monitor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('monitor', 0, 0)
        cv2.imshow("monitor", frame)  # отрисовка полученного кадра


def main():
    path_to_json_config = 'config.json'
    config_loader = ConfigLoader(path_to_json_config)
    config = config_loader.get_config()
    captor = ImageCaptor(config["video_source_main_camera"])
    captor_2 = ImageCaptor(config["video_source_second_camera"])
    displayer = ImageDisplayer()

    yolo_v8_class_obj = YOLOv8Model()

    translator_1 = config["to_russian_palm"]
    cur_model_conf_1 = config["palm_model_conf_v8"]
    weights_1 = config["palm_detection_model_path_v8"]

    translator_2 = config["to_russian_q_tools_1"]
    cur_model_conf_2 = config["drone_tools_3_1_model_conf_v8"]
    weights_2 = config["drone_tools_3_1_model_path_v8"]

    translator_3 = config["to_russian_q_details_1"]
    cur_model_conf_3 = config["drone_details_4_1_model_conf_v8"]
    weights_3 = config["drone_details_4_1_model_path_v8"]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(yolo_v8_class_obj.run_three_models(translator_1, weights_1, cur_model_conf_1,
                                                               translator_2, weights_2, cur_model_conf_2,
                                                               translator_3, weights_3, cur_model_conf_3,
                                                               captor, displayer))


if __name__ == '__main__':
    main()
