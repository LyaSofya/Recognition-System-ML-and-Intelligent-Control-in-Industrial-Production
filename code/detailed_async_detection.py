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
        """Initialize YOLOv8Model class."""
        pass

    def load_model(self, path_to_model_weights, device):
        """
        Load a YOLO model with the specified weights and device.

        Args:
            path_to_model_weights (str): Path to the model weights.
            device (str): Device to load the model on ('cpu' or 'cuda').

        Returns:
            YOLO: The loaded YOLO model.
        """
        print(f'Using device: {device}')
        current_model = YOLO(path_to_model_weights).to(device=device)
        return current_model

    async def detect_objects(self, frame, model, current_model_conf, image_dislayer, labels_translator=None):
        """
        Perform object detection on a given frame and overlay the results.

        Args:
            frame (np.ndarray): The input image frame.
            model (YOLO): The YOLO model for detection.
            current_model_conf (float): Confidence threshold for detections.
            image_dislayer (ImageDisplayer): Object to display images.
            labels_translator (dict, optional): Dictionary to translate labels. Defaults to None.

        Returns:
            None
        """
        if labels_translator is not None:
            labels_translator = {int(k): v for k, v in labels_translator.items()}
        
        # Perform object detection
        details_nn_results = model(frame, imgsz=1920)
        
        for item in details_nn_results:
            boxes = item.boxes.cpu().numpy()  # Get bounding boxes on CPU as numpy array
            
            for obj in boxes.data:
                if obj[4] > current_model_conf:  # Check confidence threshold
                    if int(obj[5]) in labels_translator:  # Check if label needs to be translated
                        descr = labels_translator[int(obj[5])]
                        # Overlay the detection results on the frame
                        image_dislayer.one_object_display(frame, obj, str(descr + ' ' + str(round(obj[4], 2))))

    def get_gpu_info(self):
        """
        Get and print GPU information using nvidia-smi command.

        Returns:
            None
        """
        gpu_info = subprocess.check_output("nvidia-smi", shell=True)
        gpu_info = gpu_info.decode("utf-8")
        print(gpu_info)

    async def run_three_models(self, translator_1, weights_1, cur_model_conf_1,
                               translator_2, weights_2, cur_model_conf_2,
                               translator_3, weights_3, cur_model_conf_3,
                               captor, displayer):
        """
        Run three YOLO models asynchronously to process frames from the video stream.

        Args:
            translator_1 (dict): Label translator for the first model.
            weights_1 (str): Path to weights for the first model.
            cur_model_conf_1 (float): Confidence threshold for the first model.
            translator_2 (dict): Label translator for the second model.
            weights_2 (str): Path to weights for the second model.
            cur_model_conf_2 (float): Confidence threshold for the second model.
            translator_3 (dict): Label translator for the third model.
            weights_3 (str): Path to weights for the third model.
            cur_model_conf_3 (float): Confidence threshold for the third model.
            captor (ImageCaptor): Object to capture video frames.
            displayer (ImageDisplayer): Object to display processed frames.

        Returns:
            None
        """
        print(f"Torch cuda is available: {torch.cuda.is_available()}")

        start_time = time.time()
        fps_list = []
        
        # Check for available CUDA devices
        if torch.cuda.is_available():
            device1 = torch.device('cuda:0')
            device2 = torch.device('cuda:1')
        
        yolo_v8_class_obj = YOLOv8Model()
        yolo_v8_class_obj.get_gpu_info()
        
        # Load models on specified devices
        palm_model = yolo_v8_class_obj.load_model(weights_1, device1)
        drone_tools_model = yolo_v8_class_obj.load_model(weights_2, device2)
        drone_details_model = yolo_v8_class_obj.load_model(weights_3, device1)
        
        while True:
            time_1 = time.time()
            ret, frame = captor.get_frame()
            if frame is None:
                break

            # Perform detection using multiple models asynchronously
            await asyncio.gather(
                self.detect_objects(frame=frame,
                                    model=palm_model,
                                    current_model_conf=cur_model_conf_1,
                                    image_dislayer=displayer,
                                    labels_translator=translator_1),
                # Uncomment the following block to enable the second model
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

            # Display the frame with detection labels
            displayer.display_frame_with_labels(frame)

            elapsed_time = time.time() - time_1
            fps = 1 / elapsed_time
            print("FPS", fps)
            fps_list.append(fps)

            # Calculate average FPS every 60 seconds
            if time.time() - start_time >= 60:
                avg_fps = sum(fps_list) / len(fps_list)
                print("Average FPS over 1 minute: ", avg_fps)
                fps_list = []
                start_time = time.time()

            # Break the loop if 'q' key is pressed
            pressed_key = cv2.waitKey(1)
            if pressed_key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

class ConfigLoader:
    def __init__(self, file_path):
        """
        Initialize ConfigLoader class.

        Args:
            file_path (str): Path to the JSON configuration file.
        """
        self.file_path = file_path
        self.config = {}
        self.load_config()

    def load_config(self):
        """
        Load configuration from a JSON file.

        Returns:
            None
        """
        with open(self.file_path, 'r', encoding="UTF-8") as file:
            self.config = json.load(file)

    def get_config(self):
        """
        Get the loaded configuration.

        Returns:
            dict: The loaded configuration.
        """
        return self.config

class ImageCaptor(object):
    def __init__(self, video_source):
        """
        Initialize ImageCaptor class to capture video frames.

        Args:
            video_source (str): The video source (e.g., camera index or video file path).
        """
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        # Set video source width, height, and FPS
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.vid.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize video writer to save the output video
        self.output = cv2.VideoWriter(f'output_video_{video_source}.mp4',
                                      cv2.VideoWriter.fourcc('M', 'P', '4', 'V'), 11,
                                      (1920, 1080))

    def get_frame(self):
        """
        Capture a frame from the video source.

        Returns:
            tuple: (ret, frame) where ret is a boolean indicating success, and frame is the captured frame.
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Save the flipped frame to the output video
                self.output.write(cv2.flip(cv2.flip(frame, 1), 0))
                return ret, cv2.flip(cv2.flip(frame, 1), 0)
            else:
                return ret, None
        else:
            return None

    def __del__(self):
        """Release the video source and writer when the object is destroyed."""
        if self.vid.isOpened():
            self.vid.release()
            self.output.release()

class VideoCaptureAsync:
    def __init__(self, src=0):
        """
        Initialize VideoCaptureAsync class for asynchronous video capture.

        Args:
            src (int or str): The video source (e.g., camera index or video file path).
        """
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())


        self.thread.daemon = True

    def start(self):
        """
        Start the asynchronous video capture.

        Returns:
            VideoCaptureAsync: The VideoCaptureAsync object.
        """
        self.thread.start()
        return self

    def update(self):
        """Update the video capture by reading frames in a separate thread."""
        while self.running:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    return
                self.q.put(frame)

    def read(self):
        """
        Read a frame from the queue.

        Returns:
            np.ndarray: The next frame from the queue.
        """
        return self.q.get()

    def stop(self):
        """Stop the asynchronous capture and release resources."""
        self.running = False
        self.thread.join()
        self.cap.release()

class ImageDisplayer:
    def __init__(self):
        """Initialize ImageDisplayer class."""
        pass

    def one_object_display(self, array: np.ndarray, cur_obj: np.ndarray, des: str) -> None:
        """
        Draw bounding box and label on the frame for one detected object.

        Args:
            array (np.ndarray): The image frame.
            cur_obj (np.ndarray): The bounding box and other details of the object.
            des (str): Description to be displayed (e.g., label and confidence).

        Returns:
            None
        """
        cv2.rectangle(array, (int(cur_obj[0]), int(cur_obj[1])),
                      (int(cur_obj[2]), int(cur_obj[3])), (255, 255, 0), 2)
        cv2.putText(array, des, (int(cur_obj[0]), int(cur_obj[1])),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        return None

    def display_frame_with_labels(self, frame):
        """
        Display the frame with detection labels in full screen.

        Args:
            frame (np.ndarray): The image frame with detection labels.

        Returns:
            None
        """
        cv2.namedWindow('monitor', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('monitor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('monitor', 0, 0)
        cv2.imshow("monitor", frame)  # Display the processed frame

def main():
    """
    Main function to load configuration, initialize objects, and run the model.

    Returns:
        None
    """
    # Load configuration from JSON file
    path_to_json_config = 'config.json'
    config_loader = ConfigLoader(path_to_json_config)
    config = config_loader.get_config()
    
    # Initialize video captors and displayer
    captor = ImageCaptor(config["video_source_main_camera"])
    captor_2 = ImageCaptor(config["video_source_second_camera"])
    displayer = ImageDisplayer()

    yolo_v8_class_obj = YOLOv8Model()

    # Extract configuration details for each model
    translator_1 = config["to_russian_palm"]
    cur_model_conf_1 = config["palm_model_conf_v8"]
    weights_1 = config["palm_detection_model_path_v8"]

    translator_2 = config["to_russian_q_tools_1"]
    cur_model_conf_2 = config["drone_tools_3_1_model_conf_v8"]
    weights_2 = config["drone_tools_3_1_model_path_v8"]

    translator_3 = config["to_russian_q_details_1"]
    cur_model_conf_3
    weights_3 = config["drone_details_4_1_model_path_v8"]

    # Create an event loop and run the detection
    loop = asyncio.get_event_loop()
    loop.run_until_complete(yolo_v8_class_obj.run_three_models(translator_1, weights_1, cur_model_conf_1,
                                                               translator_2, weights_2, cur_model_conf_2,
                                                               translator_3, weights_3, cur_model_conf_3,
                                                               captor, displayer))

if __name__ == '__main__':
    main()
