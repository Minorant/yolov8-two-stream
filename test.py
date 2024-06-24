from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# Load a model
model = YOLO('/home/xxf/xxf/code/yolov8-two-stream/runs/xingren/exp2/weights/best.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category