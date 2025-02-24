import torch 
import torch_npu 
import argparse
# import torch
from ultralytics import YOLO
import os
from torch_npu.contrib import transfer_to_npu

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument('--gpu', type=str, default="0", help='CUDA device ID')
    parser.add_argument('--data', type=str,  help='Path to the data configuration file',default="/home/xxf/xxf/code/yolov9/data/che_10.yaml")
    parser.add_argument('--project', type=str, default='runs/sb', help='Project name')
    parser.add_argument('--weight', type=str, default='', help='Project name') #"runs/10_sar/exp8/weights/epoch300.pt"

    args = parser.parse_args()
    return args
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# Load a model
model_path = args.weight
# model = YOLO(model_path)  # load an official model
model = YOLO(model_path)

checkpoint = torch.load(model_path, map_location='cpu')


# if 'epoch' in checkpoint:
#     print(f"Epoch: {checkpoint['epoch']}")
# else:
#     print("Epoch information not found in the checkpoint.") 

del checkpoint
print(model_path)
metrics = model.val(rect=False, data=args.data)  # no arguments needed, dataset and settings remembered
