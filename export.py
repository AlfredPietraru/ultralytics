from ultralytics.engine.exporter import Exporter
from ultralytics.models import YOLO
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-s', '--size', help='set model size')
args = parser.parse_args()
model_path = args.model
size = int(args.size)


if (size == None):
    print("DIMENSIUNEA RETELEI NU A FOST SPECIFICATA")
    exit(1)
if (model_path == None):
    print("NU A FOST TRANMIS CA PARAMETRU CALEA CATRE MODEL")
    exit(1)
if (not os.path.isfile(model_path)) or (not model_path.endswith(".pt")):
    print("FISIERUL TRANSMIS CA PARAMETRU PENTRU MODEL NU ESTE VALID")
    exit(1)


exporter_args = {"format": "tflite", "int8": True, "imgsz": (size, size), "optimize": True, "data" : "ultralytics/cfg/datasets/coco8.yaml"}
exporter = Exporter(overrides=exporter_args)
model=YOLO(model_path)
model.export(format='tflite')
# model.export(format='tflite', int8=True, imgsz=(size, size), optimize=True, batch=1)
# results = exporter(model=model)#

