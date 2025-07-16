#! /bin/bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
yolo settings tensorboard=True
python3 visdrone_download.py
cp models/yolov5UPDATED.pt .
python3 train.py --model yolov5UPDATED.pt --data VisDrone.yaml

