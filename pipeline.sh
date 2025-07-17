#! /bin/bash


# CLEAR_ENABLED=$1
YAML=$1
SIZE=1024

# DATASET_YAML=/home/alf/edapa/other/datasets/VisDrone/VisDrone.yaml
# DATASET_DETECT=/home/alf/edapa/other/datasets/VisDrone/VisDrone2019-DET-val/images

# DATASET_YAML=/home/alf/edapa/other/datasets/coco8/coco8.yaml
# DATASET_DETECT=/home/alf/edapa/other/datasets/coco8/images/val

# DATASET_YAML=VisDrone.yaml

# DATASET_YAML=/home/alf/edapa/other/datasets/VisDroneLonely/VisDroneLonely.yaml
# DATASET_DETECT=/home/alf/edapa/other/datasets/VisDroneLonely/VisDrone2019-DET-val/images
MODEL_NAME=$(basename $YAML .yaml)
SAVE_MODEL_PATH=${MODEL_NAME}_saved_model
MODEL=${MODEL_NAME}.pt

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv shell 3.9.17
source venv/bin/activate

if [ ! -d ${SAVE_MODEL_PATH} ]
then
    mkdir ${SAVE_MODEL_PATH}
fi


# python3 train.py \
# --data ultralytics/cfg/datasets/coco8.yaml \
# --yaml ${YAML}
# --model ${MODEL}



python3 export.py --size ${SIZE} --model ${MODEL}
# yolo export \
# imgsz=${SIZE} \
# model=${MODEL} \
# format=tflite \
# optimize=True \
# data=ultralytics/cfg/datasets/coco8.yaml
# int8=True


# edgetpu_compiler ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant.tflite
# cat ${MODEL_NAME}_full_integer_quant_edgetpu.log
# mv ${MODEL_NAME}.onnx ${MODEL_NAME}_full_integer_quant_edgetpu.log  ${MODEL_NAME}_full_integer_quant_edgetpu.tflite ${SAVE_MODEL_PATH}

# yolo detect val \
# batch=1 \
# imgsz=${SIZE} \
# model=$SAVE_MODEL_PATH/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
# source=$DATASET_DETECT \
# data=$DATASET_YAML \
# project=runs/detect \
# name=val_${MODEL_NAME}
 
mv runs/detect/train/events.out.tfevents.* runs/detect/train/results.csv ${SAVE_MODEL_PATH}
rm -rf runs/detect/train
tensorboard --logdir ${SAVE_MODEL_PATH}
