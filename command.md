nvidia-docker run -it --shm-size=16gb --cpus=16 -e NVIDIA_VISIBLE_DEVICES=0,1 -v $(pwd):/workspace -v input:/input pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime /bin/bash

apt update
apt install git -y
pip install opencv-python pandas fastai albumentations sklearn deepflash2
pip install git+https://github.com/qubvel/segmentation_models.pytorch
