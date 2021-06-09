# Install
```sh
apt update -y;apt install git -y;pip install requirement.txt
```

# Create a docker container
```sh
nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0,1 -v $(pwd):/workspace -v $(pwd)/input:/input pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime /bin/bash
```

# Prepare images and masks files
```sh
python 1024x1024.py > 1024x1024.txt 2>&1 unzip -qo /input/masks.zip -d /input/512x512-reduce-2/masks_v5;unzip -qo /input/train.zip -d /input/512x512-reduce-2/train_v5 mkdir -p input/512x512-reduce-2 mv train masks input/512x512-reduce-2/
```
