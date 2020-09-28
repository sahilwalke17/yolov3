


This repo contains Ultralytics inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Credit to Joseph Redmon for YOLO  https://pjreddie.com/darknet/yolo/.


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov3/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Notebook](https://github.com/ultralytics/yolov3/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) << highly recommended
* [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
* [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart)  ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov3?logo=docker)
* [A TensorRT Implementation of YOLOv3 and YOLOv4](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3-spp) 


## Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco2017.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt`.

**Plot Training:** `from utils import utils; utils.plot_results()`

<img src="https://user-images.githubusercontent.com/26833433/78175826-599d4800-7410-11ea-87d4-f629071838f6.png" width="900">


### Image Augmentation

`datasets.py` applies OpenCV-powered (https://opencv.org/) augmentation to the input image. We use a **mosaic dataloader** to increase image variability during training.

<img src="https://user-images.githubusercontent.com/26833433/80769557-6e015d00-8b02-11ea-9c4b-69310eb2b962.jpg" width="900">


### Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** preemptible [n1-standard-8](https://cloud.google.com/compute/docs/machine-types) (8 vCPUs, 30 GB memory)   
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.14/hr), T4 ($0.11/hr), V100 ($0.74/hr) CUDA with [Nvidia Apex](https://github.com/NVIDIA/apex) FP16/32    
**HDD:** 300 GB SSD  
**Dataset:** COCO train 2014 (117,263 images)  
**Model:** `yolov3-spp.cfg`  
**Command:**  `python3 train.py --data coco2017.data --img 416 --batch 32`

GPU | n | `--batch-size` | img/s | epoch<br>time | epoch<br>cost
--- |--- |--- |--- |--- |---
K80    |1| 32 x 2 | 11  | 175 min  | $0.41
T4     |1<br>2| 32 x 2<br>64 x 1 | 41<br>61 | 48 min<br>32 min | $0.09<br>$0.11
V100   |1<br>2| 32 x 2<br>64 x 1 | 122<br>**178** | 16 min<br>**11 min** | **$0.21**<br>$0.28
2080Ti |1<br>2| 32 x 2<br>64 x 1 | 81<br>140 | 24 min<br>14 min | -<br>-


## Inference

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8`

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067835-51d5b500-cc2f-11e9-982e-843f7f9a6ea2.jpg" width="500">

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067834-51d5b500-cc2f-11e9-9357-c485b159a20b.jpg" width="500">

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067833-51d5b500-cc2f-11e9-8208-6fe197809131.jpg" width="500">


## Pretrained Checkpoints

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)


## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'weights/yolov3-spp.weights'
```



