# 基于Detectron2 的 Mask RCNN

**Requirements**

- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. Install them together at [pytorch.org](https://pytorch.org/) to make sure of this
- OpenCV is optional but needed by demo and visualization

> ps：添加了绘图、类别评估代码具体见coco_evaluation.py分割线内代码段

## 1.环境

- Window环境：cuda10.2、VS2019
- Linux下源码编译安装方式需要 gcc & g++ ≥ 5.4，[ninja](https://ninja-build.org/)是非必要的，但是可以加速编译
- 官方安装文档：[Installation — detectron2 0.6 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

```powershell
（WSL Linux）
conda create -n detectron2 python=3.8
conda activate detectron2

pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

```

```powershell
（Windows）

"""
修改setup.py最后一行（已修改）
#cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
"""

"""
修改detectron2\layers\csrc\nms_rotated\nms_rotated_cuda.cu文件头几行（已修改）
// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
/*#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif*/
#include "box_iou_rotated/box_iou_rotated_utils.h"
"""

# 此外，如果系统环境变量设置了多个cuda环境、仅保留一个


conda create -n detectron2 python=3.8
conda activate detectron2

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

pip install setuptools==58.2.0

# 注意在setup.py的上一级执行
python -m pip install -e detectron2

pip install -r requirements.txt

```

## 2.训练自己的数据集
1. 准备好coco格式的数据集

```
|-data
  |-annotations
  |-test
  |-train
  |-val
```

2. 在train_net.py中注册数据集
```python
from detectron2.data.datasets import register_coco_instances
# 注册数据
register_coco_instances("dior_train", {}, "data/annotations/train.json", "data/train")
register_coco_instances("dior_val", {}, "data/annotations/val.json", "data/val")
register_coco_instances("dior_test", {}, "data/annotations/test.json", "data/test")
```

3. 在configs/Base-RCNN-FPN.yaml中修改注册的数据集名称
```
MODEL:
  ROI_HEADS:
    NUM_CLASSES: 20            # 数据集类别数量，如果类别标号从1开始需要在实际类别上加1（背景类）
    SCORE_THRESH_TEST: 0.0     # 置信度
DATASETS:
  TRAIN: ("dior_train",)       # 指定训练数据集名称
  TEST: ("dior_test",)         # 指定验证/测试数据集名称
  
```


## 3.分布式训练

```powershell
python train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 3

CUDA_VISIBLE_DEVICES=1,2 python train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 2
```


## 4.断点续训：

```powershell
python train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 3 --resume
```


## 5.测试：

```powershell
# 单卡
python train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --eval-only MODEL.WEIGHTS output/model_final.pth

# 多卡
python train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 3 --eval-only MODEL.WEIGHTS runs/train1/model_final.pth

# 多卡+conf+iou
python train_net.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 3 \
     --eval-only MODEL.WEIGHTS output/model_final.pth \
     TEST.SCORE_THRESH_TEST 0.5 \
     TEST.NMS_THRESH_TEST 0.5
```