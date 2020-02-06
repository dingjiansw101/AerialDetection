# Getting Started

This page provides basic tutorials about the usage of mmdetection.
For installation instructions, please see [INSTALL.md](INSTALL.md).



## Prepare DOTA dataset.
It is recommended to symlink the dataset root to `AerialDetection/data`.

Here, we give an example for single scale data preparation of DOTA-v1.0.

First, make sure your initial data are in the following structure.
```
data/dota
├── train
│   ├──images
│   └── labelTxt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```
Split the original images and create COCO format json. 
```
python DOTA_devkit/prepare_dota1.py --srcpath path_to_dota --dstpath path_to_split_1024
```
Then you will get data in the following structure
```
dota1_1024
├── test1024
│   ├── DOTA_test1024.json
│   └── images
└── trainval1024
     ├── DOTA_trainval1024.json
     └── images
```
For data preparation with data augmentation, refer to "DOTA_devkit/prepare_dota1_aug.py"

For data preparation of dota1.5, refer to "DOTA_devkit/prepare_dota1_5.py" and "DOTA_devkit/prepare_dota1_5_aug.py"


## Inference with pretrained models


### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/`.

1. Test Faster R-CNN.

```shell
python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py \
    work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth \ 
    --out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/results.pkl
```

2. Test Mask R-CNN with 4 GPUs.

```shell
./tools/dist_test.sh configs/DOTA/mask_rcnn_r50_fpn_1x_dota.py \
    work_dirs/mask_rcnn_r50_fpn_1x_dota/epoch_12.pth \
    4 --out work_dirs/mask_rcnn_r50_fpn_1x_dota/results.pkl 
```

3. Parse the results.pkl to the format needed for [DOTA evaluation](http://117.78.28.204:8001/)

For methods with only OBB Head, set the type OBB.
```
python tools/parse_results.py --config configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py --type OBB
```
For methods with both OBB and HBB Head, set the type HBBOBB.
```
python tools/parse_results.py --config configs/DOTA/faster_rcnn_h-obb_r50_fpn_1x_dota.py --type OBB
```
For methods with HBB and Mask Head, set the type Mask
```
python tools/parse_results.py --config configs/DOTA/mask_rcnn_r50_fpn_1x_dota.py --type Mask
```
For methods with only HBB Head, se the type HBB
```
python tools/parse_results.py --config configs/DOTA/faster_rcnn_r50_fpn_1x_dota.py --type HBB
```
### Demo of inference in a large size image.


```python
python demo_large_image.py
```


## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

**\*Important\***: The default learning rate in config files is for 8 GPUs.
If you use less or more than 8 GPUs, you need to set the learning rate proportional
to the GPU num, e.g., 0.01 for 4 GPUs and 0.04 for 16 GPUs.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (recommended): Perform evaluation at every k (default=1) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

### Train with multiple machines

If you run mmdetection on a cluster managed with [slurm](https://slurm.schedmd.com/), you can just use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [${GPUS}]
```

Here is an example of using 16 GPUs to train Mask R-CNN on the dev partition.

```shell
./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x.py /nfs/xxxx/mask_rcnn_r50_fpn_1x 16
```

You can check [slurm_train.sh](tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
pytorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility).
Usually it is slow if you do not have high speed networking like infiniband.


## How-to

### Use my own datasets

The simplest way is to convert your dataset to existing dataset formats (COCO or PASCAL VOC).

Here we show an example of adding a custom dataset of 5 classes, assuming it is also in COCO format.

In `mmdet/datasets/my_dataset.py`:

```python
from .coco import CocoDataset


class MyDataset(CocoDataset):

    CLASSES = ('a', 'b', 'c', 'd', 'e')
```

In `mmdet/datasets/__init__.py`:

```python
from .my_dataset import MyDataset
```

Then you can use `MyDataset` in config files, with the same API as CocoDataset.


It is also fine if you do not want to convert the annotation format to COCO or PASCAL format.
Actually, we define a simple annotation format and all existing datasets are
processed to be compatible with it, either online or offline.

The annotation of a dataset is a list of dict, each dict corresponds to an image.
There are 3 field `filename` (relative path), `width`, `height` for testing,
and an additional field `ann` for training. `ann` is also a dict containing at least 2 fields:
`bboxes` and `labels`, both of which are numpy arrays. Some datasets may provide
annotations like crowd/difficult/ignored bboxes, we use `bboxes_ignore` and `labels_ignore`
to cover them.

Here is an example.
```
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, float32> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, float32> (k, ) (optional field)
        }
    },
    ...
]
```

There are two ways to work with custom datasets.

- online conversion

  You can write a new Dataset class inherited from `CustomDataset`, and overwrite two methods
  `load_annotations(self, ann_file)` and `get_ann_info(self, idx)`,
  like [CocoDataset](mmdet/datasets/coco.py) and [VOCDataset](mmdet/datasets/voc.py).

- offline conversion

  You can convert the annotation format to the expected format above and save it to
  a pickle or json file, like [pascal_voc.py](tools/convert_datasets/pascal_voc.py).
  Then you can simply use `CustomDataset`.

### Develop new components

We basically categorize model components into 4 types.

- backbone: usually a FCN network to extract feature maps, e.g., ResNet, MobileNet.
- neck: the component between backbones and heads, e.g., FPN, PAFPN.
- head: the component for specific tasks, e.g., bbox prediction and mask prediction.
- roi extractor: the part for extracting RoI features from feature maps, e.g., RoI Align.

Here we show how to develop new components with an example of MobileNet.

1. Create a new file `mmdet/models/backbones/mobilenet.py`.

```python
import torch.nn as nn

from ..registry import BACKBONES


@BACKBONES.register
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(x):  # should return a tuple
        pass
```

2. Import the module in `mmdet/models/backbones/__init__.py`.

```python
from .mobilenet import MobileNet
```

3. Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

For more information on how it works, you can refer to [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) (TODO).
