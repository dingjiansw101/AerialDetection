
# Satellite Imagery Object Detection for Rotated Objects


Sedore is a object detection framework for rotated objects.
This codebase is based on
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [AerialDetection](https://github.com/dingjiansw101/AerialDetection)

Motivation of this repostiory is to provide a sustainable framework following recent updates of mmdetection.


![detected_results](results.jpg)
### Main Features
To adapt to object detection in aerial images, this repo has several unique and new features compared to the original [mmdetection](https://github.com/open-mmlab/mmdetection)
- **Support Oriented Object Detection**
    
    In aerial images, objects are usually annotated by oriented bounding box (OBB).
    To support oriented object detection, we implement OBB Head (OBBRoIHead and OBBDenseHead). 
    Also, we provide functions to transfer mask predictions to OBBs.

- **Cython Bbox Overlaps**
    
    Since one patch image with the size of 1024 &times; 1024 may contain over 1000 instances
     in [DOTA](https://captain-whu.github.io/DOTA/), which make the bbox overlaps memroy consuming.
     To avoid out of GPU memory, we calculate the bbox overlaps in cython. 
     The speed of cython version is close to the GPU version.

- **Rotation Augmentation**
    
    Since there are many orientation variations in aerial images, we implement the online rotation augmentation.
    
- **Rotated RoI Warping**

    Currently, we implement two types of rotated RoI Warping (Rotated RoI Align and Rotated Position Sensitive RoI Align).

   
## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

- Results are available in the [Model zoo](MODEL_ZOO.md).
- You can find the detailed configs in configs/DOTA.
- The trained models are available at [Google Drive](https://drive.google.com/drive/folders/1IsVLm7Yrwo18jcx0XjnCzFQQaf1WQEv8?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1aPeoPaQ0BJTuCsGt_DrdmQ).

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation.


    
## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing

We appreciate all contributions to improve benchmarks for object detection in aerial images. 

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)
