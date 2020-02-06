## Installation

### Requirements

- Linux
- Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
- PyTorch 1.1
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC: 4.9/5.3/5.4/7.3

### Install Aerialdetection

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n AerialDetection python=3.7 -y
source activate AerialDetection

conda install cython
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the AerialDetection repository.

```shell
git clone https://github.com/dingjiansw101/AerialDetection.git
cd AerialDetection
```

d. Compile cuda extensions.

```shell
./compile.sh
```

e. Install AerialDetection (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```

Note:

1. It is recommended that you run the step e each time you pull some updates from github. If there are some updates of the C/CUDA codes, you also need to run step d.
The git commit id will be written to the version number with step e, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, AerialDetection is installed on `dev` mode, any modifications to the code will take effect without installing it again.

### Install DOTA_devkit
```
    sudo apt-get install swig
    cd DOTA_devkit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```
### Notice
You can run `python(3) setup.py develop` or `pip install -e .` to install AerialDetection if you want to make modifications to it frequently.

If there are more than one AerialDetection on your machine, and you want to use them alternatively.
Please insert the following code to the main file
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
