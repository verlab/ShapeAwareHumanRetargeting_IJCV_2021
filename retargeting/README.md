# Motion Reconstruction and Retargeting

In order to transfer the motion and appearance between the human actors, we first need to reconstruct the 3D motion of the actor performing actions and then adapt his motion to the shape of the target actor (retargeting). 

We provide retargeting example with data from the dataset (after the install steps).

## I - Installation and Setup

### Requirements
- Python 3.5+ (Tested with Python 3.5.2 and Python 3.7.3)
- TensorFlow
- Differentiable Renderer (DIRT)

Please follow these steps for installing the requirements and setup with python virtualenv:
```
virtualenv venv_r
source venv_r/bin/activate
pip install -U pip
deactivate
source venv_r/bin/activate
pip install -r requirements.txt
```

Then install TensorFlow either with GPU support (highly recommended):
```
pip install tensorflow-gpu==1.15.0
```
or without GPU support:
```
pip install tensorflow==1.15.0
```

For installing and configuring **DIRT** (a fast Differentiable Renderer for TensorFlow), please follow the instructions in [this link](https://github.com/pmh47/dirt). Before doing DIRT setup, is necessary to install ```libnvinfer``` (Instructions for CUDA 10.1).

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1
```

### Downloading SMPL Models

For running the motion reconstruction and retargeting, we need the SMPL body priors available on Multi-view SMPLify repository: please download the models from  [DCT_Basis](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/DCT_Basis) and [Prior](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/Prior) and put them under the folder ```models```. Finally, to download the **SMPL** human body models, please go to [this page](http://smpl.is.tue.mpg.de), sign in, download the male and female basic models from version 1.0.0 (10 shape PCs) and put them into the ```models``` folder. To get the gender neutral model go to [this page](https://github.com/akanazawa/hmr) (gender neutral model), download the models as instructed and put the file ```neutral_smpl_with_cocoplus_reg.pkl``` under the ```models``` folder.


## II - Usage and Description

1. For running the motion reconstruction in a sequence of the provided dataset, containing pre-computed SPIN poses (for instance for the **box**), please run:
```
python motion_reconstruction.py --pose_path ../data/box/smpl_pose/ --model_type 1 --folder_pose_suffix _body.pkl

```

2. Then to adapt the reconstructed motion to a new actor (retargeting):
```
python retargeting.py --motion_path ../data/box/ --c_pose ../data/8-views/smpl_consensus_shape.pkl
```
