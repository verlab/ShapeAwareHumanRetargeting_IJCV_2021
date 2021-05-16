# Motion Reconstruction and Retargeting

In order to transfer the motion and appearance between the human actors, we first need to reconstruct the 3D motion of the actor performing actions and then adapt his motion to the shape of the target actor (retargeting). 

We provide retargeting example with data from the dataset (after the install steps).

## I - Setup models and files

### Downloading the male and female models

1. Go to the [SMPL project page](http://smpl.is.tue.mpg.de) and Sign In.
2. Go to the section Downloads, and get the 1.0.0 SMPL version for Python2.7.
3. Put the ```basicModel_f_lbs_10_207_0_v1.0.0.pkl``` and ```basicmodel_m_lbs_10_207_0_v1.0.0.pkl``` in the models/ folder

### Downloading the neutral gender model

1. Download the gender neutral model from [HMR project page](http://https://github.com/akanazawa/hmr)
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```
2. Put the ```neutral_smpl_with_cocoplus_reg.pkl``` in the models/ folder

### Downloading the SMPL body priors

1. Go to [this repository](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/)
2. Download the ```DCT_Basis``` and ```Prior``` and put them in the folder ```models``` 

## II - Installation

### Case 1 - Run with Singularity

- Install [Singularity](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps)
- Build the image: ``` sudo singularity build <image-name>.sif singularity/retargeting.def ```
- Run the image with: ``` sudo singularity run -B <bind-dir> --nv <image-name>.sif ```

### Case 2 - Run with Docker

- Install [Docker](https://docs.docker.com/engine/install/)
- Build the image: ``` docker build -t <image-name> -f docker/Dockerfile ../```
- Run the image: ``` docker run --runtime=nvidia <image-name>:latest ```

### Case 3 - Run with virtualenv

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

## III - Usage and Description

1. For running the motion reconstruction in a sequence of the provided dataset, containing pre-computed SPIN poses (for instance for the **box**), please run:
```
python motion_reconstruction.py --pose_path ../data/box/smpl_pose/ --model_type 1 --folder_pose_suffix _body.pkl

```

2. Then to adapt the reconstructed motion to a new actor (retargeting):
```
python retargeting.py --motion_path ../data/box/ --c_pose ../data/8-views/smpl_consensus_shape.pkl
```
