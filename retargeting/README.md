# Motion Reconstruction and Retargeting

In order to transfer the motion and appearance between the human actors, we first need to reconstruct the 3D motion of the actor performing actions and then adapt his motion to the shape of the target actor (retargeting). 

We provide retargeting example with data from the dataset (after the install steps).

## I - Installation and Setup

### Requirements
- Python 2.7
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

Then install TensorFlow either with GPU support:
```
pip install tensorflow-gpu==1.3.0
```
or without GPU support:
```
pip install tensorflow==1.3.0
```

For installing and configuring **DIRT** (a fast Differentiable Renderer for TensorFlow), please follow the instructions in [this link](https://github.com/pmh47/dirt). If you need to get chumpy, plese install from this repository:
https://github.com/mattloper/chumpy/tree/db6eaf8c93eb5ae571eb054575fb6ecec62

### Downloading SMPL Models

For running the motion reconstruction and retargeting, we need the SMPL body priors available on Multi-view SMPLify repository: please download the models from  [DCT_Basis](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/DCT_Basis) and [Prior](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/Prior). Finally, to download the **SMPL** human body models, please go to [this](http://smpl.is.tue.mpg.de) (male and female models) and [this](https://github.com/akanazawa/hmr) (gender neutral model) projects and register to get access to the downloads section. 


## II - Usage and Description

1. For running the motion reconstruction in a sequence of the provided dataset, containing pre-computed SPIN poses (for instance for the **box**), please run:
```
python motion_reconstruction.py --pose_path ../data/box/smpl_pose/ --model_type 1 --folder_pose_suffix _body.pkl

```

2. Then to adapt the reconstructed motion to a new actor (retargeting):
```
python retargeting.py --motion_path ../data/box/ --c_pose ../data/8-views/smpl_consensus_shape.pkl
```
