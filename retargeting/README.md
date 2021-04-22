# Motion Reconstrucion and Retargeting

## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Downloading the model](#downloading-the-model)

## Description
1. Run the motion recostruction with:
```
python motion_reconstruction.py --pose_path ../data/box/smpl_pose/ --model_type 1 --folder_pose_suffix _body.pkl

```

2. Run the retargeting with:
```
python retargeting.py --motion_path ../data/box/ --c_pose ../data/8-views/smpl_consensus_shape.pkl

```


## Installation

### Requirements
- Python 2.7

### Linux Setup with virtualenv
```
virtualenv venv_r
source venv_r/bin/activate
pip install -U pip
deactivate
source venv_r/bin/activate
pip install -r requirements.txt
```

### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==1.3.0
```
Without GPU:
```
pip install tensorflow==1.3.0
```

### Install DIRT

DIRT: a fast Differentiable Renderer for TensorFlow[this](https://github.com/pmh47/dirt)

### if you need to get chumpy 
https://github.com/mattloper/chumpy/tree/db6eaf8c93eb5ae571eb054575fb6ecec62

## Downloading the models

To download the body model priors got to [this](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/DCT_Basis) and [this](https://github.com/YinghaoHuang91/MuVS/tree/master/Data/Prior) 

To download the *SMPL* model go to [this](http://smpl.is.tue.mpg.de) (male and female models) and [this](https://github.com/akanazawa/hmr) (gender neutral model) project website and register to get access to the downloads section. 

