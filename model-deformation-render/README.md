# Model Deformation and Rendering

## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Downloading the model](#downloading-the-model)

## Description
1. Run the model deformation and rendering:
```
python vd-render.py ../data/box/ ../data/8-views/ ../data/out/ --model_type 2

```

## Installation

### Requirements
- Python 2.7

### Linux Setup with virtualenv
```
virtualenv venv_d
source venv_d/bin/activate
pip install -U pip
deactivate
source venv_d/bin/activate
pip install -r requirements.txt
```

### if you need to get chumpy 
https://github.com/mattloper/chumpy/tree/db6eaf8c93eb5ae571eb054575fb6ecec62

## Install SMPL

1. Please follow the procedure described on the SMPL corresponding websites [this](http://smpl.is.tue.mpg.de).
2. Edit the config.py file


### Install CGAL

The Computational Geometry Algorithms Library [this](https://www.cgal.org/download/linux.html)

```

### Install SWIG

SWIG is a software development tool that connects programs written in C and C++ with a variety of high-level programming languages. [this](http://www.swig.org)


###  Parse the C/C++ interfaces to Python

1. See deformation folder read
2. See normal folder read

###  Install Generative Image Inpainting

1. Generative Image Inpainting [this](https://github.com/JiahuiYu/generative_inpainting)
2. Edit run_inpaint.sh file with correct configurition to the inpaiting



## Downloading the models

To download the *SMPL* model to models/3D/. Go to [this](http://smpl.is.tue.mpg.de) (male and female models) and [this](https://github.com/akanazawa/hmr) (gender neutral model) project website and register to get access to the downloads section. 

