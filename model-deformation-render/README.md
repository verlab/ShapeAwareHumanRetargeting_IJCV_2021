# Model Deformation and Rendering

## Table of Contents
  * [Install](#installation)
  * [Downloading the model](#downloading-the-model)
  * [Description](#description)

## I - Installation and Setup

Requirements: Python 2.7

### Linux Setup with virtualenv
```
virtualenv venv_d
source venv_d/bin/activate
pip install -U pip
deactivate
source venv_d/bin/activate
pip install -r requirements.txt
```

### Install SMPL
1. Go to the [SMPL project page](http://smpl.is.tue.mpg.de) and Sign In.
2. Go to the section Downloads, and get the 1.0.0 SMPL version for Python2.7
3. Edit the variable SMPL_FP on the ```config.py``` file with the location of the downloaded SMPL package


### Install Python Tkinter
```
sudo apt install python-tk
```

### Install CGAL

The Computational Geometry Algorithms Library
```
sudo apt install libeigen3-dev
sudo apt install libcgal-dev
```

### Install SWIG

SWIG is a software development tool that connects programs written in C and C++ with a variety of high-level programming languages used to link some dependencies in this project.
```
sudo apt install swig
```

### Parse the C/C++ interfaces to Python

1. Follow the procedures described on the ```deformation``` and ```normal``` folders.

###  Install Generative Image Inpainting
1. Create a virtual enviroment and install requirements for [Generative Image Inpainting](https://github.com/JiahuiYu/generative_inpainting).
```
virtualenv venv_p
source venv_p/bin/activate
pip install -U pip
deactivate
source venv_p/bin/activate
pip install -r requirements_inpainting.txt
```
2. Edit the file **run_inpaint.sh** with the correct environment path to activate it when needed.


### Downloading the male and female models

1. Go to the [SMPL project page](http://smpl.is.tue.mpg.de) and Sign In.
2. Go to the section Downloads, and get the 1.0.0 SMPL version for Python2.7.
3. Put the ```basicModel_f_lbs_10_207_0_v1.0.0.pkl``` and ```basicmodel_m_lbs_10_207_0_v1.0.0.pkl``` in the models/3D folder

### Downloading the neutral gender model

1. Download the gender neutral model from [HMR project page](http://https://github.com/akanazawa/hmr)
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```
2. Put the ```neutral_smpl_with_cocoplus_reg.pkl``` in the models/3D folder

## II - Usage and Description

After the install you can run the model deformation and rendering with the following command:
```
python vd-render.py ../data/box/ ../data/8-views/ ../data/out/ --model_type 2
```
