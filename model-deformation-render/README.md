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
For downloading chumpy:
https://github.com/mattloper/chumpy/tree/db6eaf8c93eb5ae571eb054575fb6ecec62

### Install SMPL

1. Please follow the procedure described on the [SMPL project page](http://smpl.is.tue.mpg.de).
2. Edit the config.py file


### Install CGAL

The Computational Geometry Algorithms Library [this](https://www.cgal.org/download/linux.html)


### Install SWIG

SWIG is a software development tool that connects programs written in C and C++ with a variety of high-level programming languages. Please see the instructions in this [project page](http://www.swig.org)


### Parse the C/C++ interfaces to Python

1. See deformation folder read
2. See normal folder read

###  Install Generative Image Inpainting

1. Generative Image Inpainting [this](https://github.com/JiahuiYu/generative_inpainting)
2. Edit the file **run_inpaint.sh** with the correct configuration to the inpainting


### Downloading the models

To download the **SMPL** models and place then into models/3D/. Please follow the instructions given in [the SMPL page](http://smpl.is.tue.mpg.de) (male and female models) and [HMR project](https://github.com/akanazawa/hmr) (gender neutral model), and register to get access to the downloads section.

## II - Usage and Description

After the install you can run the model deformation and rendering with the following command:
```
python vd-render.py ../data/box/ ../data/8-views/ ../data/out/ --model_type 2

