FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN apt-get install -y build-essential python2.7 python-pip python3.7 python3-pip python-tk 
RUN apt-get install -y libeigen3-dev libcgal-dev swig cmake git ffmpeg libsm6 libxext6 libxrender-dev libsm6 libxrender-dev

RUN python3.7 -m pip install -U pip
RUN python2.7 -m pip install -U pip

WORKDIR /app
COPY model-deformation-render/ model-deformation-render/
COPY data/ data/

RUN python2.7 -m pip install numpy==1.16.6
RUN python2.7 -m pip install -r model-deformation-render/requirements.txt
RUN python3.7 -m pip install -r model-deformation-render/requirements_inpainting.txt

WORKDIR /app/model-deformation-render/deformation
RUN swig -c++ -python deformation.i
RUN cmake .
RUN make
ENV PYTHONPATH "$PYTHONPATH:/app/model-deformation-render/deformation"

WORKDIR /app/model-deformation-render/normal
RUN swig -c++ -python compute_normals.i
RUN cmake .
RUN make
ENV PYTHONPATH "$PYTHONPATH:/app/model-deformation-render/normal"

COPY venv_p /app/venv_p
RUN echo $PYTHONPATH

WORKDIR /app/model-deformation-render
CMD python vd-render.py ../data/box/ ../data/8-views/ ../data/out/ --model_type 2