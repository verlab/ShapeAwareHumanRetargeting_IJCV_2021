# use CUDA + OpenGL
FROM nvidia/cudagl:10.0-devel-ubuntu16.04

# install apt dependencies
RUN apt-get update && apt-get install -y \
	git \
	vim \
	wget \
	software-properties-common \
	curl

# install newest cmake version
RUN apt-get purge cmake && cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz && tar -xvf cmake-3.14.5.tar.gz
RUN cd ~/cmake-3.14.5 && ./bootstrap && make -j6 && make install

# install python3.7 and pip
RUN apt-add-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py | python

# set environment variables
ENV CUDA_BASE_VERSION=10.0
ENV CUDNN_VERSION=7.6.0.64

# setting up cudnn
RUN apt-get install -y --no-install-recommends \             
	libcudnn7=$(echo $CUDNN_VERSION)-1+cuda$(echo $CUDA_BASE_VERSION) \             
	libcudnn7-dev=$(echo $CUDNN_VERSION)-1+cuda$(echo $CUDA_BASE_VERSION) 
RUN apt-mark hold libcudnn7 && rm -rf /var/lib/apt/lists/*

ENV TENSORFLOW_VERSION=1.14.0

# install python dependencies
RUN python -m pip install tensorflow-gpu==$(echo $TENSORFLOW_VERSION)

# install dirt
ENV CUDAFLAGS='-DNDEBUG=1'
RUN cd ~ && git clone https://github.com/pmh47/dirt.git && \ 
 	python -m pip install dirt/

# run dirt test command
COPY retargeting/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

WORKDIR /app

COPY retargeting/ retargeting/
COPY data/ data/

WORKDIR /app/retargeting
CMD python motion_reconstruction.py --pose_path ../data/box/smpl_pose/ --model_type 1 --folder_pose_suffix _body.pkl && python retargeting.py --motion_path ../data/box/ --c_pose ../data/8-views/smpl_consensus_shape.pkl