FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3
LABEL maintainer="shimada"

# RUN apt-key del 3bf863cc
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip3 install pandas
RUN pip3 install pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python matplotlib
RUN pip3 install sympy
RUN pip3 install matplotlib tqdm

RUN mkdir /dataset