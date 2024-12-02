# Path Direction Estimation

Training scripts for estimating path direction

## Installation

Basically, it runs on Docker.

```bash
make docker-build
```

## Usage

All training scripts should be run on a Docker container.  
When working in a container, execute the following Make command.

```bash
# When you want to see how a script works
make docker-bash
# When you want to mount a data directory for training
make docker-bash-connect-dataset
```
  
Manipulate Makefile variables as necessary
- GPU_NUM
  - The number of the GPU you want to use
- DATASET_PATH
  - Dataset Path
  
There are also tools for formatting datasets that can be used to create training data.  
- path_dataset/generate.py
  - Create a path direction data set from a combination of coordinates
- result
  - Scripts are available to visualize the training results.All of the tools are designed to create histograms.

## Directory
For each directory
- data
  - Module for loading training data
- log
  - Directory to save loss or accuracy during training
- model
  - path direction estimation model
- settings
  - Module for handling TensorFlow (the GPU to be used is configured here)
- tool
  - Directory of tools for formatting CSV and expanding data
- weight
  - Directory for storing model weight data
