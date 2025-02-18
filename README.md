<img src="https://crns.rnrt.tn/img/logo.svg">

# Welcome to the official implementation of "Cross-Attention Fusion of Visual and Geometric Features for Large Vocabulary Arabic Lipreading"
 ![image](assets/merged_image.png)

[![arXiv](https://img.shields.io/badge/arXiv-2208.11424-b31b1b.svg)](https://arxiv.org/abs/2402.11520) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<pre>
@article{daou2024cross,
  title={Cross-Attention Fusion of Visual and Geometric Features for Large Vocabulary Arabic Lipreading},
  author={Daou, Samar and Rekik, Ahmed and Ben-Hamadou, Achraf and Kallel, Abdelaziz},
  journal={arXiv preprint arXiv:2402.11520},
  year={2024}
}
</pre>

## Pre-trained model 
Pre-trained model can be downloaded following the link. You may save the model to ./checkpoints folder
- [Download link](https://drive.google.com/file/d/186P4fFTOsWhDqpML5lbaKilhPkVi-Yfq/view)
## LRW-AR Training and testing datasets can be downloaded following this link. You may save the datasets to ./data folder
- [Download link](https://osf.io/rz49x)

unzip the zip file in the ./data folder and then update your configuration yaml files accordingly (see below)

## Facial landmarks detection model
DLIB facial landmarks detection model can be downloaded following this link. You may save the model to ./data folder
- [Download link](https://drive.google.com/file/d/1HZW_xVIZySJ-5yFoE-nztfo4VxhDxGyT)
## Runners and configuration files
- "**run_generate_training_data.py**", configuration yaml file in "**config/config_generate_training_data.yaml**"
- "**run_training.py**", configuration yaml file in "**config/config_train.yaml**"
- "**run_validation.py**", configuration yaml file in "**config/config_validation.yaml**"

## Setup for Dev on local machine
This code base is tested only on Ubuntu 20.04 LTS, TitanV and RTX2080-ti NVIDIA GPUs.
- Install local environment and requirements
First install Anaconda3 then install the requirements as follows:

> **conda create -n crns---lrw-ar python=3.8**

- a new virtual environment is now created in **~/anaconda3/envs/crns---lrw-ar**
Now activate the virtual environment by running:

> **source activate crns---lrw-ar**

- In case you would like stop your venv **`conda deactivate`**

- To install dependencies, cd to the directory where requirements.txt is located and run the following command in your shell:

> **cat requirements.txt  | xargs -n 1 -L 1 pip3 install**

## Git pre-commit hooks
> if not already installed from the requirements.txt then first install pre-commit and black using these commands: **`pip3 install pre-commit`**
> and **`pip3 install black`**

> run **`pre-commit install`** to set up the git hook scripts
>
> You can also **`flake8 <YOURSCRIPT>.py`** to check if your python script is compliant with the project
>
> or directly fix your script using **`black <YOURSCRIPT>.py`**

## Known issues