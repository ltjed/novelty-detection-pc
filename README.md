# Novelty detection with Predictive Coding Networks (PCNs)

## 1. Description
This repository contains code to perform experiments with recurrent predictive coding (recPCN) or hierarchical predictive coding network (hPCN) on various novelty detection tasks.

The preprint associated with the code repository can be found [here]().

## 2. Installation
To run the code, you should first install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html) (preferably the latter), 
and then clone this repository to your local machine.

Once these are installed and cloned, you can simply use the appropriate `.yml` file to create a conda environment. 
For Ubuntu or Mac OS, open a terminal, go to the repository directory; for Windows, open the Anaconda Prompt, and then enter:

1. `conda env create -f environment.yml`  
2. `conda activate cov-env`
3. `pip install -e .`  

## 3. Use
Once the above is done, you can select one of the .ipynb files in the root directory and use 'run all cells' to run the code to produce the figures in the preprint.

A directory named `results` will then be created to store all the data and figures collected from the experiments.
