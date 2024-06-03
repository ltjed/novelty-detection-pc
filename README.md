# Novelty detection with Predictive Coding Networks

## 1. Description
This repository contains code to perform experiments with predictive coding networks (PCNs) in novelty detection tasks. We investigate both hierarchical PCNs (hPCNs) and recurrent PCNs (rPCNs).

The preprint associated with the code repository can be found [here]().

## 2. Installation
To run the code, you should first install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html), 
and then clone this repository to your local machine.

Once these are installed and cloned, you can simply use the appropriate `.yml` file to create a conda environment. 
For Ubuntu or Mac OS, open a terminal, go to the repository directory; for Windows, open the Anaconda Prompt, and then enter:

0. `git clone https://github.com/ltjed/novelty-detection-pc.git`
1. `cd novelty-detection-pc`
2. `conda env create -f environment.yml`  
3. `conda activate cov-env`
4. `pip install -e .`  

## 3. Usage
Once the above is done, you can select one of the `.py` or `.ipynb` files to run experiments associated with a particular model.

For example, to obtain results with recurrent PCNs simply run `python recurrent_PCN.py`. You can play with the hyperparameters to see the performances.

Specifically:
- To obtain results with Figure 4 in the manuscript, run `recurrent_PCN.py` and `hPCN.py`;
- To obtain results with Figure 5 in the manuscript, run `recurrent_PCN.py` and `hopfield.py`;
- To obtain results with Figure 6 in the manuscript, run `theory.ipynb`;
- To obtain results with Figure 7 in the manuscript, run `locallyy_connected_hPCN.py`;

A directory named `results` will then be created to store all the data and figures collected from the experiments.

## 4. Contact
For any inquiries or questions regarding the project, please feel free to contact Ed Tianjin Li at <tianjin.li@hertford.ox.ac.uk> or Mufeng Tang at <mufeng.tang@ndcn.ox.ac.uk>.


