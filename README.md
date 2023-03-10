# GCN-layer-wise-sampling

This is the official implementation for "Calibrate and Debias Layer-wise Sampling for Graph Convolutional Networks" (TMLR 2023).

## Main idea

Multiple sampling-based methods have been developed for approximating and accelerating node embedding aggregation in graph convolutional networks (GCNs) training. 

Among them, a layer-wise approach recursively performs importance sampling to select neighbors jointly for existing nodes in each layer. 

This paper revisits the approach from a matrix approximation perspective, and identifies two issues in the existing layer-wise sampling methods: 

1. suboptimal sampling probabilities
2. estimation biases induced by sampling without replacement. 

To address these issues, we accordingly propose two remedies: 
1. a new principle for constructing sampling probabilities (implemented as `sketch_sampler()` in `./sketch/samplers.py`);
2. an efficient debiasing algorithm, which can be used for any WRS-based methods (weighted random sampling). 

To do an unbiased WRS, we can run the following code (within `./sketch/utils_new.py`), where `p` is the sampling probability vector and `m` is the sample size, a scalar.

```python
def estWRS_weights(p, m):
    n = len(p)
    wrs_index = np.random.choice(n, m, False, p)

    weights = np.zeros(m)
    p_sum = 0
    
    for i in range(m):
        
        alpha = n / (i + 1) / (n - i)
        weights[i] = (1-p_sum) / p[wrs_index[i]] * alpha
        weights[:i] = weights[:i] * (1 - alpha) + alpha
        p_sum += p[wrs_index[i]]

    return wrs_index, weights
```

Concrete details can be found in [our paper](https://openreview.net/forum?id=JyKNuoZGux).


## Installation

To prepare the conda environment for the code in this repo, the users can create the environment through
```sh
conda env create -f graph.yml
```

## Implementation for GCN tasks

The code for gcn tasks is partially adapted from [this repo](https://github.com/UCLA-DM/LADIES).

## Setup Data

- Download `data.zip` from [here](https://1drv.ms/u/s!AqsZ7ICy6kBI5Udq6aMEicMQNYrQ?e=zgJzSQ), and unzip it. There will be a `data` folder.
- Place the `data` folder under the root directory `./`.

## Run Code

The initialization directory is the root directory `./`.

```
sh scripts/run_gcn.sh
```


## Citation

If you find the repository helpful, please consider citing our papers:

```
@article{
chen2022calibrate,
title={Calibrate and Debias Layer-wise Sampling for Graph Convolutional Networks},
author={Yifan Chen and Tianning Xu and Dilek Hakkani-Tur and Di Jin and Yun Yang and Ruoqing Zhu},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=JyKNuoZGux}
}
```
