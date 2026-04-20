# Query-Conditioned Multi-Token Attention for Re-Ranking in Segmentation-Free KWS

This repository contains the implementation of our proposed method for improving re-ranking performance in segmentation-free keyword spotting (KWS) using query-conditioned multi-token attention.

## Overview of the proposed architecture
![Overview of the proposed architecture](/architecture-overview2.png)

## Requirements
To run the code, install the following python packages:
```
pytorch, torchvision, numpy, polars, editdistance, evaluate, tqdm, transformers, sentence-transformers
```

## Data Preparation
The main script operates on precomputed KWS results, similar to [kws-semantic-reranking](https://github.com/stevepapazis/kws-semantic-reranking).
Please refer to the instructions in that repository to complete the data setup.

## Citation
If you use this codebase or build upon the ideas presented in our work, please consider citing our paper. Citation details will be added once available.
