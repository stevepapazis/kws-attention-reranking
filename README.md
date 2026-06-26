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
```
Giotis, A.P.; Papazis S; Nikou, C.
Query-Conditioned Multi-Token Attention for Re-Ranking in Segmentation-Free KWS.
2026 14th International Workshop on Biometrics and Forensics (IWBF), 1-6. https://doi.org/10.1109/IWBF68042.2026.11558149
```

```bibtex
@inproceedings{Giotis2026AttentionReRanking,
  author    = {Giotis, Angelos P. and Papazis, Stergios and Nikou, Christophoros},
  booktitle = {2026 14th International Workshop on Biometrics and Forensics (IWBF)}, 
  title     = {Query-Conditioned Multi-Token Attention for Re-Ranking in Segmentation-Free KWS}, 
  year      = {2026},
  volume    = {},
  number    = {},
  pages     = {1-6},
  doi       = {10.1109/IWBF68042.2026.11558149}
}
```
