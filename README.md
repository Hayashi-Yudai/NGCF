# Neural Graph Collaborative Filtering (NGCF)

Pytorch implementation of NGCF.

> Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019.Neural Graph Collaborative Filtering. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’19), July 21–25, 2019, Paris, France. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3331184.3331267

## Environment

- Python 3.8.10
- torch ^1.13.1

## How to train

```bash
poetry install
poetry run python main.py
```

If you want to change some parameters, edit `Config` in main.py.


## Experiment

Run the training with the same condition of the original paper.

- Epochs: 400
- Learning rate: 1e-4
- batch size: 1024
- embedding dimension: 64
- Number of embedding propagation layers: 3
- (Node|Message) Dropout ratio: 0.1
- L2 regulation strength: 1e-5
- Evaluation metrics to choose best model: Recall@20
- Early stopping: 50

### Dataset

- [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
- [Amazon-Book](http://jmcauley.ucsd.edu/data/amazon/)

Experiments were performed on both dataset in the 10-core setting as the original paper did.

### GPU Memory consumption

Used GPU: Tesla T4

- Gowalla: 1.5 GB
- Amazon-Book: 3.3 GB

### Result

Dataset | Recall@20 | NDCG@20 | Precision@20
------- | --------- | ------- | ------------
Gowalla | 0.1538    | 0.1295  | 0.0472
Amazon  | 0.0311 | 0.0239 | 0.0132

Recall@20 and NDCG@20 is almost as large as the original paper's result.

#### The number of embedding propagation layer vs scores

Dataset: Gowalla

model  | Recall@20 | NDCG@20 | Precision@20
------ | --------- | ------- | ------------
NGCF-1 | 0.1446    | 0.1233  | 0.0446
NGCF-2 | 0.1472    | 0.0816  | 0.0456
NGCF-3 | 0.1538    | 0.1295  | 0.0472
NGCF-4 | 0.1534    | 0.0849  | 0.0468
