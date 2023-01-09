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