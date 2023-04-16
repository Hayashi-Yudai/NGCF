from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from typing import List
import numpy as np

from data_loader import Dataset, Preprocessor, Sampler
from model import NGCF, MF
from losses import BPRLoss
from evaluate import evaluate
from util import simple_logger


class Config:
    dataset_name: str = "gowalla"
    experiment_name: str = "ngcf"
    batch_size: int = 1024
    batch_size_val: int = 1024
    epoch: int = 400
    emb_size: int = 64
    layers: List[int] = [64, 64, 64]

    lr: float = 1e-4
    decay: float = 1e-5
    early_stop_limit: int = 50
    node_dropout_flag: bool = True
    node_dropout: List[float] = [0.0]
    mess_dropout: List[float] = [0.1, 0.1, 0.1]  # Message dropout

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


class NGCFTrainer:
    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: Config,
    ):
        self.dataset = dataset
        self.norm_adj = Preprocessor(dataset).get_adjacency_matrix()
        self.sampler = sampler
        self.model = NGCF(dataset.user_num, dataset.item_num, self.norm_adj, config).to(
            config.device
        )
        self.optimizer = optimizer(self.model.parameters(), lr=config.lr)
        self.loss_fn = loss_fn

        self.precisions: List[float] = []
        self.recalls: List[float] = []
        self.ndcgs: List[float] = []
        self.best_recall = 0.0
        self.successive_non_improve_cnt = 0

    def fit(self) -> None:
        simple_logger("Start training", __name__)
        for epoch in range(Config.epoch):
            loss = 0.0
            n_batch = dataset.train_num // Config.batch_size + 1

            simple_logger(f"n_batch: {n_batch}", __name__)
            self.model.train()
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = self.sampler.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.model(
                    users, pos_items, neg_items, drop_flag=Config.node_dropout_flag
                )
                batch_loss = self.loss_fn(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                loss += batch_loss

            simple_logger(f"Epoch {epoch}", __name__)
            simple_logger(f"\tloss: {loss:.5f}", __name__)

            self.model.eval()
            self.validate()

            if self.successive_non_improve_cnt >= Config.early_stop_limit:
                simple_logger("Early stopped", __name__)

        self.save_history()

    def validate(self):
        simple_logger("Start evaluation", __name__)
        evals = evaluate(
            self.model,
            dataset.users,
            dataset.items,
            dataset.train_data,
            dataset.test_data,
            k=20,
            batch_size=Config.batch_size_val,
        )
        self.precisions.append(evals["precision"])
        self.recalls.append(evals["recall"])
        self.ndcgs.append(evals["ndcg"])
        if evals["recall"] > self.best_recall:
            successive_non_improve_cnt = 0
            simple_logger(
                f"Recall@20 increased {self.best_recall} → {evals['recall']}",
                __name__,
            )
            self.best_recall = evals["recall"]
            torch.save(
                self.model.state_dict(),
                f"./outputs/{Config.dataset_name}_{Config.experiment_name}_best_model.pkl",
            )
            simple_logger("Saved model", __name__)
        else:
            self.successive_non_improve_cnt += 1
            simple_logger(f"Best precision remain {self.best_recall}", __name__)

    def save_history(self):
        np.save(
            f"./outputs/{Config.dataset_name}_{Config.experiment_name}_precision",
            self.precisions,
        )
        np.save(
            f"./outputs/{Config.dataset_name}_{Config.experiment_name}_recall",
            self.recalls,
        )
        np.save(
            f"./outputs/{Config.dataset_name}_{Config.experiment_name}_ndcg", self.ndcgs
        )


class MFTrainer:
    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: Config,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.model = MF(dataset.user_num, dataset.item_num, config).to(config.device)
        self.optimizer = optimizer(self.model.parameters(), lr=config.lr)
        self.loss_fn = loss_fn

        self.precisions: List[float] = []
        self.recalls: List[float] = []
        self.ndcgs: List[float] = []
        self.best_recall = 0.0
        self.successive_non_improve_cnt = 0

    def fit(self) -> None:
        for epoch in range(Config.epoch):
            loss = 0.0
            n_batch = dataset.train_num // Config.batch_size + 1

            simple_logger(f"n_batch: {n_batch}", __name__)
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = self.sampler.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.model(
                    users, pos_items, neg_items, False
                )
                batch_loss = self.loss_fn(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                loss += batch_loss

            if (epoch + 1) % 10 != 0:
                simple_logger(f"Epoch {epoch}", __name__)
                simple_logger(f"\tloss: {loss:.5f}", __name__)

            self.validate()

            if self.successive_non_improve_cnt >= Config.early_stop_limit:
                simple_logger(f"Early stopped at epoch {epoch}", __name__)
                break

        self.save_history()

    def validate(self):
        simple_logger("Start evaluation", __name__)
        evals = evaluate(
            self.model,
            dataset.users,
            dataset.items,
            dataset.train_data,
            dataset.test_data,
            k=20,
            batch_size=Config.batch_size_val,
        )
        self.precisions.append(evals["precision"])
        self.recalls.append(evals["recall"])
        self.ndcgs.append(evals["ndcg"])
        if evals["recall"] > self.best_recall:
            successive_non_improve_cnt = 0
            simple_logger(
                f"Recall@20 increased {self.best_recall} → {evals['recall']}",
                __name__,
            )
            self.best_recall = evals["recall"]
            torch.save(
                self.model.state_dict(),
                f"./outputs/{Config.dataset_name}_{Config.experiment_name}_best_model.pkl",
            )
            simple_logger("Saved model", __name__)
        else:
            self.successive_non_improve_cnt += 1
            simple_logger(f"Best precision remain {self.best_recall}", __name__)

    def save_history(self):
        np.save(
            f"./outputs/{Config.dataset_name}_{Config.experiment_name}_precision",
            self.precisions,
        )
        np.save(
            f"./outputs/{Config.dataset_name}_{Config.experiment_name}_recall",
            self.recalls,
        )
        np.save(
            f"./outputs/{Config.dataset_name}_{Config.experiment_name}_ndcg", self.ndcgs
        )


if __name__ == "__main__":
    simple_logger(f"Use device {Config.device}", __name__)

    simple_logger("Initializing data loader", __name__)
    dataset = Dataset(path=f"./Data/{Config.dataset_name}")
    dataset.load()

    sampler = Sampler(dataset, batch_size=Config.batch_size)
    optimizer = optim.Adam
    loss_fn = BPRLoss(regularize=Config.decay)

    trainer = MFTrainer(dataset, sampler, optimizer, loss_fn, Config())
    trainer.fit()
