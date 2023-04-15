from datetime import datetime
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import List
import numpy as np

from data_loader import Dataset, Preprocessor, Sampler
from model import NGCF, MF
from evaluate import evaluate
from util import simple_logger


class Config:
    dataset_name: str = "gowalla"
    experiment_name: str = "test"
    batch_size: int = 1024
    batch_size_val: int = 1024
    epoch: int = 400
    emb_size: int = 64
    layers: List[int] = [64, 64, 64]

    lr: float = 1e-4
    decay: float = 1e-5
    early_stop_limit: int = 50
    node_dropout_flag: bool = True
    node_dropout: List[float] = [0.1]
    mess_dropout: List[float] = [0.1, 0.1, 0.1]  # Message dropout

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


class NGCFTrainer:
    def __init__(self, dataset: Dataset, config: Config):
        self.dataset = dataset
        self.norm_adj = Preprocessor(dataset).get_adjacency_matrix()
        self.model = NGCF(dataset.user_num, dataset.item_num, self.norm_adj, config).to(
            config.device
        )

        self.lr = config.lr

    def train(self, optimizer: torch.optim.Optimizer, sampler: Sampler) -> None:
        optimizer = optimizer(self.model.parameters(), lr=self.lr)

        best_recall = 0.0
        successive_non_improve_cnt = 0
        precisions, recalls, ndcgs = [], [], []
        simple_logger("Start training", __name__)
        for epoch in range(Config.epoch):
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = dataset.train_num // Config.batch_size + 1

            simple_logger(f"n_batch: {n_batch}", __name__)
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = sampler.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.model(
                    users, pos_items, neg_items, drop_flag=Config.node_dropout_flag
                )
                batch_loss, batch_mf_loss, batch_emb_loss = self.model.create_bpr_loss(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            if (epoch + 1) % 10 != 0:
                simple_logger(f"Epoch {epoch}", __name__)
                simple_logger(f"\tloss: {loss:.5f}", __name__)
                simple_logger(f"\tmf_loss: {mf_loss:.5f}", __name__)
                simple_logger(f"\temb_loss: {emb_loss:.5f}", __name__)

            simple_logger(f"Finished epoch: {epoch}", __name__)
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
            precisions.append(evals["precision"])
            recalls.append(evals["recall"])
            ndcgs.append(evals["ndcg"])
            if evals["recall"] > best_recall:
                successive_non_improve_cnt = 0
                simple_logger(
                    f"Recall@20 increased {best_recall} → {evals['recall']}",
                    __name__,
                )
                best_recall = evals["recall"]
                torch.save(
                    self.model.state_dict(),
                    f"{Config.dataset_name}_{Config.experiment_name}_best_model.pkl",
                )
                simple_logger("Saved model", __name__)
            else:
                successive_non_improve_cnt += 1
                simple_logger(f"Best precision remain {best_recall}", __name__)

            if successive_non_improve_cnt >= Config.early_stop_limit:
                simple_logger(f"Early stopped at epoch {epoch}", __name__)
                break

        np.save(f"{Config.dataset_name}_{Config.experiment_name}_precision", precisions)
        np.save(f"{Config.dataset_name}_{Config.experiment_name}_recall", recalls)
        np.save(f"{Config.dataset_name}_{Config.experiment_name}_ndcg", ndcgs)


class MFTrainer:
    def __init__(self, dataset: Dataset, config: Config):
        self.dataset = dataset
        self.model = MF(dataset.user_num, dataset.item_num, config).to(config.device)
        self.lr = config.lr

    def train(self, optimizer: torch.optim.Optimizer, sampler: Sampler) -> None:
        optimizer = optimizer(self.model.parameters(), lr=self.lr)

        best_recall = 0.0
        successive_non_improve_cnt = 0
        precisions, recalls, ndcgs = [], [], []
        for epoch in range(Config.epoch):
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = dataset.train_num // Config.batch_size + 1

            simple_logger(f"n_batch: {n_batch}", __name__)
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = sampler.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.model(
                    users, pos_items, neg_items, drop_flag=Config.node_dropout_flag
                )
                batch_loss = self.model.create_bpr_loss(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss

            if (epoch + 1) % 10 != 0:
                simple_logger(f"Epoch {epoch}", __name__)
                simple_logger(f"\tloss: {loss:.5f}", __name__)
                simple_logger(f"\tmf_loss: {mf_loss:.5f}", __name__)
                simple_logger(f"\temb_loss: {emb_loss:.5f}", __name__)

            simple_logger(f"Finished epoch: {epoch}", __name__)
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
            precisions.append(evals["precision"])
            recalls.append(evals["recall"])
            ndcgs.append(evals["ndcg"])
            if evals["recall"] > best_recall:
                successive_non_improve_cnt = 0
                simple_logger(
                    f"Recall@20 increased {best_recall} → {evals['recall']}",
                    __name__,
                )
                best_recall = evals["recall"]
                torch.save(
                    self.model.state_dict(),
                    f"{Config.dataset_name}_{Config.experiment_name}_best_model.pkl",
                )
                simple_logger("Saved model", __name__)
            else:
                successive_non_improve_cnt += 1
                simple_logger(f"Best precision remain {best_recall}", __name__)

            if successive_non_improve_cnt >= Config.early_stop_limit:
                simple_logger(f"Early stopped at epoch {epoch}", __name__)
                break

        np.save(f"{Config.dataset_name}_{Config.experiment_name}_precision", precisions)
        np.save(f"{Config.dataset_name}_{Config.experiment_name}_recall", recalls)
        np.save(f"{Config.dataset_name}_{Config.experiment_name}_ndcg", ndcgs)


if __name__ == "__main__":
    simple_logger(f"Use device {Config.device}", __name__)

    simple_logger("Initializing data loader", __name__)
    dataset = Dataset(path=f"./Data/{Config.dataset_name}")
    dataset.load()

    sampler = Sampler(dataset, batch_size=Config.batch_size)

    trainer = MFTrainer(dataset, Config())
    trainer.train(optim.RMSprop, sampler)
