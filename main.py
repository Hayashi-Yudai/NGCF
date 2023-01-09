from datetime import datetime
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import List

from data_loader import Dataset, Preprocessor, Sampler
from model import NGCF
from evaluate import evaluate
from util import simple_logger


class Config:
    batch_size: int = 1024
    batch_size_val: int = 1024
    epoch: int = 400
    emb_size: int = 64
    layers: List[int] = [64, 64, 64]

    lr: float = 1e-4
    decay: float = 1e-5
    node_dropout_flag: bool = True
    node_dropout: List[float] = [0.1]
    mess_dropout: List[float] = [0.1, 0.1, 0.1]  # Message dropout

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    simple_logger(f"Use device {Config.device}", __name__)

    simple_logger("Initializing data loader", __name__)
    dataset = Dataset(path="./Data/gowalla")
    dataset.load()

    norm_adj = Preprocessor(dataset).get_adjacency_matrix()

    simple_logger("Initializing model", __name__)
    model = NGCF(dataset.user_num, dataset.item_num, norm_adj, Config).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    sampler = Sampler(dataset, batch_size=Config.batch_size)

    simple_logger("Start training", __name__)
    best_precision = 0.0
    precisions, recalls, ndcgs = [], [], []
    for epoch in range(Config.epoch):
        loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
        n_batch = dataset.train_num // Config.batch_size + 1

        simple_logger(f"n_batch: {n_batch}", __name__)
        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = sampler.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(
                users, pos_items, neg_items, drop_flag=Config.node_dropout_flag
            )
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(
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
            model,
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
        if evals["precision"] > best_precision:
            simple_logger(
                f"Precision@20 increased {best_precision} → {evals['precision']}",
                __name__,
            )
            best_precision = evals["precision"]
            torch.save(model.state_dict(), "best_model.pkl")
            simple_logger("Saved model", __name__)
        else:
            simple_logger(f"Best precision remain {best_precision}", __name__)

    np.save("precision", precisions)
    np.save("recall", recall)
    np.save("ndcg", ndcg)
