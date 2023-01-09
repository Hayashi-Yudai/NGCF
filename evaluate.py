import heapq
import multiprocessing
import numpy as np
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm import tqdm

from data_loader import Dataset
from metrics import recall_at_k, precision_at_k, ndcg_at_k
from util import simple_logger


def evaluate(
    model: nn.Module,
    users: List[int],
    items: List[int],
    train_data: Dict[int, List[int]],
    test_data: Dict[int, List[int]],
    k: int,
):
    batch_size = 1024  # TODO: Move to config
    user_batches = np.array_split(users, len(users) // batch_size + 1)

    evaluate_vals = {"recall": [], "precision": [], "ndcg": []}
    for num, user_batch in enumerate(user_batches):
        simple_logger(f"Batch: {num+1}/{len(user_batches)}", __name__)
        # NOTE: Split in batch if OOM
        user_embs, item_embs, _ = model(user_batch, items, [], drop_flag=False)
        ratings = (
            model.rating(user_embs, item_embs).detach().cpu()
        )  # user_num x item_num

        for idx, user in tqdm(enumerate(user_batch)):
            user_ratings = ratings[idx]
            items_in_train = train_data[user]

            ranking_idx = np.argsort(-user_ratings)
            # Create ranking with items not in training data
            ranking = [
                items[idx] for idx in ranking_idx if items[idx] not in items_in_train
            ][:k]
            ground_truth = test_data[user]  # Item IDs
            gt_scores = [1.0] * len(ground_truth)
            scores = np.zeros_like(ranking)
            for idx, item in enumerate(ranking):
                if item in ground_truth:
                    scores[idx] = 1.0

            evaluate_vals["recall"].append(recall_at_k(ranking, ground_truth, k))
            evaluate_vals["precision"].append(precision_at_k(ranking, ground_truth, k))
            evaluate_vals["ndcg"].append(ndcg_at_k(scores, gt_scores, k))

        simple_logger(f"Recall: {np.mean(evaluate_vals['recall'])}", __name__)
        simple_logger(f"Precision: {np.mean(evaluate_vals['precision'])}", __name__)
        simple_logger(f"NDCG: {np.mean(evaluate_vals['ndcg'])}", __name__)
        break


if __name__ == "__main__":
    import torch

    from model import NGCF
    from data_loader import Dataset, Preprocessor
    from main import Config

    dataset = Dataset(path="./Data/gowalla")
    dataset.load()

    norm_adj = Preprocessor(dataset).get_adjacency_matrix()
    model = NGCF(dataset.user_num, dataset.item_num, norm_adj, Config).to("cuda:0")
    model.load_state_dict(torch.load("best_model.pkl"))

    evaluate(
        model, dataset.users, dataset.items, dataset.train_data, dataset.test_data, 10
    )
