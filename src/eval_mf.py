import numpy as np
import torch.nn as nn
from torch import LongTensor
from typing import Dict, List
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
    batch_size: int,
):
    test_users = list(test_data.keys())
    user_batches = np.array_split(test_users, len(test_users) // batch_size + 1)

    items = np.array(items)

    evaluate_vals = {"recall": [], "precision": [], "ndcg": []}
    simple_logger(f"Evaluate with {len(user_batches)} batches", __name__)
    for num, user_batch in tqdm(enumerate(user_batches)):
        # NOTE: Split in batch if OOM
        # user_embs, item_embs, _ = model(user_batch, items)
        ratings: np.ndarray = (
            model.rating(LongTensor(user_batch), LongTensor(items))
            .detach()
            .cpu()
            .numpy()
        )  # user_num x item_num

        for idx, user in enumerate(user_batch):
            user_ratings = ratings[idx]
            items_in_train = train_data[user]

            eval_idx = np.isin(items, items_in_train)

            # Create ranking with items not in training data
            items_for_eval = items[~eval_idx]
            user_ratings = user_ratings[~eval_idx]
            ranking_idx = np.argpartition(user_ratings, -k)[-k:]
            ranking = items_for_eval[ranking_idx]
            ground_truth = test_data[user]  # Item IDs
            gt_scores = [1.0] * len(ground_truth)
            scores = np.zeros_like(ranking)
            for idx, item in enumerate(ranking):
                if item in ground_truth:
                    scores[idx] = 1.0

            evaluate_vals["recall"].append(recall_at_k(ranking, ground_truth, k))
            evaluate_vals["precision"].append(precision_at_k(ranking, ground_truth, k))
            evaluate_vals["ndcg"].append(ndcg_at_k(scores, gt_scores, k))

    return {k: np.mean(v) for k, v in evaluate_vals.items()}


if __name__ == "__main__":
    import torch
    from mf import MatrixFactorization

    dataset = Dataset(path="./Data/gowalla")
    dataset.load()

    model = MatrixFactorization(dataset.user_num, dataset.item_num, 128)
    model.load_state_dict(torch.load("outputs/gowalla_MF_best.pkl"))

    print(
        evaluate(
            model,
            dataset.users,
            dataset.items,
            dataset.train_data,
            dataset.test_data,
            20,
            1024,
        )
    )
