import numpy as np
import torch.nn as nn
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
        user_embs, item_embs, _ = model(user_batch, items, [], drop_flag=False)
        ratings: np.ndarray = (
            model.rating(user_embs, item_embs).detach().cpu().numpy()
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
    import time

    from model import NGCF
    from data_loader import Dataset, Preprocessor
    from main import Config

    dataset = Dataset(path="./Data/gowalla")
    dataset.load()

    norm_adj = Preprocessor(dataset).get_adjacency_matrix()
    model = NGCF(dataset.user_num, dataset.item_num, norm_adj, Config).to("cuda:0")
    model.load_state_dict(torch.load("best_model.pkl"))

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
