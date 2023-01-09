import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List


def recall_at_k(ranking: List[int], ground_truth: List[int], k: int) -> float:
    return len(set(ranking[:k]) & set(ground_truth)) / len(set(ground_truth))


def precision_at_k(ranking: List[int], ground_truth: List[int], k: int) -> float:
    return len(set(ranking[:k]) & set(ground_truth)) / k


def hit_rate_at_k(ranking: List[int], ground_truth: List[int], k: int) -> float:
    if len(set(ranking[:k]) & set(ground_truth)) > 0:
        return 1
    else:
        return 0


def auc(ranking: List[int], ground_truth: List[int]) -> float:
    return roc_auc_score(y_true=ground_truth, y_score=ranking)


def dcg_at_k(scores: List[float], k: int, method: int = 1) -> float:
    scores = np.array(scores)
    if method == 0:
        return scores[0] + np.sum(scores[1:k] / np.log2(np.arange(2, k + 1)))
    if method == 1:
        return np.sum((2**scores[:k] - 1) / np.log2(np.arange(1, k + 1) + 1))
    else:
        raise ValueError("method must be 0 or 1")


def ndcg_at_k(
    scores: List[float], ground_truth_score: List[float], k: int, method: int = 1
) -> float:
    if len(ground_truth_score) < k:
        ground_truth_score = list(ground_truth_score) + [0.0] * (
            k - len(ground_truth_score)
        )
    dcg_max = dcg_at_k(ground_truth_score, k, method)
    if dcg_max == 0:
        return 0.0

    return dcg_at_k(scores, k, method) / dcg_max


if __name__ == "__main__":
    ranking = [5, 2, 1]
    gt = [5, 3, 2]

    print(ndcg_at_k(ranking, gt, 3, method=1))
