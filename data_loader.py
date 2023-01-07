import numpy as np
import pathlib
import random
from scipy import sparse as sp
from typing import Dict, List, Tuple

from util import simple_logger


class Dataset:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)

        self.users: List[int] = []
        self.items: List[int] = []
        self.train_num: int = 0
        self.test_num: int = 0

        self.train_data: Dict[int, List[int]] = {}
        self.test_data: Dict[int, List[int]] = {}

    @property
    def user_num(self) -> int:
        return max(self.users) + 1

    @property
    def item_num(self) -> int:
        return max(self.items) + 1

    def load(self):
        self._load_train(train_file=self.path / "train.txt")
        self._load_test(test_file=self.path / "test.txt")

        simple_logger(f"n_users: {len(self.users)}", __name__)
        simple_logger(f"n_items: {len(self.items)}", __name__)
        simple_logger(f"train dataset: {self.train_num}", __name__)
        simple_logger(f"test dataset: {self.test_num}", __name__)

    def _load_train(self, train_file: pathlib.Path):
        simple_logger("Loading train data", __name__)
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) == 0:
                    continue

                line = line.strip("\n").split(" ")
                items = [int(i) for i in line[1:]]
                uid = int(line[0])

                self.users.append(uid)
                self.items += items
                self.train_num += len(items)

                self.train_data[uid] = items

            self.items = list(set(self.items))

    def _load_test(self, test_file: pathlib.Path):
        simple_logger("Loading test data", __name__)
        with open(test_file) as f:
            for line in f.readlines():
                if len(line) == 0:
                    continue

                try:
                    line = line.strip("\n").split(" ")
                    uid = int(line[0])
                    items = [int(i) for i in line[1:]]
                except Exception:
                    continue
                self.test_num += len(items)

                self.test_data[uid] = items


class Preprocessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.path: pathlib.Path = self.dataset.path

        self.R = sp.dok_matrix((dataset.user_num, dataset.item_num), dtype=np.float32)
        for user, items in self.dataset.train_data.items():
            for item in items:
                self.R[user, item] = 1.0

    def get_adjacency_matrix(self) -> sp.csr_matrix:
        if (self.path / "s_norm_adj_mat.npz").exists():
            norm_adj_mat = sp.load_npz(self.path / "s_norm_adj_mat.npz")
        else:
            simple_logger("Adjacency matrix does not exist, creating...", __name__)
            norm_adj_mat = self._create_adjacency_matrix()
            sp.save_npz(self.path / "s_norm_adj_mat.npz", norm_adj_mat)

        simple_logger(f"Loaded adjacency matrix shape: {norm_adj_mat.shape}", __name__)

        return norm_adj_mat

    def _create_adjacency_matrix(self) -> sp.csr_matrix:
        matrix = sp.dok_matrix(
            (
                self.dataset.user_num + self.dataset.item_num,
                self.dataset.user_num + self.dataset.item_num,
            ),
            dtype=np.float32,
        ).tolil()  # Convert to list-to-list format
        R = self.R.tolil()

        matrix[: self.dataset.user_num, self.dataset.user_num :] = R
        matrix[self.dataset.user_num :, : self.dataset.user_num] = R.T

        rowsum = np.array(matrix.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)

        norm_matrix = d_mat_inv.dot(matrix)

        return norm_matrix.tocsr()


class Sampler:
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def sample(self) -> Tuple[List[int]]:
        users = random.sample(self.dataset.users, self.batch_size)

        pos_items = []
        neg_items = []
        for user in users:
            pos_items.append(self._sample_pos_item(user))
            neg_items.append(self._sample_neg_item(user))

        return users, pos_items, neg_items

    def _sample_pos_item(self, user: int) -> int:
        users_positive_items = self.dataset.train_data[user]
        idx = np.random.randint(low=0, high=len(users_positive_items))
        return users_positive_items[idx]

    def _sample_neg_item(self, user: int) -> int:
        users_positive_items = self.dataset.train_data[user]
        while True:
            idx = np.random.randint(low=0, high=len(self.dataset.items))
            if self.dataset.items[idx] not in users_positive_items:
                return self.dataset.items[idx]


if __name__ == "__main__":
    data = Dataset(path="./Data/gowalla")
    data.load()

    print(f"n_users: {data.user_num}")
    print(f"n_items: {data.item_num}")

    # _ = data.create_adjacency_matrix()
    sampler = Sampler(data, batch_size=2)
    print(sampler.sample())
