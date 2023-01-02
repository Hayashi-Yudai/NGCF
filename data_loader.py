import numpy as np
import pathlib
from scipy import sparse as sp

class DataLoader:
    def __init__(self, path: str, batch_size: int):
        self.path = pathlib.Path(path)
        self.batch_size = batch_size

        self.exist_users = []

        self.n_users = 0
        self.n_items = 0
        self.n_train = 0
        self.n_test = 0
        self.train_data = {}
        self.test_data = {}

        self._load_train(train_file=self.path / "train.txt")
        self._load_test(test_file=self.path / "test.txt")
        
        self.n_items += 1
        self.n_users += 1

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u, items in self.train_data.items():
            for item in items:
                self.R[u, item] = 1.0

    def _load_train(self, train_file: pathlib.Path):
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) == 0: continue

                line = line.strip("\n").split(" ")
                items = [int(i) for i in line[1:]]
                uid = int(line[0])

                self.exist_users.append(uid)

                self.n_items = max(self.n_items, max(items))
                self.n_users = max(self.n_users, uid)
                self.n_train += len(items)

                self.train_data[uid] = items
    
    def _load_test(self, test_file: pathlib.Path):
        with open(test_file) as f:
            for line in f.readlines():
                if len(line) == 0: continue

                try:
                    line = line.strip("\n").split(" ")
                    uid = int(line[0])
                    items = [int(i) for i in line[1:]]
                except Exception:
                    continue
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)

                self.test_data[uid] = items



if __name__ == "__main__":
    data = DataLoader(path="./Data/gowalla", batch_size=64)

    print(f"n_items: {data.n_items}")
    print(f"n_users: {data.n_users}")