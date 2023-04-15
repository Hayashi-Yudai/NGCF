import numpy as np
import random
import torch
from torch import nn, tensor, optim
from tqdm import tqdm

from data_loader import Dataset
from util import simple_logger


class MFConfig:
    dataset_name = "gowalla"

    epochs = 400
    lr = 1e-2
    decay = 1e-4
    batch_size = 1024
    train_ratio = 0.6


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int) -> None:
        super().__init__()

        self.user_matrix = nn.Embedding(n_users, k, sparse=True)
        self.item_matrix = nn.Embedding(n_items, k, sparse=True)

    def forward(self, user_ids: tensor, item_ids: tensor):
        user_embedding = self.user_matrix(user_ids)
        item_embedding = self.item_matrix(item_ids)

        return (user_embedding * item_embedding).sum(axis=1)

    def rating(self, user_ids: tensor, item_ids: tensor):
        user_embedding = self.user_matrix(user_ids)
        item_embedding = self.item_matrix(item_ids)

        return torch.matmul(user_embedding, item_embedding.t())


class BiasedMSELoss(nn.Module):
    def __init__(self, decay: float) -> None:
        super(BiasedMSELoss, self).__init__()

        self.decay = decay

    def forward(self, inputs, targets):
        mse_loss = nn.MSELoss()(inputs, targets)

        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.sum(param.pow(2))

        loss = mse_loss + self.decay * reg_loss

        return loss


class PreprocessorMF:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.processed_data = []

        simple_logger("Preprocessor start", __name__)
        for user, items in self.dataset.train_data.items():
            for item in items:
                self.processed_data.append([user, item, 1.0])

        simple_logger("Preprocessor end", __name__)


if __name__ == "__main__":
    dataset = Dataset(path=f"./Data/{MFConfig.dataset_name}")
    dataset.load()

    model = MatrixFactorization(
        n_users=dataset.user_num, n_items=dataset.item_num, k=128
    )

    loss = BiasedMSELoss(decay=MFConfig.decay)
    opt = optim.SGD(model.parameters(), lr=MFConfig.lr)

    sample = PreprocessorMF(dataset).processed_data
    random.shuffle(sample)

    train_samples = sample[: int(len(sample) * MFConfig.train_ratio)]
    valid_samples = sample[int(len(sample) * MFConfig.train_ratio) :]

    best_loss = 1e9
    for epoch in range(MFConfig.epochs):
        random.shuffle(train_samples)
        accum_loss = 0.0

        for i in tqdm(range(0, len(train_samples), MFConfig.batch_size)):
            model.zero_grad()

            d = np.array(train_samples[i : i + MFConfig.batch_size])

            user = torch.LongTensor(d.T[0])
            item = torch.LongTensor(d.T[1])
            rate = torch.FloatTensor(d.T[2])
            preds = model(user, item)
            ls = loss(preds, rate)
            accum_loss += ls.item()
            ls.backward()

            opt.step()

        simple_logger(f"Epoch {epoch + 1}: Loss: {accum_loss}", __name__)

        # Validation
        accum_loss_valid = 0.0
        valid_data = np.array(valid_samples)
        user = torch.LongTensor(d.T[0])
        item = torch.LongTensor(d.T[1])
        rate = torch.FloatTensor(d.T[2])
        preds = model(user, item)
        ls = loss(preds, rate)
        accum_loss_valid += ls.item()
        simple_logger(f"Validation Loss: {accum_loss_valid}", __name__)

        if accum_loss_valid < best_loss:
            torch.save(
                model.state_dict(), f"./outputs/{MFConfig.dataset_name}_MF_best.pkl"
            )
            simple_logger("Saved model", __name__)
            best_loss = accum_loss_valid
