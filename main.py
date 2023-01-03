from datetime import datetime

from data_loader import DataLoader
from util import simple_logger


if __name__ == "__main__":
    simple_logger("Initializing data loader", __file__)
    dataloader = DataLoader(path="./Data/gowalla", batch_size=64)
    norm_adj = dataloader.get_adjacency_matrix()