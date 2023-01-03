import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp
from typing import List


class NGCF(nn.Module):
    def __init__(self, n_user: int, n_item: int, norm_adj: sp.csr_matrix) -> None:
        super(NGCF, self).__init__()

        self.n_user: int = n_user
        self.n_item: int = n_item
        self.device: str = "cuda:0"  # TODO: Fix
        self.emb_size: int = 64
        self.batch_size: int = 64
        self.layers: List[int] = [64, 64, 64]
        self.decay: float = 1e-5

        self.norm_adj = norm_adj

        self.embedding_dict, self.weight_dict = self.init_weight()

    def init_weight(self):
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(self.n_user, self.emb_size))
                ),
                "item_emb": nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(self.n_item, self.emb_size))
                ),
            }
        )
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update(
                {
                    f"W_gc_{k}": nn.Parameter(
                        nn.init.xavier_uniform_(torch.empty(layers[k], layers[k + 1]))
                    ),
                    f"b_gc_{k}": nn.Parameter(
                        nn.init.xavier_uniform_(torch.empty(1, layers[k + 1]))
                    ),
                    f"W_bi_{k}": nn.Parameter(
                        nn.init.xavier_uniform_(torch.empty(layers[k], layers[k + 1]))
                    ),
                    f"b_bi_{k}": nn.Parameter(
                        nn.init.xavier_uniform_(torch.empty(1, layers[k + 1]))
                    ),
                }
            )

        return embedding_dict, weight_dict
