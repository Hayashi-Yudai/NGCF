import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp
from typing import List, Tuple, Dict


class NGCF(nn.Module):
    def __init__(
        self, n_user: int, n_item: int, norm_adj: sp.csr_matrix, config
    ) -> None:
        super(NGCF, self).__init__()

        self.n_user: int = n_user
        self.n_item: int = n_item
        self.device: str = config.device
        self.emb_size: int = config.emb_size
        self.batch_size: int = config.batch_size
        self.layers: List[int] = config.layers
        self.decay: float = config.decay
        self.mess_dropout: List[float] = config.mess_dropout
        self.node_dropout: float = config.node_dropout[0]

        self.norm_adj = norm_adj
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(
            self.device
        )

        self.embedding_dict, self.weight_dict = self.init_weight()

    def init_weight(self) -> Tuple[Dict[str, torch.Tensor]]:
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

    def forward(
        self,
        users: List[int],
        pos_items: List[int],
        neg_items: List[int],
        drop_flag: bool = True,
    ) -> Tuple[torch.Tensor]:
        if drop_flag:
            A_hat = self.sparse_dropout(
                self.sparse_norm_adj, self.node_dropout, self.sparse_norm_adj._nnz()
            )
        else:
            A_hat = self.sparse_norm_adj

        # Eq. (1) in the paper
        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            # Eq. (7) in the paper
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            sum_embeddings = (
                torch.matmul(
                    side_embeddings + ego_embeddings, self.weight_dict[f"W_gc_{k}"]
                )
                + self.weight_dict[f"b_gc_{k}"]
            )
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = (
                torch.matmul(bi_embeddings, self.weight_dict[f"W_bi_{k}"])
                + self.weight_dict[f"b_bi_{k}"]
            )

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings
            )
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        # Eq. (9) in the paper
        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[: self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user :, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def sparse_dropout(
        self, x: torch.Tensor, rate: float, noise_shape: Tuple[int]
    ) -> torch.Tensor:
        """
        Node dropout
        """
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

        return out * (1.0 / (1 - rate))

    def _convert_sp_mat_to_sp_tensor(self, X) -> torch.Tensor:
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def create_bpr_loss(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -torch.sum(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (
            torch.norm(users) ** 2
            + torch.norm(pos_items) ** 2
            + torch.norm(neg_items) ** 2
        )
        for w in self.weight_dict.values():
            regularizer += torch.norm(w) ** 2

        emb_loss = self.decay * regularizer

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(
        self, u_g_embeddings: torch.Tensor, i_g_embeddings: torch.Tensor
    ) -> float:
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
