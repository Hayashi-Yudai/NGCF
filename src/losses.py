import torch.nn as nn
import torch

from typing import Optional


class BPRLoss(nn.Module):
    def __init__(self, regularize: Optional[float] = None):
        super(BPRLoss, self).__init__()

        self.regularize = regularize

    def forward(
        self, user: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> torch.Tensor:
        pos_score = torch.sum(torch.mul(user, pos_item), axis=1)
        neg_score = torch.sum(torch.mul(user, neg_item), axis=1)

        loss = -torch.sum(nn.LogSigmoid()(pos_score - neg_score))

        if self.regularize:
            norms = (
                torch.norm(user) ** 2
                + torch.norm(pos_item) ** 2
                + torch.norm(neg_item) ** 2
            )

            loss += self.regularize * norms

        return loss
