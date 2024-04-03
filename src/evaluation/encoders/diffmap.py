import torch

from src.model.arch import Model
from .base import BaseEncoder
from src import const
import scanpy as sc


class DiffMap(BaseEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, batch):
        sc.pp.neighbors(batch, 50)
        res = sc.tl.diffmap(batch, 50, copy=True)
        return torch.tensor(res.obsm["X_diffmap"][:, 1]).to(const.DEVICE)
