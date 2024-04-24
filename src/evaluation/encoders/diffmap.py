import torch

from src.model.arch import Model
from .base import BaseEncoder
from src import const
import scanpy as sc
import anndata as ad


class DiffMap(BaseEncoder):
    def __init__(self):
        super().__init__()
    def encode(self, batch):
        df = ad.AnnData(batch.cpu().numpy())
        sc.pp.neighbors(df, n_neighbors=25, n_pcs = const.HIDDEN_SIZE+1, use_rep = 'X')
        sc.tl.diffmap(df, n_comps = const.HIDDEN_SIZE+1)
        sliced = df.obsm["X_diffmap"][:, 1:].copy()
        return (torch.tensor(sliced).unsqueeze(dim=1)).to(const.DEVICE)
