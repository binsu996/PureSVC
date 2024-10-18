from matcha.models.components.flow_matching import CFM
import torch
from torch import nn
import math


class Bucketize(torch.nn.Module):
    def __init__(self, start, end, n_bins, log_scale=True):
        super().__init__()
        self.start = start if not log_scale else math.log(start)
        self.end = end if not log_scale else math.log(end)
        self.steps = n_bins+1
        self.log = log
        self.boundaries = torch.linspace(
            self.start,
            self.end,
            steps=self.steps
        )

    @torch.no_grad()
    def forward(self, x):
        return torch.bucketize(x, self.boundaries)


class BucketizeEmbedding(nn.Module):
    def __init__(self, start, end, n_bins, embedding_dim=256, log_scale=True):
        super().__init__()
        self.bucketize = Bucketize(start, end, n_bins, log_scale)
        self.embedding = nn.Embedding(
            num_embeddings=n_bins,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        ids = self.bucketize(x)
        embeddings = self.embedding(ids)
        return embeddings


class SVCCFM(torch.nn.Module):
    def __init__(self, model_config, attr_names):
        self.in_channels = 100
        self.out_channels = 513
        self.cfm = CFM(
            in_channels=self.in_channels,
            out_channel=self.out_channels
        )
        self.attr_names = attr_names

        if attr_names.get("whisper"):
            self.whisper_proj = nn.Linear(1280, self.in_channels)
        if attr_names.get("cam"):
            self.cam_proj = nn.Linear(192, self.in_channels)
        if attr_names.get("crepe"):
            self.crepe_proj = BucketizeEmbedding(
                0, 8000, 256,
                embedding_dim=self.in_channels
            )
        if attr_names.get("hubert"):
            self.hubert_proj = nn.Linear(256, self.in_channels)

    def prepare_mu(self, batch):
        mu = 0
        if self.attr_names.get("whisper"):
            mu += self.whisper_proj(batch.whisper.data)
        if self.attr_names.get("cam"):
            mu += self.cam_proj(batch.cam.data).unsqueeze(1)
        if self.attr_names.get("crepe"):
            mu += self.crepe_proj(batch.crepe.data)
        if self.attr_names.get("hubert"):
            mu += self.hubert_proj(batch.hubert.data)
        return mu

    def forward(self, batch):
        mu = self.prepare_mu(batch)
        x1 = batch.specs.data
        mask = batch.specs.mask

        loss = self.cfm.compute_loss(x1, mask, mu)
        return loss

    def inference(self, batch):
        mu = self.prepare_mu(batch)
        mask = batch.specs.mask
        sampled = self.cfm(mu, mask, 20)
        return sampled
