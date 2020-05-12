import numpy as np
import torch
import torch.nn as nn

from . import GatedCNN


class UnsupervisedCNNEmbeddingNetwork(nn.Module):
    def __init__(self, embedding, num_channels, kernel_size=3, pos=10, neg=50):
        super(UnsupervisedCNNEmbeddingNetwork, self).__init__()

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        n_items, embedding_size = embedding.shape
        self._embedding = nn.Embedding(n_items, embedding_size)
        self._embedding.weight = nn.Parameter(torch.from_numpy(embedding).float())
        self._embedding.weight.requires_grad = False

        self.network = GatedCNN(embedding_size, num_channels, kernel_size)

        self.pos = pos
        self.neg = neg

    def forward(self, x, y):
        # x: Tensor (batch_size, seq_len)
        # y: Tensor (batch_size, pos+neg)

        x = self._embedding(x).permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)
        x = self.network(x).unsqueeze(1)  # (batch_size, 1, embedding_size)

        y = self._embedding(y).permute(0, 2, 1)  # (batch_size, embedding_size, pos+neg)
        logits = torch.bmm(x, y).squeeze(1)  # (batch_size, 1, pos+neg)

        return logits
