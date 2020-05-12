import torch.nn as nn

'''
    In : (N, input_channels, sentence_len)
    Out: (N, output_channels, sentence_len)
'''


class GatedCNN(nn.Module):

    def __init__(self, embedding_size, num_channels, kernel_size=3, max_k=3):
        super(GatedCNN, self).__init__()

        self.max_k = max_k

        num_layers = len(num_channels)
        layers = []

        for i in range(num_layers):
            in_channel = embedding_size if i == 0 else num_channels[i - 1]
            out_channel = num_channels[i]
            layers += [GatedBlock(in_channel, out_channel, kernel_size)]

        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[num_layers - 1] * max_k, embedding_size)

    def forward(self, x):
        # x: (bs, embedding_size, seq_len)
        bs = x.size(0)
        x = self.cnn(x)
        # --> x: (bs, out_channel, seq_len)
        x = self.kmax_pooling(x, 2, k=self.max_k).view(bs, -1)
        # --> x: (bs, out_channel * k)
        x = self.fc(x)
        # --> x: (bs, embedding_size)
        return x

    @staticmethod
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)


class GatedBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dropout=0.5):
        super(GatedBlock, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding)
        self.conv_gated = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv_gated(x)

        h = a * b
        h = self.dropout(h)
        return h
