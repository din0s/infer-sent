from torch import Tensor
from torch.nn import LSTM, Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch


class BaseLSTMEncoder(Module):
    def __init__(self, embedding_dim: int, state_dim: int, bidirectional: bool, pooling: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.lstm = LSTM(embedding_dim, state_dim, batch_first=True, bidirectional=bidirectional)

    def forward(self, x: Tensor, lens: Tensor) -> Tensor:
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_t, _) = self.lstm(x)

        if not self.bidirectional:
            # normal LSTM, return just the final hidden state
            return h_t[0]

        if self.pooling:
            # BiLSTM with max pooling over each dimension of the hidden states
            out, _ = pad_packed_sequence(out, batch_first=True)

            # substitute zeroes (e.g. from padding) with -inf
            with torch.no_grad():
                out[out == 0] = -1e9

            out, _ = torch.max(out, dim=1)
            return out

        # BiLSTM, stack the two final hidden states
        return torch.hstack([*h_t])


class LSTMEncoder(BaseLSTMEncoder):
    def __init__(self, embedding_dim: int, state_dim: int):
        super().__init__(embedding_dim, state_dim, bidirectional=False, pooling=False)


class BiLSTMEncoder(BaseLSTMEncoder):
    def __init__(self, embedding_dim: int, state_dim: int):
        super().__init__(embedding_dim, state_dim, bidirectional=True, pooling=False)


class MaxBiLSTMEncoder(BaseLSTMEncoder):
    def __init__(self, embedding_dim: int, state_dim: int):
        super().__init__(embedding_dim, state_dim, bidirectional=True, pooling=True)
