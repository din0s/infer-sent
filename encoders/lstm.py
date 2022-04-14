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
            # https://github.com/ihsgnef/InferSent-1/blob/master/models.py#L74
            out, _ = pad_packed_sequence(out, batch_first=True)
            out = [o[:l].data for o, l in zip(out, lens)]
            out = [torch.max(o, dim=0)[0] for o in out]
            return torch.vstack(out)

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
