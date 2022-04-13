from torch import Tensor
from torch.nn import LSTM, Module


class LSTMEncoder(Module):
    def __init__(self, embedding_dim: int, lstm_repr_dim: int):
        super().__init__()
        self.lstm = LSTM(embedding_dim, lstm_repr_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        out, (h_t, c_t) = self.lstm(x)
        return h_t[0]
