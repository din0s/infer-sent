from torch import Tensor
from torch.nn import Module


class BaselineEncoder(Module):
    def forward(self, x: Tensor) -> Tensor:
        # Count non-zero embeddings per batch
        nonzero = (x.sum(dim=2) != 0.).sum(dim=1, keepdim=True)
        # Average the individual word embeddings excluding pads
        return x.sum(dim=1) / nonzero
