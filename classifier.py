from argparse import ArgumentParser
from encoders import BaselineEncoder
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Embedding, Linear, Module, Sequential
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchmetrics.functional as TF


class Classifier(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("Classifier")

        parser.add_argument("--classifier_hidden_dim", type=int, default=512,
                            help="The dimensionality of the hidden layer in the MLP classifier.")

        parser.add_argument("--lr", type=float, default=0.1,
                            help="The initial learning rate for the classifier.")

        return parent_parser

    def __init__(
            self,
            embeddings: Tensor,
            encoder: Module,
            n_classes: int,
            classifier_hidden_dim: int,
            lr: float,
            **kwargs):
        super().__init__()
        self.save_hyperparameters("classifier_hidden_dim", "lr", ignore=["encoder"])

        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        self.encoder = encoder

        if isinstance(encoder, BaselineEncoder):
            rep_dim = 300

        self.classifier = Sequential(
            Linear(4 * rep_dim, self.hparams.classifier_hidden_dim),
            Linear(self.hparams.classifier_hidden_dim, self.hparams.classifier_hidden_dim),
            Linear(self.hparams.classifier_hidden_dim, n_classes)
        )

    def forward(self, sentences: Tensor) -> Tensor:
        u, v = torch.chunk(sentences, 2, dim=1)
        u = self.encoder(self.embed(u))
        v = self.encoder(self.embed(v))

        z = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        z = self.classifier(z)
        return z

    def step(self, batch: Dict[str, Tensor], stage: str) -> Tensor:
        sentences, labels = batch['sentences'], batch['labels']

        logits = self(sentences)
        loss = F.cross_entropy(logits, labels)
        acc = TF.accuracy(logits, labels)

        self.log(f'{stage}_acc', acc)
        self.log(f'{stage}_loss', loss)

        return loss

    def training_step(self, batch: Dict[str, Tensor], _: Tensor) -> Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: Dict[str, Tensor], _: Tensor) -> Tensor:
        return self.step(batch, 'val')

    def test_step(self, batch: Dict[str, Tensor], _: Tensor) -> Tensor:
        return self.step(batch, 'test')

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = SGD(self.parameters(), lr=self.hparams.lr)

        scheduler1 = ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=0)
        scheduler2 = StepLR(optimizer, step_size=1, gamma=0.99)

        scheduler1_cfg = {
            "scheduler": scheduler1,
            "interval": "epoch",
            "monitor": "val_acc",
            "name": "lr"
        }

        scheduler2_cfg = {
            "scheduler": scheduler2,
            "interval": "epoch",
            "name": "lr"
        }

        return [optimizer], [scheduler1_cfg, scheduler2_cfg]
