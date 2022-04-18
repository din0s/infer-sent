from argparse import ArgumentParser
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Embedding, Linear, Module, Parameter, Sequential
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchmetrics.functional as TF


class Classifier(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
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
            repr_dim: int,
            n_classes: int,
            classifier_hidden_dim: int,
            lr: float,
            **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["embeddings", "encoder"])

        embed_num, embed_dim = embeddings.data.shape
        self.embed = Embedding(embed_num, embed_dim)
        self.embed.weight.requires_grad = False
        self.encoder = encoder

        self.classifier = Sequential(
            Linear(4 * repr_dim, self.hparams.classifier_hidden_dim),
            Linear(self.hparams.classifier_hidden_dim, self.hparams.classifier_hidden_dim),
            Linear(self.hparams.classifier_hidden_dim, n_classes)
        )

    @torch.no_grad()
    def load_embeddings(self, embeddings: Tensor):
        self.embed.weight = Parameter(embeddings)

    def forward(self, p: Tensor, h: Tensor, p_len: Tensor, h_len: Tensor) -> Tensor:
        # Embed
        p = self.embed(p)
        h = self.embed(h)

        # Encode
        p = self.encoder(p, p_len)
        h = self.encoder(h, h_len)

        # Concat representations
        z = torch.cat([p, h, torch.abs(p - h), p * h], dim=1)

        # Classify
        z = self.classifier(z)

        return z

    def step(self, batch: Tuple[Tuple, Tuple, Tensor], stage: str) -> Tensor:
        (p, p_len), (h, h_len), labels = batch

        logits = self(p, h, p_len, h_len)
        loss = F.cross_entropy(logits, labels)
        acc = TF.accuracy(logits, labels)

        self.log(f'{stage}_acc', acc)
        self.log(f'{stage}_loss', loss)

        return loss

    def training_step(self, batch: Tuple[Tuple, Tuple, Tensor], _: Tensor) -> Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: Tuple[Tuple, Tuple, Tensor], _: Tensor) -> Tensor:
        return self.step(batch, 'val')

    def test_step(self, batch: Tuple[Tuple, Tuple, Tensor], _: Tensor) -> Tensor:
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

    def on_load_checkpoint(self, checkpoint: dict):
        # old models still have embedding weights and not just their shape
        # TODO: manually remove the embeddings before turning in the assignment
        if 'embed.weight' not in checkpoint['state_dict']:
            checkpoint['state_dict']['embed.weight'] = torch.empty(checkpoint['embed_shape'])

    def on_save_checkpoint(self, checkpoint: dict):
        weights = checkpoint['state_dict']['embed.weight']
        checkpoint['embed_shape'] = weights.shape
        del weights
