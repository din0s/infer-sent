from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


class LRStopping(LearningRateMonitor):
    def __init__(self, threshold: float = 1e-5, interval: str = "epoch", log_momentum: bool = False):
        super().__init__(interval, log_momentum)
        self.threshold = threshold

    def on_validation_epoch_end(self, trainer: Trainer, module: LightningModule):
        super().on_validation_epoch_end(trainer, module)

        lrs = self._extract_stats(trainer, self.logging_interval)
        min_val = min(lrs.values())

        if min_val < self.threshold:
            trainer.should_stop = True
