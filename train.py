from argparse import ArgumentParser, Namespace
from classifier import Classifier
from encoders.baseline import BaselineEncoder
from lr_stopping import LRStopping
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from snli import SNLIDataModule

import os


def train(args: Namespace):
    seed_everything(args.seed, workers=True)

    snli = SNLIDataModule(args.data_dir, args.batch_size)
    snli.prepare_data()
    snli.setup(stage="fit")

    if args.encoder_arch == "baseline":
        encoder = BaselineEncoder()
    else:
        raise Exception(f"Unsupported encoder architecture '{args.encoder_arch}'")

    model = Classifier(snli.glove.vectors, encoder, n_classes=snli.num_classes, **vars(args))

    log = TensorBoardLogger(args.log_dir, name=None, default_hp_metric=False)
    gpus = 1 if args.use_gpu else 0
    trainer = Trainer.from_argparse_args(args, logger=log, gpus=gpus, callbacks=[LRStopping()])
    trainer.fit(model, snli)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--seed", type=int, default=420,
                        help="The seed to use for the RNG.")

    parser.add_argument("--max_epochs", type=int, default=-1,
                        help="The max amount of epochs to train the classifier.")

    parser.add_argument("--use_gpu", action='store_true', default=True,
                        help="Whether to use a GPU accelerator for training.")

    # Model arguments
    Classifier.add_model_specific_args(parser)

    parser.add_argument("--encoder_arch", type=str, default="baseline",
                        choices=["baseline"],  # TODO: implement more
                        help="The name of the encoder architecture to use.")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch size used by the dataloaders.")

    # Directory arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The data directory to use for embeddings & datasets.")

    parser.add_argument("--log_dir", type=str, default="./lightning_logs",
                        help="The logging directory for Pytorch Lightning.")

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    train(args)
