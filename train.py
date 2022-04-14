from argparse import ArgumentParser, Namespace
from classifier import Classifier
from encoders import BaselineEncoder, LSTMEncoder
from lr_stopping import LRStopping
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from snli import SNLIDataModule

import os


def train(args: Namespace):
    seed_everything(args.seed, workers=True)

    snli = SNLIDataModule(args.data_dir, args.batch_size, args.num_workers)
    snli.prepare_data()
    snli.setup(stage="fit")

    if args.encoder_arch == "baseline":
        repr_dim = snli.embed_dim
        encoder = BaselineEncoder()
    elif args.encoder_arch == "lstm":
        repr_dim = 2048
        encoder = LSTMEncoder(snli.embed_dim, repr_dim)
    else:
        raise Exception(f"Unsupported encoder architecture '{args.encoder_arch}'")

    log = TensorBoardLogger(args.log_dir, name=args.encoder_arch, default_hp_metric=False)
    gpus = 0 if args.no_gpu else 1
    trainer = Trainer.from_argparse_args(args, logger=log, gpus=gpus, callbacks=[LRStopping()])

    model_args = {"embeddings": snli.glove.vectors, "encoder": encoder}

    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        try:
            model = Classifier.load_from_checkpoint(args.checkpoint, **model_args)
        except TypeError:
            # old checkpoints missing repr_dim and n_classes in saved hparams
            model_args = {
                **model_args,
                "repr_dim": repr_dim,
                "n_classes": snli.num_classes
            }
            model = Classifier.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model_args = {
            **model_args,
            "repr_dim": repr_dim,
            "n_classes": snli.num_classes,
            **vars(args)
        }
        model = Classifier(**model_args)

    trainer.fit(model, snli)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--seed", type=int, default=420,
                        help="The seed to use for the RNG.")

    parser.add_argument("--max_epochs", type=int, default=-1,
                        help="The max amount of epochs to train the classifier.")

    parser.add_argument("--enable_progress_bar", action='store_true',
                        help="Whether to enable the progress bar (NOT recommended when logging to file).")

    parser.add_argument("--no_gpu", action='store_true',
                        help="Whether to NOT use a GPU accelerator for training.")

    parser.add_argument("--checkpoint", type=str,
                        help="The checkpoint from which to load a model.")

    # Model arguments
    Classifier.add_model_specific_args(parser)

    parser.add_argument("--encoder_arch", type=str, default="baseline",
                        choices=["baseline", "lstm"],  # TODO: implement more
                        help="The name of the encoder architecture to use.")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch size used by the dataloaders.")

    parser.add_argument("--num_workers", type=int, default=3,
                        help="The number of subprocesses used by the dataloaders.")

    # Directory arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The data directory to use for embeddings & datasets.")

    parser.add_argument("--log_dir", type=str, default="./lightning_logs",
                        help="The logging directory for Pytorch Lightning.")

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)
