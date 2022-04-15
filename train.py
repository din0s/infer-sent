from argparse import ArgumentParser, Namespace
from callbacks import LRStopping
from encoders import BaselineEncoder, BiLSTMEncoder, LSTMEncoder, MaxBiLSTMEncoder
from models import Classifier
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from snli import SNLIDataModule

import os


def train(args: Namespace):
    seed_everything(args.seed, workers=True)

    snli = SNLIDataModule(args.data_dir, args.batch_size, args.num_workers, args.enable_progress_bar)
    snli.prepare_data()
    snli.setup(stage="fit")

    if args.encoder_arch == "baseline":
        repr_dim = snli.embed_dim
        encoder = BaselineEncoder()
    else:
        repr_dim = args.lstm_state_dim

        if args.encoder_arch == "lstm":
            encoder = LSTMEncoder(snli.embed_dim, args.lstm_state_dim)
        elif args.encoder_arch == "bilstm":
            repr_dim *= 2
            encoder = BiLSTMEncoder(snli.embed_dim, args.lstm_state_dim)
        elif args.encoder_arch == "bilstm-max":
            repr_dim *= 2
            encoder = MaxBiLSTMEncoder(snli.embed_dim, args.lstm_state_dim)
        else:
            raise Exception(f"Unsupported encoder architecture '{args.encoder_arch}'")

    ckpt_cb = ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")
    log = TensorBoardLogger(args.log_dir, name=args.encoder_arch, default_hp_metric=False)
    gpus = 0 if args.no_gpu else 1
    trainer = Trainer.from_argparse_args(args, logger=log, gpus=gpus, callbacks=[LRStopping(), ckpt_cb])

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

    trainer_args = {}
    if args.checkpoint:
        # THIS DOES NOT WORK IN PL 1.6.0 WHEN USING EARLY STOPPING OR AN LR SCHEDULER THAT MONITORS VAL LOSS/ACC
        # trainer_args['ckpt_path'] = args.checkpoint
        pass

    trainer.fit(model, snli, **trainer_args)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--seed", type=int, default=420,
                        help="The seed to use for the RNG.")

    parser.add_argument("--max_epochs", type=int, default=20,
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
                        choices=["baseline", "lstm", "bilstm", "bilstm-max"],
                        help="The name of the encoder architecture to use.")

    parser.add_argument("--lstm_state_dim", type=int, default=2048,
                        help="The dimensionality of the hidden state in each LSTM cell.")

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
