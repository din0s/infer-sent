from argparse import ArgumentParser, Namespace
from classifier import Classifier
from encoders import BaselineEncoder, BiLSTMEncoder, LSTMEncoder, MaxBiLSTMEncoder
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.exceptions import ConvergenceWarning
from snli import SNLIDataModule
from torch.nn.utils.rnn import pad_sequence
from typing import List

import json
import logging
import numpy as np
import os
import senteval
import torch
import warnings


def handle_senteval(model: Classifier, encoder_arch: str, snli: SNLIDataModule, args: Namespace):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # def prepare(params: dict, samples: List[str]):

    def seq_to_ids(seq: List[str]) -> torch.IntTensor:
        return torch.IntTensor([snli.glove.get_id(t) for t in seq])

    def batcher(params: dict, batch: List[List[str]]) -> np.ndarray:
        batch_lens = torch.LongTensor([len(s) for s in batch])
        batch_ids = [seq_to_ids(s) for s in batch]

        m = params['model']
        x = pad_sequence(batch_ids, batch_first=True)
        x = m.embed(x)
        x = m.encoder(x, batch_lens)
        return x.detach().numpy()

    se_params = {
        "model": model,
        "task_path": args.task_dir,
        "seed": args.seed,
        "usepytorch": False,
        "kfold": 5,
        "classifier": {
            "nhid": 0,
            "optim": "adam",
            "batch_size": args.batch_size,
            "tenacity": 3,
            "epoch_size": 2
        }
    }

    # optional: setup logger
    logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(se_params, batcher)
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC", "SICKRelatedness", "SICKEntailment", "STS14"]
    results = se.eval(tasks)

    fname = os.path.join(args.log_dir, f"results_{encoder_arch}.json")
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            curr = json.load(f)
            results = {**curr, **results}

    with open(fname, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def test(args: Namespace):
    seed_everything(args.seed, workers=True)

    snli = SNLIDataModule(args.data_dir, args.batch_size, args.num_workers, args.enable_progress_bar)
    snli.prepare_data()
    snli.setup(stage="test")

    ckpt = torch.load(args.checkpoint)
    hparams = ckpt['hyper_parameters']
    encoder_arch = hparams['encoder_arch']
    lstm_state_dim = hparams['lstm_state_dim']

    if encoder_arch == "baseline":
        encoder = BaselineEncoder()
    elif encoder_arch == "lstm":
        encoder = LSTMEncoder(snli.embed_dim, lstm_state_dim)
    elif encoder_arch == "bilstm":
        encoder = BiLSTMEncoder(snli.embed_dim, lstm_state_dim)
    elif encoder_arch == "bilstm-max":
        encoder = MaxBiLSTMEncoder(snli.embed_dim, lstm_state_dim)
    else:
        raise Exception(f"Unsupported encoder architecture '{encoder_arch}'")

    model_args = {"embeddings": snli.glove.vectors, "encoder": encoder}
    model = Classifier.load_from_checkpoint(args.checkpoint, **model_args)

    if args.senteval:
        handle_senteval(model, encoder_arch, snli, args)
    else:
        log = TensorBoardLogger(args.log_dir, name=encoder_arch, version='eval', default_hp_metric=False)
        trainer = Trainer.from_argparse_args(args, logger=log, gpus=(0 if args.no_gpu else 1), max_epochs=-1)
        trainer.test(model, snli)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The checkpoint from which to load a model.")

    parser.add_argument("--senteval", action='store_true',
                        help="Whether to evaluate the model using the SentEval toolkit.")

    parser.add_argument("--seed", type=int, default=420,
                        help="The seed to use for the RNG.")

    parser.add_argument("--enable_progress_bar", action='store_true',
                        help="Whether to enable the progress bar (NOT recommended when logging to file).")

    parser.add_argument("--no_gpu", action='store_true',
                        help="Whether to NOT use a GPU accelerator for training.")

    parser.add_argument("--task", type=str,
                        choices=['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                                 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                                 'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                                 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'],
                        help="The specific SentEval task to evaluate for. " +
                             "If not defined, the model will be evaluated on the tasks of the Conneau et al. paper.")

    # Model arguments
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch size used by the dataloaders.")

    parser.add_argument("--num_workers", type=int, default=3,
                        help="The number of subprocesses used by the dataloaders.")

    # Directory arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The data directory to use for embeddings & datasets.")

    parser.add_argument("--log_dir", type=str, default="./lightning_logs",
                        help="The logging directory for Pytorch Lightning.")

    parser.add_argument("--task_dir", type=str, default="./SentEval/data",
                        help="The directory with the downloaded datasets for SentEval.")

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    test(args)
