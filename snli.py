from datasets import load_dataset, load_from_disk
from glove import GloVeEmbeddings
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from typing import Dict, List, Tuple, Optional

import os
import pytorch_lightning as pl
import spacy
import torch


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "hf_cache")
        self.dataset_dir = os.path.join(data_dir, "snli")
        self.aligned_dir = self.dataset_dir + "_aligned"
        self.batch_size = batch_size
        self.workers = os.cpu_count()

    def prepare_data(self):
        if os.path.exists(self.aligned_dir):
            return

        if not spacy.util.is_package("en_core_web_sm"):
            print("Downloading SpaCy English model (small)")
            spacy.cli.download("en_core_web_sm")

        if not os.path.exists(self.dataset_dir):
            tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

            def tokenize(sample_batch: Dict[str, List]) -> Dict[str, List]:
                for field in ('premise', 'hypothesis'):
                    sample_batch[field] = [tokenizer(f.lower()) for f in sample_batch[field]]
                return sample_batch

            print("Downloading SNLI data from HuggingFace")
            dataset = load_dataset('snli', cache_dir=self.cache_dir)
            print("Preprocessing data (lowercase + tokenize)")
            dataset = dataset.map(tokenize, batched=True)
            print("Saving to disk")
            dataset.save_to_disk(self.dataset_dir)

    def setup(self, stage: Optional[str] = None):
        if os.path.exists(self.aligned_dir):
            dataset = load_from_disk(self.aligned_dir)
            glove = GloVeEmbeddings(self.data_dir)
        else:
            dataset = load_from_disk(self.dataset_dir)

            def collect_words(sample_batch: Dict[str, List]) -> Dict[str, List]:
                words = []
                for field in ('premise', 'hypothesis'):
                    for sentence in sample_batch[field]:
                        words += sentence
                return {"words": words}

            print("Collecting train vocabulary")
            vocab = dataset['train'].map(collect_words, batched=True, remove_columns=dataset['train'].column_names)
            vocab = ["<pad>", "<unk>"] + list(set(vocab['words']))

            print("Aligning GloVe embeddings with train vocabulary")
            glove = GloVeEmbeddings(self.data_dir)
            glove.update(vocab)

            def to_ids(sample_batch: Dict[str, List]) -> Dict[str, List]:
                def tokens_to_ids(tokens: List[str]) -> List[int]:
                    return [glove.get_id(token) for token in tokens]

                for field in ('premise', 'hypothesis'):
                    sample_batch[field] = [tokens_to_ids(t) for t in sample_batch[field]]
                return sample_batch

            print("Converting tokens to ids")
            dataset = dataset.map(to_ids, batched=True)

            # def concat(sample: Dict[str, List]) -> Dict[str, List]:
            #     ids1 = sample['premise']
            #     ids2 = sample['hypothesis']
            #     sentence = ids1 + [-1] + ids2  # -1 acts as the chunking id
            #     return {"sentences": sentence, "labels": sample["label"]}
            #
            # print("Concatenating sentences")
            # dataset = dataset.map(concat, remove_columns=["label", "premise", "hypothesis"])

            print("Saving aligned dataset to disk")
            dataset.save_to_disk(self.aligned_dir)

        dataset = dataset.filter(lambda e: e['label'] >= 0)
        dataset.set_format(type="torch", columns=["premise", "hypothesis", "label"])
        if stage == "fit" or stage is None:
            self.snli_train = dataset['train']
            self.snli_val = dataset['validation']
            self.num_classes = len(torch.unique(self.snli_train['label']))
        if stage == "test" or stage is None:
            self.snli_test = dataset['test']

        # expose glove
        self.glove = glove

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_train,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=True, drop_last=True,
                          collate_fn=self._collate_fn
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_val,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          collate_fn=self._collate_fn
                          )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_test,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          collate_fn=self._collate_fn
                          )

    def _collate_fn(self, batch: List[Dict[str, Tensor]]) -> Tuple[Tuple, Tuple, Tensor]:
        premises = [x['premise'] for x in batch]
        hypotheses = [x['hypothesis'] for x in batch]
        labels = torch.LongTensor([x['label'] for x in batch])

        p_padded = pad_sequence(premises, batch_first=True)
        h_padded = pad_sequence(hypotheses, batch_first=True)

        p_lengths = torch.LongTensor([len(x) for x in premises])
        h_lengths = torch.LongTensor([len(x) for x in hypotheses])

        return (p_padded, p_lengths), (h_padded, h_lengths), labels
