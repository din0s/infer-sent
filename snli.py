from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from typing import Dict, List, Optional

import os
import pytorch_lightning as pl
import spacy

class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, w2i: Dict[str, int], data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.w2i = w2i
        self.max_len = 402
        self.data_dir = os.path.join(data_dir, "snli")
        self.batch_size = batch_size

    def prepare_data(self):
        if not spacy.util.is_package("en_core_web_sm"):
            print("Downloading SpaCy English model (small)")
            spacy.cli.download("en_core_web_sm")

        if not os.path.exists(self.data_dir):
            tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
            def tokenize_to_ids(sentence: str) -> List[int]:
                tokens = tokenizer(sentence.lower())
                ids = [self.w2i[t] if t in self.w2i else -1 for t in tokens]
                return ids

            pad_id = self.w2i["<pad>"]
            def pad(ids: List[int]) -> List[int]:
                return ids + (self.max_len - len(ids)) * [pad_id]

            def concat_padded(ids1: List[int], ids2: List[int]) -> List[int]:
                return pad(ids1) + pad(ids2)

            def preprocess(sample: Dict[str, List]) -> Dict[str, List]:
                premise = tokenize_to_ids(sample['premise'])
                hypothesis = tokenize_to_ids(sample['hypothesis'])
                return { "sentences": concat_padded(premise, hypothesis), "labels": sample['label'] }

            print("Downloading SNLI data from HuggingFace")
            dataset = load_dataset('snli')
            print("Preprocessing data")
            dataset = dataset.map(preprocess, remove_columns=["premise", "hypothesis", "label"])
            print("Saving to disk")
            dataset.save_to_disk(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        dataset = load_from_disk(self.data_dir)
        dataset.set_format(type="torch", columns=["sentences", "labels"])
        if stage == "fit" or stage is None:
            self.snli_train = dataset['train']
            self.snli_val = dataset['validation']
        if stage == "test" or stage is None:
            self.snli_test = dataset['test']

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_test, batch_size=self.batch_size, num_workers=4)
