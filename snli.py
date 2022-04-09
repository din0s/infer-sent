from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from typing import Dict, List, Optional

import os
import pytorch_lightning as pl
import spacy


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "snli")
        self.batch_size = batch_size

    def prepare_data(self):
        if not spacy.util.is_package("en_core_web_sm"):
            print("Downloading SpaCy English model (small)")
            spacy.cli.download("en_core_web_sm")

        if os.path.exists(self.data_dir):
            print("SNLI dataset exists locally, loading from disk")
            self.dataset = load_from_disk(self.data_dir)
        else:
            tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
            def preprocess(sbatched: Dict[str, List[str]]) -> Dict[str, List[str]]:
                sbatched['premise'] = [tokenizer(p.lower()) for p in sbatched['premise']]
                sbatched['hypothesis'] = [tokenizer(h.lower()) for h in sbatched['hypothesis']]
                return sbatched

            print("Downloading SNLI data from HuggingFace")
            self.dataset = load_dataset('snli')
            print("Preprocessing data (lowercasing + tokenization)")
            self.dataset = self.dataset.map(preprocess, batched=True)
            print("Saving to disk")
            self.dataset.save_to_disk(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.snli_train = self.dataset['train']
            self.snli_val = self.dataset['validation']
        if stage == "test" or stage is None:
            self.snli_test = self.dataset['test']

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_test, batch_size=self.batch_size, num_workers=4)
