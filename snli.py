from datasets import load_dataset, load_from_disk
from glove import GloVeEmbeddings
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from typing import Dict, List, Optional

import os
import pytorch_lightning as pl
import spacy


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.max_len = 402
        self.data_dir = os.path.join(data_dir, "snli")
        self.aligned_dir = self.data_dir + "_aligned"
        self.batch_size = batch_size

    def prepare_data(self):
        if not spacy.util.is_package("en_core_web_sm"):
            print("Downloading SpaCy English model (small)")
            spacy.cli.download("en_core_web_sm")

        if not os.path.exists(self.data_dir):
            tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
            def tokenize(sample_batch: Dict[str, List]) -> Dict[str, List]:
                for field in ('premise', 'hypothesis'):
                    sample_batch[field] = [tokenizer(f.lower()) for f in sample_batch[field]]
                return sample_batch
            
            print("Downloading SNLI data from HuggingFace")
            dataset = load_dataset('snli')
            print("Preprocessing data (lowercase + tokenize)")
            dataset = dataset.map(tokenize, batched=True)
            print("Saving to disk")
            dataset.save_to_disk(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        if os.path.exists(self.aligned_dir):
            dataset = load_from_disk(self.aligned_dir)
            glove = GloVeEmbeddings()
        else:
            dataset = load_from_disk(self.data_dir)

            def collect_words(sample_batch: Dict[str, List]) -> Dict[str, List]:
                words = []
                for field in ('premise', 'hypothesis'):
                    for sentence in sample_batch[field]:
                        words += sentence
                return { "words": words }

            print("Collecting train vocabulary")
            vocab = dataset['train'].map(collect_words, batched=True, remove_columns=dataset['train'].column_names)
            vocab = ["<pad>", "<unk>"] + list(set(vocab['words']))

            print("Aligning GloVe embeddings with train vocabulary")
            glove = GloVeEmbeddings()
            glove.update(vocab)
            
            def to_ids(sample_batch: Dict[str, List]) -> Dict[str, List]:
                def tokens_to_ids(tokens: List[str]) -> List[int]:
                    return [glove.get(token) for token in tokens]
                
                for field in ('premise', 'hypothesis'):
                    sample_batch[field] = [tokens_to_ids(t) for t in sample_batch[field]]
                return sample_batch
        
            print("Converting tokens to ids")
            dataset = dataset.map(to_ids, batched=True)

            pad_id = glove.w2i["<pad>"]
            def pad(ids: List[int]) -> List[int]:
                return ids + (self.max_len - len(ids)) * [pad_id]

            def concat_padded(sample: Dict[str, List]) -> Dict[str, List]:
                ids1 = sample['premise']
                ids2 = sample['hypothesis']
                sentence = pad(ids1) + pad(ids2)
                return { "sentences": sentence, "labels": sample["label"] }

            print("Concatenating sentences & padding")
            dataset = dataset.map(concat_padded, remove_columns=["label", "premise", "hypothesis"])

            print("Saving aligned dataset to disk")
            dataset.save_to_disk(self.aligned_dir)

        dataset = dataset.filter(lambda e: e['labels'] >= 0)
        dataset.set_format(type="torch", columns=["sentences", "labels"])
        if stage == "fit" or stage is None:
            self.snli_train = dataset['train']
            self.snli_val = dataset['validation']
        if stage == "test" or stage is None:
            self.snli_test = dataset['test']

        # expose glove
        self.glove = glove

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.snli_test, batch_size=self.batch_size, num_workers=4)
