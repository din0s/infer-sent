from torch import Tensor
from tqdm import tqdm
from typing import List

import os
import re
import requests
import shutil
import torch
import zipfile

GLOVE_URL = "http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip"
ZIP_NAME = GLOVE_URL.split("/")[-1]


class GloVeEmbeddings:
    def __init__(self, data_dir: str = "./data/", pbar: bool = True):
        self.data_dir = os.path.join(data_dir, "glove")
        self.zip_path = os.path.join(self.data_dir, ZIP_NAME)
        self.txt_path = os.path.join(self.data_dir, re.sub(r"\.zip$", ".txt", ZIP_NAME))
        self.pt_path = os.path.join(self.data_dir, re.sub(r"\.zip$", ".pt", ZIP_NAME))
        self.pbar = pbar

        if not (os.path.exists(self.txt_path) or os.path.exists(self.pt_path)):
            self._check_zip_exists()

        self._build_vocabulary()

    def __getitem__(self, id: int) -> Tensor:
        if id < 0 or id >= len(self.vectors):
            id = self.w2i["<unk>"]

        return self.vectors[id]

    def get_id(self, token: str) -> int:
        if token not in self.w2i:
            token = "<unk>"

        return self.w2i[token]

    def update(self, vocab: List[str], update_disk: bool = True):
        indices = sorted([self.w2i[w] for w in vocab if w in self.w2i])

        self.i2w = [self.i2w[i] for i in indices]
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        self.vectors = self.vectors[indices]

        if update_disk:
            print("Updating vocabulary + vectors on disk")
            backup_path = self.pt_path + ".bck"
            if not os.path.exists(backup_path):
                shutil.copy(self.pt_path, backup_path)

            torch.save((self.i2w, self.w2i, self.vectors, self.dim), self.pt_path)

    def _check_zip_exists(self):
        file_size = int(requests.head(GLOVE_URL).headers["Content-Length"])

        if os.path.exists(self.zip_path):
            first_byte = os.path.getsize(self.zip_path)

            if first_byte >= file_size:
                return

        os.makedirs(self.data_dir, exist_ok=True)
        print("Please download the GloVe embeddings first by running 'sh scripts/glove.sh'.")
        exit(1)

    def _build_vocabulary(self):
        if os.path.exists(self.pt_path):
            print("Reading pre-trained GloVe embeddings from disk")
            self.i2w, self.w2i, self.vectors, self.dim = torch.load(self.pt_path)
            return

        if not os.path.exists(self.txt_path):
            print("Extracting downloaded zip")
            with zipfile.ZipFile(self.zip_path) as zipf:
                zipf.extractall(self.data_dir)

        print("Building vocabulary + vectors")
        with open(self.txt_path, 'r') as txtf:
            voc_len, embed_dim = 0, None
            for wline in txtf:
                if embed_dim is None:
                    vec = wline.rstrip().split(" ")[1:]
                    if len(vec) > 2:
                        embed_dim = len(vec)
                        voc_len += 1
                else:
                    voc_len += 1
            txtf.seek(0)

            vocab = ["<pad>", "<unk>"]
            vectors = torch.zeros((voc_len + 2, embed_dim))
            v_loaded = 2

            for wline in tqdm(txtf, total=voc_len, disable=(not self.pbar)):
                wline = wline.rstrip().split(" ")
                word, vec = wline[0], wline[1:]
                vocab += [word]
                vectors[v_loaded] = Tensor([float(v) for v in vec])
                v_loaded += 1

        # set vector for <unk> to be the avg of all others
        vectors[1] = torch.mean(vectors[2:, :], dim=0)

        self.i2w = vocab
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.vectors = vectors
        self.dim = embed_dim

        print("Saving vocabulary + vectors to disk")
        torch.save((self.i2w, self.w2i, self.vectors, self.dim), self.pt_path)
