#!/bin/bash

DIR="./data/glove"
ZIP="$DIR/glove.840B.300d.zip"

if [ ! -f "$ZIP" ]; then
    mkdir -p "$DIR"
    cd "$DIR"
    echo "Downloading GloVe embeddings... (this may take a while)"
    wget -q http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip
    echo "Done!"
    cd ../../
else
    echo "GloVe embeddings already downloaded"
fi
