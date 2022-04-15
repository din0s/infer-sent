#!/bin/bash
(
  if [ ! -d "./SentEval" ]; then
      echo "SentEval toolkit not found, cloning..."
      git clone https://github.com/facebookresearch/SentEval.git

      cd SentEval
      python setup.py install
  fi
)
(
  cd SentEval/data/downstream
  FCOUNT=$(find ./* -maxdepth 0 -type d | wc -l)

  if [ "$FCOUNT" != 11 ]; then
      echo "Downloading SentEval datasets... (this may take a while)"
      ./get_transfer_data.bash
  else
      echo "SentEval datasets present."
  fi
)
