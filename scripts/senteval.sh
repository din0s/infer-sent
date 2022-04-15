#!/bin/bash
(
  if [ ! -d "./SentEval" ]; then
      # SentEval has not been detected as a gitmodule
      echo "SentEval toolkit not found, cloning..."
      git clone https://github.com/facebookresearch/SentEval.git
  elif [ ! -d "./SentEval/data" ]; then
      # SentEval has been detected as a gitmodule BUT not initialized
      echo "SentEval submodule not initialized, pulling..."
      git submodule update --init --recursive
  fi

  type module &> /dev/null && LISA=1
  if [ "$LISA" ]; then
      module purge
      module load 2021
      module load Anaconda3/2021.05
  else
      source "$(conda info --base)/etc/profile.d/conda.sh"
  fi

  conda activate atcs
  python -c 'import senteval' &> /dev/null || NO_SE=1
  if [ "$NO_SE" ]; then
      echo "SentEval module not found, setting up..."
      cd SentEval && python setup.py install
  else
      echo "SentEval module successfully loaded."
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
