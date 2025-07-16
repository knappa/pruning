#!/usr/bin/env bash

cd ~/pruning/ || exit

source venv/bin/activate

PARENT_DIR=~/pruning/data

NSITES="$1"
printf -v IDX "%03d" "$2"
SEQ_ERR="$3"
ADO="$4"

pruning \
  --seqs "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$IDX".phy \
  --tree "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$IDX".nwk \
  --output "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased_dna16-"$IDX" \
  --model PHASED_DNA16 \
  --log

