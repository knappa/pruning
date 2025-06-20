#!/usr/bin/env bash

cd ~/pruning/ || exit

source venv/bin/activate

PARENT_DIR=~/pruning/data
SEQ_ERR=0.00
ADO=0.00

NSITES="$0"
printf -v SEQ "%03d" "$1"

pruning \
  --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$SEQ".phy \
  --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$SEQ".nwk \
  --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased_dna16_4-"$SEQ" \
  --model PHASED_DNA16_4 \
  --log

