#!/usr/bin/env bash

cd ~/pruning/ || exit

source venv/bin/activate

PARENT_DIR=~/pruning/data
SEQ_ERR=0.00
ADO=0.00

NSITES="$1"
printf -v IDX "%03d" "$2"

pruning \
  --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$IDX".phy \
  --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$IDX".nwk \
  --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased_dna16_4_fp-"$IDX" \
  --model PHASED_DNA16_4 \
  --fixed_rate_params 0.839 0.112 2.239 0.600 3.119 0.560 \
  --fix_freq_params4 0.293 0.2 0.207 0.3 \
  --log

