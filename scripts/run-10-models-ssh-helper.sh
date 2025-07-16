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
  --output "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-unphased-"$IDX" \
  --model UNPHASED_DNA \
  --log

pruning \
  --seqs "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$IDX".phy \
  --tree "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$IDX".nwk \
  --output "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-cellphy-"$IDX" \
  --model CELLPHY \
  --log

pruning \
  --seqs "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$IDX".phy \
  --tree "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$IDX".nwk \
  --output "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10z-"$IDX" \
  --model GTR10Z \
  --log

pruning \
  --seqs "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$IDX".phy \
  --tree "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$IDX".nwk \
  --output "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10-"$IDX" \
  --model GTR10 \
  --log

pruning \
  --seqs "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-"$IDX".phy \
  --tree "$PARENT_DIR"/diploid-jp-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-"$IDX".nwk \
  --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased_dna16_4-"$IDX" \
  --model PHASED_DNA16_4 \
  --log


