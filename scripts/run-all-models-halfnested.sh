#!/usr/bin/env bash

cd ~/pruning/ || exit

source venv/bin/activate

#### half nested version
# that is, two sequences
# 1. unphased -> gtr10z
# 2. cellphy -> gtr10z

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning_halfstack \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-half-nested-0{} \
      --log
done
