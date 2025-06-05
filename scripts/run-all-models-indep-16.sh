#!/usr/bin/env bash

cd ~/pruning/ || exit

source venv/bin/activate

NJOBS=15

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs ~/pruning/data/temp/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/temp/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/temp/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased16-0{} \
      --model PHASED_DNA16 \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs ~/pruning/data/temp/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/temp/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/temp/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased16mp-0{} \
      --model PHASED_DNA16_MP \
      --log
done
