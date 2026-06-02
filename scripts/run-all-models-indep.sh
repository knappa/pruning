#!/usr/bin/env bash

cd ~/pruning/ || exit

source venv/bin/activate

PARENT_DIR=~/pruning/data
NJOBS=$(nproc --ignore 2)
SEQ_ERR=0.00
ADO=0.00

for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-4state-0{} \
      --model DNA \
      --log
done

#####

for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-unphased-0{} \
      --model UNPHASED_DNA \
      --log
done


for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-cellphy-0{} \
      --model CELLPHY \
      --log
done

for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10z-0{} \
      --model GTR10Z \
      --log
done

for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs "$NJOBS" -I {} \
    pruning \
      --seqs "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output "$PARENT_DIR"/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10-0{} \
      --model GTR10 \
      --log
done
