#!/usr/bin/env bash

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased4-0{} \
      --model PHASED_DNA4 \
      --method L-BFGS-B \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased16-0{} \
      --model PHASED_DNA16 \
      --method L-BFGS-B \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased16mp-0{} \
      --model PHASED_DNA16_MP \
      --method L-BFGS-B \
      --log
done


#####

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-unphased-0{} \
      --model UNPHASED_DNA \
      --method L-BFGS-B \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-cellphy-0{} \
      --model CELLPHY \
      --method L-BFGS-B \
      --log
done

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-cellphy-pi-0{} \
      --model CELLPHY \
      --optimize_freqs \
      --method L-BFGS-B \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10z-0{} \
      --model GTR10Z \
      --method L-BFGS-B \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10-0{} \
      --model GTR10 \
      --method L-BFGS-B \
      --log
done