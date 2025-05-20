#!/usr/bin/env bash

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-4state-0{} \
      --model DNA \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased16-0{} \
      --model PHASED_DNA16 \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-phased16mp-0{} \
      --model PHASED_DNA16_MP \
      --log
done


#####

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-unphased-0{} \
      --model UNPHASED_DNA \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-cellphy-0{} \
      --model CELLPHY \
      --log
done

#SEQ_ERR=0.00
#ADO=0.00
#for NSITES in 1000 10000
#do
#  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
#    pruning \
#      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
#      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
#      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-cellphy-pi-0{} \
#      --model CELLPHY \
#      --optimize_freq_params \
#      --log
#done



SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10z-0{} \
      --model GTR10Z \
      --log
done


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-gtr10-0{} \
      --model GTR10 \
      --log
done


#### nested versions


SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning_stack \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-nested-0{} \
      --log
done

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 5 -I {} \
    pruning_stack \
      --ploidy 1 \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-nested-p1-0{} \
      --log
done



SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --jobs 15 -I {} \
    pruning_stack \
      --final_rp_norm \
      --seqs ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/diploid-0{}.phy \
      --tree ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/tree-0{}.nwk \
      --output ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO"/reconstructed-tree-nested-final_rp_norm-0{} \
      --log
done



#### half nested versions


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
