#!/usr/bin/env bash

SEQ_ERR=0.00
ADO=0.00
for NSITES in 1000 10000
do
  cd ~/pruning/data/diploid-sites-"$NSITES"-seq-err-"$SEQ_ERR"-ado-"$ADO" || exit;

  for idx in $(seq -w 0 99);
  do
    ~/pruning/convert.py diploid-0"$idx".phy
    ~/other-peoples-src/raxml-ng/bin/raxml-ng \
      --evaluate \
      --msa diploid-0"$idx".phy.haploid \
      --tree tree-0"$idx".nwk \
      --brlen scaled \
      --model GTGTR4+IU0 \
      --msa-format phylip
  done

  cd ..;
done