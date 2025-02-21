#!/usr/bin/env bash

~/pruning/cellcoal-runner-matpat.py \
  --cellcoal ~/pruning/cellcoal-modified/bin/cellcoal-1.3.0 \
  --ncells 100 \
  --nsamples 100 \
  --nsites 1000 \
  --eff_pop 10000 \
  --ado 0 \
  --amp_err_mean 0 \
  --amp_err_var 0 \
  --seq_err 0 \
  --somatic_mut_rate 1e-6 \
  --exp_growth_rate 1e-4 \
  --lin_rate_var 1.0 \
  --gamma 1.0 \
  --doublet_rate_mean 0.0 \
  --base_freqs 0.293 0.2 0.207 0.3 \
  --mut_matrix \
  0.0 2.1 0.3 5.6 \
  2.1 0.0 1.5 7.8 \
  0.3 1.5 0.0 1.4 \
  5.6 7.8 1.4 0.0


# seq_err 0.00, 0.01, 0.05, 0.10
# ado 0.00, 0.10, 0.25, 0.50

# GTR from https://dapogon.github.io/cellcoal/cellcoal.manual.v1.1.html
# 0.0 2.1 0.3 5.6 \
# 2.1 0.0 1.5 7.8 \
# 0.3 1.5 0.0 1.4 \
# 5.6 7.8 1.4 0.0 #

# GTnR from https://dapogon.github.io/cellcoal/cellcoal.manual.v1.1.html
# 0.0 2.1 0.3 5.6 \
# 1.9 0.0 1.5 7.8 \
# 2.1 3.7 0.0 1.4 \
# 0.8 2.4 0.9 0.0


for idx in $(seq -w 0 99);
  do
    python ~/pruning/nex-to-phy.py --nex tree-0"$idx"-nexus-files/16state.nex --output diploid-0"$idx".phy;
    cp tree-0"$idx"-nexus-files/tree-no-outgcell.nwk tree-0"$idx".nwk;
  done
