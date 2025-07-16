#!/usr/bin/env bash

for NSITES in 1000 10000
do
  seq 0 99 | parallel --line-buffer --sshlogin :,kirby,brouwer,riemann -I {} \
    ~/pruning/scripts/run-164frp-models-ssh-helper.sh "$NSITES" {}
done
