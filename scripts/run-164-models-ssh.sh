#!/usr/bin/env bash

#cd ~/pruning/ || exit
#
#source venv/bin/activate

# PARENT_DIR=~/pruning/data
# SEQ_ERR=0.00
# ADO=0.00

for NSITES in 1000 10000
do
  seq -w 0 99 | parallel --line-buffer --sshlogin kirby,brouwer,: -I {} \
    ~/pruning/scripts/run-164-models-ssh-helper.sh "$NSITES" {}
done
