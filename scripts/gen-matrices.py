#!/usr/bin/env python3

import csv
import argparse

parser = argparse.ArgumentParser(
    description="Compute log likelihood using the pruning algorithm"
)
parser.add_argument(
    "--seqs", type=str, required=True, help="sequence alignments in phylip format"
)
