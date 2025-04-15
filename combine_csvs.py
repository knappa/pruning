#!/usr/bin/env python3

import csv
import os

import numpy as np

files = sorted(
    [file for file in os.listdir(".") if file.endswith(".csv") and file.startswith("recon")]
)
with open(files[0], "r") as file:
    headers = next(file).strip().split(",")
    data = [list(map(float, next(file).strip().split(",")))]
for fn in files[1:]:
    with open(fn, "r") as file:
        next(file)
        data.append(list(map(float, next(file).strip().split(","))))
data = np.array(data)

with open("combined-fit-stats.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(headers)
    csvwriter.writerows(data)
