#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
print(args)

directory <- args[1]
count <- strtoi(args[2])

library(ape)
library(TreeDist)
library(stringr)
library(phangorn)

df <- data.frame(matrix(
  ncol = 10,
  nrow = count, dimnames = list(
    NULL,
    c(
      "id", "SPR", "RF", "wRF", "KF", "path", "treedist.symmetric",
      "treedist.branch.score", "treedist.path", "treedist.quadratic.path"
    )
  )
))



for (item in 1:count) {
  tree <- read.tree(
    paste(directory,
      "/diploid-",
      str_pad(item - 1, 3, pad = "0"),
      ".phy.haploid.raxml.bestTree",
      sep = ""
    )
  )
  true_tree <- read.tree(
    paste(
      directory, "/tree-", str_pad(item - 1, 3, pad = "0"), ".nwk",
      sep = ""
    )
  )

  df[item, 1] <- paste(
    "raxml-",
    item,
    sep = ""
  )
  df[item, 2] <- SPR.dist(tree, true_tree)
  df[item, 3] <- RF.dist(tree, true_tree)
  df[item, 4] <- wRF.dist(tree, true_tree)
  df[item, 5] <- KF.dist(tree, true_tree)
  df[item, 6] <- path.dist(tree, true_tree)

  td <- treedist(tree, true_tree)
  df[item, 7] <- td["symmetric.difference"]
  df[item, 8] <- td["branch.score.difference"]
  df[item, 9] <- td["path.difference"]
  df[item, 10] <- td["quadratic.path.difference"]
}


write.csv(df, paste(
  directory,
  "/tree-stats-raxml-output.csv",
  sep = ""
), row.names = FALSE)
