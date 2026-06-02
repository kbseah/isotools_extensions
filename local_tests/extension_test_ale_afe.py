#!/usr/bin/env python3
"""Test isotools_extensions with some data."""

import isotools
import isotools_extensions
import logging

# from multiprocessing import freeze_support
from sys import stderr
from statsmodels.stats.multitest import multipletests

# set up logger and set log level to debug, ensure that logging messages from
# isotools and isotools_extensions are printed to stderr
logger = logging.getLogger("isotools_extensions")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stderr)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

infile = "/data/CaiyuLab/aaeg-sex-isoforms/results/isotools/AaegyptiLVP_AGWG.VectorBase-68/transcriptome_isotools.with_orfs.pkl"
# outfile = "test_ale_afe_output.tsv" # brain
outfile = "test_ale_afe_output.gonad.tsv"
# outfile = "test_ale_afe_output.testis.tsv"

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)

    logger.info("Performing test")
    df = isoseq.test_ale_afe(
        # groups={"brain": ["s269_1","s269_4"] ,"carcass": ["s269_3","s269_6"] },
        groups={"gonad": ["s269_2","s269_5"] ,"carcass": ["s269_3","s269_6"] },
        # groups={"testis": ["s269_2"] ,"carcass": ["s269_3"] },
        min_total=100,
        min_alt_fraction=0.01,
        query="not NOVEL_GENE and EXPRESSED",
    )

    logger.info("Performing multiple testing correction")
    df["padj"] = multipletests(df["pvalue"], method="fdr_bh")[1]
    df.sort_values("padj", inplace=True)

    logger.info("Writing results")
    df.to_csv(outfile, sep="\t", index=False)
