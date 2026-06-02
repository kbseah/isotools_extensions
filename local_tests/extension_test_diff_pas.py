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
outfile = "test_diff_lastexon_len.tsv"

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)
    tissue_groups = {'brain':['s269_1','s269_4'], 'gonad':['s269_2','s269_5'], 'carcass':['s269_3','s269_6']}
    logger.info("Performing test")
    df = isoseq.test_diff_lastexon_len(
        groups=tissue_groups,
        pairs=[('brain','carcass'),('gonad','carcass')],
        alpha=0.05,
        method='fdr_bh',
        query="(FSM or SUBSTANTIAL) and not UNSPLICED",
        min_coverage=20,
    )
    logger.info("Writing results")
    df.to_csv(outfile, sep="\t", index=False)
