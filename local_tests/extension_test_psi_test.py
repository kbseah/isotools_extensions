#!/usr/bin/env python3
"""Test isotools_extensions with some data."""
import json
import logging
import isotools
import isotools_extensions
from isotools_extensions import psi_test
from sys import stderr

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
outfile = "test_psi_test.json"

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)
    # tissue_groups = {'brain':['s269_1','s269_4'], 'gonad':['s269_2','s269_5'], 'carcass':['s269_3','s269_6']}
    logger.info("Performing ALE test")
    for goi in ["AAEL012323", "AAEL020804", "AAEL012740"]:
        print(goi)
        print("ALE")
        res_ale = psi_test.ale_psi_by_group(
            isoseq, goi, isoseq.groups(), query="not UNSPLICED", mincov_frac=0.01, n_bootstrap=1,
        )
        if res_ale is not None:
            print(res_ale["actual"])
        print()
        print("APA")
        res_apa = psi_test.apa_psi_by_group(
            isoseq, goi, isoseq.groups(), query="not UNSPLICED", mincov_frac=0.01, n_bootstrap=1,
        )
        print(res_apa)
        print()
