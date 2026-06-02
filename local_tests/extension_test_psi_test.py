#!/usr/bin/env python3
"""Test isotools_extensions with some data."""
import json
import logging
import isotools
import isotools_extensions
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
outfile_apa = "test_psi_test.apa.tsv"
outfile_ale = "test_psi_test.ale.tsv"

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)
    mygroups = {'brain':['s269_1','s269_4'], 'gonad':['s269_2','s269_5'], 'carcass':['s269_3','s269_6']}
    logger.info("Performing ALE test")
    gois = ["AAEL012323", "AAEL020804", "AAEL012740"]
    out_ale = isoseq.test_ale_psi(
        mygroups, contrast=["gonad","carcass"], gois=gois,
    )
    out_ale.to_csv(outfile_ale, sep="\t", index=False)
    out_apa = isoseq.test_apa_psi(
        mygroups, contrast=["gonad","carcass"], gois=gois,
    )
    out_apa.to_csv(outfile_apa, sep="\t", index=False)
