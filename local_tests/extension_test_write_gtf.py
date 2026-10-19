#!/usr/bin/env python3
"""Test isotools_extensions with some data."""

import logging
from sys import stderr

import isotools
import isotools_extensions

# set up logger and set log level to debug, ensure that logging messages from
# isotools and isotools_extensions are printed to stderr
logger = logging.getLogger("isotools")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stderr)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

infile = "/data/CaiyuLab/aaeg-sex-isoforms/results/isotools/AaegyptiLVP_AGWG.VectorBase-71/transcriptome_isotools.with_orfs.pkl"
outfile = "test_write_gtf.gtf"

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)

    isoseq.write_gtf_with_cds(
        fn=outfile,
        query="FSM or (SUBSTANTIAL and not (INTERNAL_PRIMING or FRAGMENT))",
    )
