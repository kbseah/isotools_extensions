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

infile = "/data/CaiyuLab/aaeg-tra2b-isoforms/results/isotools/AaegyptiLVP_AGWG.VectorBase-68/transcriptome_isotools.with_orfs.pkl"
genome_fh = "/data/CaiyuLab/aaeg-tra2b-isoforms/results/get_genome/AaegyptiLVP_AGWG.VectorBase-68.fasta"
pfam_path = "/data/CaiyuLab/pfam"

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)
    pfam_a = isotools.domains.import_hmmer_models(pfam_path)
    gois = ["AAEL004293","AAEL020113","AAEL019814","AAEL009114","AAEL022912"]

    isoseq.add_hmmer_domains_parallel(
        domain_models=pfam_a,
        genome=genome_fh,
        query="FSM or SUBSTANTIAL",
        ref_query=False,
        region=None,
        gois=gois,
        cpus=8,
    )
