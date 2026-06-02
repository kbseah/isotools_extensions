#!/usr/bin/env python3
"""Test isotools_extensions with some data."""

import isotools
import isotools_extensions
import logging

# from multiprocessing import freeze_support
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

if __name__ == "__main__":
    logger.info("Loading transcriptome")
    isoseq = isotools.Transcriptome.load(infile)

    mygroups = {
        "M": [
            "s269_1",
            "s269_2",
            "s269_3",
        ],
        "F": [
            "s269_4",
            "s269_5",
            "s269_6",
        ],
    }

    mygroups2 = {
        "MT": [
            "s269_2",
        ],
        "FO": [
            "s269_5",
        ],
    }

    fig, axs = isoseq["AAEL013863"].domains_figure(
        transcriptome=isoseq,
        groups=mygroups,
    )
    fig.savefig("test_AAEL013863.mygroups.png")

    fig, axs = isoseq["AAEL013863"].domains_figure(
        transcriptome=isoseq,
        groups=mygroups2,
    )
    fig.savefig("test_AAEL013863.mygroups2.png")

    fig, axs = isoseq["AAEL013863"].domains_figure(
        transcriptome=isoseq,
        groups=isoseq.groups(),
    )
    fig.savefig("test_AAEL013863.default_groups.png")
