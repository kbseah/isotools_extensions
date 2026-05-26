import logging

import pyhmmer
from isotools import Transcriptome
from isotools._utils import genomic_position
from isotools.domains import get_hmmer_sequences

logger = logging.getLogger("isotools")


def add_hmmer_domains_parallel(
    self: Transcriptome,
    domain_models: tuple,
    genome: str,
    query: str | bool = True,
    ref_query: str | bool = False,
    region: str | tuple | None = None,
    min_coverage: int | None = None,
    max_coverage: int | None = None,
    gois: list | None = None,
    cpus: int = 8,
):
    """Align domains to protein sequences and add to transcript annotations.

    Does the same job as isotools.domains.add_hmmer_domains, but uses
    parallelization implemented in pyhmmer to speed up the alignment. Requires
    pyhmmer >=0.12.0. Modifies Transcriptome object in-place.

    :param self: isotools.Transcriptome object.
    :param domain_models: Domain models and metadata, imported by
        `isotools.domains.import_hmmer_models`
    :param genome: Path to genome Fasta file.
    :param query: Query string to select the isotools transcripts, or
        True/False to include/exclude all transcripts.
    :param ref_query: Query string to select the reference transcripts, or
        True/False to include/exclude all transcripts.
    :param region: The region to be considered. Either a string
        "chr:start-end", or a tuple (chr,start,end). Start and end is optional.
    :param min_coverage: The minimum coverage threshold. Transcripts with less
        reads in total are ignored.
    :param max_coverage: The maximum coverage threshold. Transcripts with more
        reads in total are ignored.
    :param gois: List of gene IDs to include. If None, all genes are included.
    :param cpus: The number of CPUs to use for parallel processing.
    """
    metadata, models = domain_models
    # Boolean doesn't work with get_hmmer_sequences, so convert to query string
    # that includes all transcripts
    if isinstance(query, bool) and query:
        query = "FSM or (not FSM)"

    # 1) get the protein sequences
    logger.info("Extracting protein sequences from genome...")
    pipeline = pyhmmer.plan7.Pipeline(models[0].alphabet)
    sequences, seq_ids = get_hmmer_sequences(
        self,
        genome,
        models[0].alphabet,
        query=query,
        ref_query=ref_query,
        region=region,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        gois=gois,
        progress_bar=False,
    )
    logger.info("Found %d sequences to align domains to.", len(sequences))

    # 2) align domain models to sequences
    logger.info("Aligning domain models to sequences...")
    sequences_block = pyhmmer.easel.DigitalSequenceBlock(
        alphabet=sequences[0].alphabet,
        iterable=sequences,
    )
    hmmsearch_out = list(
        pyhmmer.hmmer.hmmsearch(models, sequences_block, cpus=cpus),
    )
    hits = {h.query.accession: h for h in hmmsearch_out}

    # 3) sort domains by gene/source/transcript
    domains = {}
    for pfam_acc, hL in hits.items():
        infos = metadata.loc[pfam_acc]
        for h in hL:
            seq_nr = int(h.name)
            for domL in h.domains:
                ali = domL.alignment
                transcript_pos = (ali.target_from * 3, ali.target_to * 3)
                domains.setdefault(seq_nr, []).append(
                    (
                        pfam_acc,
                        infos["ID"],
                        infos["TP"],
                        transcript_pos,
                        h.score,
                        h.pvalue,
                    ),
                )

    # 4) add domains to transcripts
    logger.info("Adding domains to transcripts...")
    dom_count = [0, 0]
    tr_count = [0, 0]
    for seq_nr, domL in domains.items():
        for gene_id, reference, transcript_id in seq_ids[seq_nr]:
            gene = self[gene_id]  # modify Transcriptome object in place
            transcript = (
                gene.ref_transcripts[transcript_id]
                if reference
                else gene.transcripts[transcript_id]
            )
            # get the genomic position of the domain boundaries
            orf = sorted(
                gene.find_transcript_positions(
                    transcript_id,
                    transcript.get("CDS", transcript.get("ORF"))[:2],
                    reference=reference,
                ),
            )
            pos_map = genomic_position(
                [p + orf[0] for dom in domL for p in dom[3]],
                transcript["exons"],
                gene.strand == "-",
            )
            trdom = tuple(
                (
                    *dom[:4],
                    (pos_map[dom[3][0] + orf[0]], pos_map[dom[3][1] + orf[0]]),
                    *dom[4:],
                )
                for dom in domL
            )
            transcript.setdefault("domain", {})["hmmer"] = trdom
            tr_count[reference] += 1
            dom_count[reference] += len(domL)
    logger.info(
        "Found domains at %d loci for %d reference transcripts and %d loci for %d long read transcripts.",
        dom_count[1],
        tr_count[1],
        dom_count[0],
        tr_count[0],
    )
