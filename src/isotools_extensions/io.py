import logging

from isotools import Gene
from isotools import Transcriptome
from isotools._transcriptome_io import openfile

logger = logging.getLogger("isotools")


def convert_transcript_genomic_coords(
    exons: list[tuple[int, int]],
    tx_start: int,
    tx_end: int,
) -> list[tuple[int, int]]:
    """Convert transcript interval to genomic coordinates, given exon coords.

    If an interval within a transcript falls across exon boundaries, then
    the corresponding genomic interval is discontinuous. Given an interval
    in transcript-based positions, and the genomic coordinates of the exons,
    convert to genomic interval or intervals. Strand is not accounted for.

    :param exons: list of (start, end) tuples, 0-based, end-exclusive
    :param tx_start, tx_end: transcript coordinates, 0-based, end-exclusive
    :returns: list of (genomic_start, genomic_end) tuples (0-based, end-exclusive)
    """
    result = []
    tx_pos = 0
    for exon_start, exon_end in exons:
        exon_len = exon_end - exon_start
        exon_tx_start = tx_pos
        exon_tx_end = tx_pos + exon_len
        # Find overlap between transcript interval and this exon
        overlap_start = max(tx_start, exon_tx_start)
        overlap_end = min(tx_end, exon_tx_end)
        if overlap_start < overlap_end:
            g_start = exon_start + (overlap_start - exon_tx_start)
            g_end = exon_start + (overlap_end - exon_tx_start)
            result.append((g_start, g_end))
        tx_pos += exon_len
        if tx_pos >= tx_end:
            break
    return result


def get_cds_frames(lengths: list[int], strand: str) -> list[int]:
    """Get frame of each segment of a multi-exon CDS feature.

    :param lengths: Length of each segment, sorted in genomic order.
    :param strand: Feature is on forward (`+`) or reverse (`-`) strand.
    :returns: List of frames for each segment.
    """
    if sum(lengths) % 3 != 0:
        logger.warning("CDS total length must be multiple of 3")
        return None
    if strand == "-":
        lengths = lengths[::-1]
    elif strand != "+":
        logger.warning("strand must be either '+' or '-'")
        return None
    # each segment's frame depends on total length of preceding segments
    # length of last segment doesn't matter
    cumul_lens = [sum(lengths[0:i]) for i in range(1, len(lengths))]
    remainders = [i % 3 for i in cumul_lens]
    # convert remainder to frame
    frames = [(3 - i) % 3 for i in remainders]
    # first frame must be zero
    frames = [0, *frames]
    if strand == "-":
        return frames[::-1]
    else:
        return frames


def get_cds_intervals(gene: Gene, trid: int) -> tuple[list, list, list]:
    """Get CDS intervals from isotools transcript.

    All coordinates are 0-based end-exclusive.

    :param gene: isotools.Gene object
    :param trid: Transcript ID
    :returns: Tuple of CDS, start codon, and stop codon interval(s) as lists of
        tuples.
    """
    transcript = gene.transcripts[trid]
    if "ORF" not in transcript:
        logger.debug("No ORF reported in gene %s transcript %s", gene.id, str(trid))
        return None
    orfs, orfe = transcript["ORF"][0:2]
    cds_intervals = []
    for exons, exone in transcript["exons"]:
        if exone < orfs:
            continue
        if exons > orfe:
            continue
        # kiv orf may be contained in one exon
        mystart = orfs if (orfs > exons and orfs < exone) else exons
        myend = orfe if (orfe > exons and orfe < exone) else exone
        cds_intervals.append((mystart, myend))
    cds_len = sum(i[1] - i[0] for i in cds_intervals)
    if cds_len % 3 != 0:
        logger.warning("CDS total length should be multiple of 3; gene %s transcript %s", gene.id, str(trid))
        return None, None, None
    # get start, stop codon and strip stop codon
    # isotools includes stop codon in ORF but in GTF standard stop codon is
    # excluded from CDS feature
    if gene.strand == "+":
        start_codon = convert_transcript_genomic_coords(cds_intervals, 0, 3)
        stop_codon = convert_transcript_genomic_coords(
            cds_intervals, cds_len - 3, cds_len
        )
        cds_intervals = convert_transcript_genomic_coords(cds_intervals, 0, cds_len - 3)
    elif gene.strand == "-":
        start_codon = convert_transcript_genomic_coords(
            cds_intervals, cds_len - 3, cds_len
        )
        stop_codon = convert_transcript_genomic_coords(cds_intervals, 0, 3)
        cds_intervals = convert_transcript_genomic_coords(cds_intervals, 3, cds_len)
    return (cds_intervals, start_codon, stop_codon)


def cds_intervals_to_gtf(gene: Gene, trid: int, source: str = "isoseq") -> list[tuple]:
    """Report CDS, start_codon and stop_codon features in GTF format.

    :param gene: isotools.Gene object
    :param trid: Transcript index
    :param source: Value for the `source` field of feature.
    :returns: List of tuples of GTF features.
    """
    out = []
    if "ORF" not in gene.transcripts[trid]:
        return []
    # Assumes that exon intervals are already sorted by isotools!
    cds_intervals, start_codon, stop_codon = get_cds_intervals(gene, trid)
    cds_frames = get_cds_frames([i[1] - i[0] for i in cds_intervals], gene.strand)
    if cds_intervals is None or cds_frames is None:
        return []
    # start/stop codons may span exon-exon boundary
    start_codon_frames = get_cds_frames([i[1] - i[0] for i in start_codon], gene.strand)
    stop_codon_frames = get_cds_frames([i[1] - i[0] for i in stop_codon], gene.strand)
    if gene.name:
        common_attr = f'gene_id "{gene.id!s}"; gene_name "{gene.name!s}"; transcript_id "{gene.id!s}_{trid!s}";'
    else:
        common_attr = f'gene_id "{gene.id!s}"; transcript_id "{gene.id!s}_{trid!s}";'
    # <seqname> <source> <feature> <start> <end> <score> <strand> <frame> [attributes] [comments]
    for intervals, frames, feature_type in [
        (cds_intervals, cds_frames, "CDS"),
        (start_codon, start_codon_frames, "start_codon"),
        (stop_codon, stop_codon_frames, "stop_codon"),
    ]:
        for i, (interval, frame) in enumerate(zip(intervals, frames, strict=True)):
            out.append(
                [
                    gene.chrom,
                    source,
                    feature_type,
                    interval[0] + 1,  # convert to 1-based
                    interval[1],
                    ".",
                    gene.strand,
                    frame,
                    common_attr
                    + f' {feature_type.lower()!s}_id "{gene.id!s}_{trid!s}_{feature_type!s}_{i!s}";',
                ]
            )
    return out


def write_gtf_with_cds(
    self: Transcriptome,
    fn: str,
    source: str = "isotools",
    gzip: bool = False,
    **filter_args,
) -> None:
    """Export transcripts and CDS features in GTF format to a file.

    Use with isotools.Transcriptome object after running .add_orf_prediction.

    :param fn: The filename to write the gtf.
    :param source: String for the source column of the gtf file.
    :param region: Specify genomic region to export to gtf. If omitted, export whole genome.
    :param gzip: Compress the output as gzip.
    :param filter_args: Specify transcript filter query.
    """
    with openfile(fn, gzip) as f:
        logger.info("writing %sgtf file to %s", "gzip compressed " if gzip else "", fn)
        for gene, transcript_ids, _ in self.iter_transcripts(
            genewise=True,
            **filter_args,
        ):
            for trid in transcript_ids:
                lines = gene._to_gtf(transcript_ids=[trid], source=source)
                cds_lines = cds_intervals_to_gtf(gene, trid=trid, source=source)
                f.write(
                    "\n".join(
                        (
                            "\t".join(str(field) for field in line)
                            for line in [*lines, *cds_lines]
                        )
                    )
                    + "\n"
                )
