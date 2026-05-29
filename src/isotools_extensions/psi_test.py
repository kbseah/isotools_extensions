"""Implement tests based on LABRAT-style psi metric."""
import logging
import numpy as np
from isotools import Transcriptome
from isotools_extensions.alt_pas import get_gene_last_exons


logger = logging.getLogger("isotools")

def gene_coverage_sum_samplewise(
    coverage:np.ndarray,
    sample_groups:dict,
    sort_descending:bool=False
) -> tuple:
    """Sum grouped coverage by sample groups.

    :param coverage: 2D coverage counts array, with samples as rows and
        transcripts as columns, following convention of `isotools.Gene.coverage`
    :param sample_groups: Groups of samples, keyed by group name, values are
        lists of sample indices.
    :param sort_descending: Sort sample group names in reverse order if True,
        else in ascending order.
    :returns: Tuple of: (1) numpy.ndarray of coverage counts per sample group
        (rows) per transcript (columns). Number of columns should equal input
        `coverage`. (2) sorted list of keys from `sample_groups`.
    """
    keys = sorted(sample_groups, reverse=sort_descending)
    counts_sum = np.asarray([
        coverage[sample_groups[key], :].sum(axis=0)
        for key in keys
    ])
    return counts_sum, keys


def gene_coverage_sum_transcriptwise(
    coverage:np.ndarray,
    transcript_groups:dict,
    sort_descending:bool=False
)->tuple:
    """Sum grouped coverage by transcript groups.

    :param coverage: 2D coverage counts array, with samples as rows and
        transcripts as columns, following convention of `isotools.Gene.coverage`
    :param transcript_groups: Groups of transcripts, keyed by group name,
        values are lists of transcript indices.
    :param sort_descending: Sort transcript group names in reverse order if
        True, else in ascending order.
    :returns: Tuple of: (1) numpy.ndarray of coverage counts per sample (rows)
        per group (columns). Number of rows should equal input `coverage`. (2)
        sorted list of keys from `transcript_groups`.
    """
    keys = sorted(transcript_groups, reverse=sort_descending)
    counts_sum = np.asarray([
        coverage[:,transcript_groups[key]].sum(axis=1)
        for key in keys
    ]).T
    return counts_sum, keys


def filter_transcripts_by_coverage(
    coverage:np.ndarray,
    mincov_frac:float = 0.01,
)->tuple:
    """Filter matrix of coverage counts by minimum coverage fraction.

    Filters the columns of the input matrix, i.e. the column sum must be over
    the cutoff to be included. For example, if mincov_frac=0.1 and the coverage
    total is 100, the column sum must be >10 to be included.

    :param coverage: 2D coverage counts array, with samples as rows and
        transcripts as columns, following convention of `isotools.Gene.coverage`
    :param mincov_frac: Minimum fraction of total (exclusive) for a column to
        be included.
    :returns: Tuple of: (1) Filtered counts matrix, and (2) indices of the
        input column which passed the filter.
    """
    if mincov_frac > 1 or mincov_frac < 0:
        raise ValueError("`mincov_frac` should be between 0 and 1")
    mask = coverage.sum(axis=0) > mincov_frac * coverage.sum()
    return coverage[:, mask], np.where(mask)[0]


def coverage_matrix_to_psi(
    coverage:np.ndarray,
) -> np.ndarray:
    """Calculate LABRAT-style psi from counts matrix.

    :param coverage: 2D matrix of coverage per sample (row) per ALE/APA
        (columns). The columns must already be sorted from proximal (lowest
        index) to distal (highest index).
    :returns: Array of length `coverage.shape[0]`, of the psi values
        for each sample.
    """
    # m values
    mm = (np.array(range(coverage.shape[1]))/(coverage.shape[1]-1))
    return (coverage * mm).sum(axis=1) / coverage.sum(axis=1)


def bootstrap_count_matrix(obs:list|np.ndarray, rng=None)->np.ndarray:
    """Resample with replacement a count matrix.

    :param obs: 2D array of integer counts.
    :param rng: Random number generator to use, passed to `np.random.default_rng`.
    :returns: Resampled matrix of the same shape as `obs`.
    """
    obs = np.asarray(obs)
    if obs.ndim != 2:
        raise ValueError("obs must be a 2D array")
    if np.any(obs < 0):
        raise ValueError("obs must contain nonnegative counts")

    total = obs.sum()
    if total == 0:
        return np.zeros_like(obs)

    rng = np.random.default_rng(rng)
    p = obs.ravel() / total
    boot = rng.multinomial(total, p)
    return boot.reshape(obs.shape)


def ale_psi_by_group(
    transcriptome:Transcriptome,
    gene_id:str,
    groups:dict,
    query:str="not FRAGMENT and not UNSPLICED",
    n_bootstrap:int=10,
    mincov_frac:float=0.01,
)->dict | None:
    # Get indices of samples and groups
    sample2idx = {s: idx for (idx, s) in enumerate(transcriptome.samples)}
    groups2idx = {g : [sample2idx[s] for s in samples] for g, samples in groups.items()}

    # Get junction coordinates for exons of interest
    trids = transcriptome[gene_id].filter_transcripts(query)
    if len(trids) <= 1:
        logger.warning("No transcripts after filter query")
        return None
    junc2trid = {
        junc : [t for t in transcripts if t in trids]
        for junc, transcripts in get_gene_last_exons(transcriptome[gene_id]).items()
    }

    # Sum counts per last exon and sort from proximal to distal
    counts_junc, juncs = gene_coverage_sum_transcriptwise(
        transcriptome[gene_id].coverage,
        junc2trid,
        sort_descending=(transcriptome[gene_id].strand == "-"),
    )
    if len(juncs) <= 1:
        logger.warning("No junctions found for %s??", gene_id)
        return None

    # Sum counts by sample group
    counts_junc_group, groups = gene_coverage_sum_samplewise(counts_junc, groups2idx)

    # Filter only those junctions that represent at least x% of total
    counts_filt, filt_idx = filter_transcripts_by_coverage(counts_junc_group, mincov_frac)
    if len(filt_idx) <= 1:
        logger.warning(
            "Zero or only one exon after coverage filtering for gene %s",
            gene_id,
        )
        return None
    juncs_filt = [juncs[i] for i in filt_idx]
    psi = coverage_matrix_to_psi(counts_filt)

    out = {}
    out["actual"] = {
        "counts" : counts_filt,
        "juncs" : { j: junc2trid[j] for j in juncs_filt },
        "psi" : psi,
        "juncs_ordered" : juncs_filt,
        "groups_ordered" : groups,
    }

    # bootstrapping
    out["bootstraps"] = []
    for _i in range(n_bootstrap):
        bs = bootstrap_count_matrix(counts_filt)
        bs_psi = coverage_matrix_to_psi(bs)
        out["bootstraps"].append({"counts" : bs, "psi": bs_psi})

    return out
