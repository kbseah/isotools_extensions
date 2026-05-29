"""Implement tests based on LABRAT-style psi metric."""
import numpy as np
from isotools import Gene, Transcriptome
from isotools._transcriptome_stats import TESTS, _check_groups
from isotools_extensions.alt_pas import get_gene_last_exons


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
        (columns). The columns must already be sorted from proximal
        (lowest index) to distal (highest index).
    :returns: Array of length `coverage.shape[0]`, of the psi values
        for each sample.
    """
    # m values
    mm = (np.array(range(coverage.shape[1]))/(coverage.shape[1]-1))
    return (coverage * mm).sum(axis=1) / coverage.sum(axis=1)



