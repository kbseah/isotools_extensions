from collections import defaultdict
from itertools import combinations

import isotools._utils
import numpy as np
import pandas as pd
from isotools._transcriptome_stats import TESTS, _check_groups
from scipy.signal import find_peaks


def pileup_to_smoothed(pileup, smooth_window: int = 31):
    """Smooth a coverage pileup

    :param pileup: Dict of coverage values keyed by genomic coordinate
    :param smooth_window: Window size for smoothing function
    :returns: Pileup values as array, coordinates as array, and smoothed values
    """
    # X coordinates of the pileup
    # Add flanking buffer of 1*smooth_window to avoid smoothed max falling
    # right at the first or last position and not being found by find_peaks
    coords = range(
        min(pileup.keys()) - smooth_window, max(pileup.keys()) + 1 + smooth_window
    )
    pileup_arr = [pileup.get(pos, 0) for pos in coords]
    smoothed = isotools._utils.smooth(np.array(pileup_arr), smooth_window)
    return (pileup_arr, coords, smoothed)


def get_transcript_terminal_peaks(
    gene,
    trid: int,
    which="PAS",
    total: bool = False,
    smooth_window: int = 31,
    prominence: int = 2,
):
    """Get PAS/TSS peaks for an isotools transcript

    Unlike default isotools behavior, this calls all peaks without choosing a
    unified consensus for each transcript.

    :param gene: isotools.Gene object
    :param trid: Transcript index
    :param which: Either "PAS" or "TSS"
    :param total: Sum pileups for all samples if True, else plot samples separately
    :param smooth_window: Window size for smoothing function
    :param prominence: Minimum peak prominence to retain
    """
    pileup, coords, smoothed, peaks = {}, {}, {}, {}
    if total:
        pileup_sum = {}
        for sample in gene.transcripts[trid][which]:
            for pos, cov in gene.transcripts[trid][which][sample].items():
                pileup_sum[pos] = pileup_sum.get(pos, 0) + cov
        pileup["total"], coords["total"], smoothed["total"] = pileup_to_smoothed(
            pileup_sum, smooth_window
        )
        peaks["total"] = translate_peaks_offset(
            find_peaks(smoothed["total"], prominence=(prominence, None)),
            min(coords["total"]),
        )
    else:
        for sample in gene.transcripts[trid][which]:
            pileup[sample], coords[sample], smoothed[sample] = pileup_to_smoothed(
                gene.transcripts[trid][which][sample], smooth_window
            )
            peaks[sample] = translate_peaks_offset(
                find_peaks(smoothed[sample], prominence=(prominence, None)),
                min(coords[sample]),
            )
    return (pileup, coords, smoothed, peaks)


def assign_to_closest_peak(pos, peaks):
    # Assumes peaks and pos have already been converted to genomic coordinates
    dist = [abs(p - pos) for p in peaks]
    closest_index = np.argmin(dist)
    return closest_index, dist[closest_index]


def translate_peaks_offset(peaks, offset):
    # Translate array indices back to genomic coordinates
    return (
        np.array([p + offset for p in peaks[0]]),
        {
            "prominences": peaks[1]["prominences"],
            "left_bases": np.array([lb + offset for lb in peaks[1]["left_bases"]]),
            "right_bases": np.array([rb + offset for rb in peaks[1]["right_bases"]]),
        },
    )


def get_gene_terminal_peaks(
    gene,
    trids: list = None,
    which="PAS",
    smooth_window: int = 31,
    prominence: int = 2,
):
    """Get PAS/TSS peaks for an isotools Gene, summing across all transcripts

    Unlike default isotools behavior, this calls all peaks without choosing a
    unified consensus for each transcript.

    :param gene: isotools.Gene object
    :param trids: List of transcript IDs to include (as ints); if None, include
        all transcripts
    :param which: Either "PAS" or "TSS"
    :param smooth_window: Window size for smoothing function
    :param prominence: Minimum peak prominence to retain
    """
    if which not in ["PAS", "TSS"]:
        raise ValueError("Option `which` must be either PAS or TSS only")
    # Check that trids are valid
    if trids is None:
        trids = list(range(len(gene.transcripts)))
    elif min(trids) < 0 or max(trids) >= len(gene.transcripts):
        raise ValueError("At least one transcript ID is out of range")
    pileup, coords, smoothed, peaks, peak_assignments = {}, {}, {}, {}, {}
    pileup_sum = {}
    for trid, transcript in enumerate(gene.transcripts):
        if trid in trids:
            for sample in transcript[which]:
                for pos, cov in transcript[which][sample].items():
                    pileup_sum[pos] = pileup_sum.get(pos, 0) + cov
    pileup["total"], coords["total"], smoothed["total"] = pileup_to_smoothed(
        pileup_sum, smooth_window
    )
    # peaks coordinates are indices of coords
    peaks["total"] = find_peaks(smoothed["total"], prominence=(prominence, None))
    peaks["total"] = translate_peaks_offset(peaks["total"], min(coords["total"]))
    # If no peaks found, return None
    if len(peaks["total"][0]) == 0:
        return pileup, coords, smoothed, None, None
    # We cannot use left_base and right_base from find_peaks directly, because
    # the intervals overlap, see https://github.com/scipy/scipy/issues/19232
    # Assign counts to closest peaks
    peak_assignments["total"] = defaultdict(
        lambda: defaultdict(int)
    )  # peak, sample -> count
    for trid, transcript in enumerate(gene.transcripts):
        if trid in trids:
            for sample in transcript[which]:
                for pos in transcript[which][sample]:
                    closest_index, distance = assign_to_closest_peak(
                        pos, peaks["total"][0]
                    )
                    if distance <= smooth_window:
                        peak_assignments["total"][closest_index][sample] += transcript[
                            which
                        ][sample][pos]
    return pileup, coords, smoothed, peaks, peak_assignments


def get_gene_last_exons(gene):
    """Get last exons for all transcripts of a gene

    :param gene: isotools.Gene object
    :returns: Dict mapping transcript IDs to start positions of last exons
    """
    last_exons = {}
    for trid, transcript in enumerate(gene.transcripts):
        last_exon_start = (
            transcript["exons"][-1][0]
            if gene.strand == "+"
            else transcript["exons"][0][1]
        )
        last_exons[last_exon_start] = last_exons.get(last_exon_start, []) + [trid]
    return last_exons


def test_alternative_pas(
    self,
    groups: dict,
    smooth_window: int = 31,
    prominence: int = 2,
    min_total: int = 100,
    min_alt_fraction: float = 0.01,  # different from Isotools default
    min_n: int = 5,
    min_sa: float = 0.51,
    test="auto",  # either string with test name or a custom test function
    **kwargs,
):
    """Identify and test alternative PAS within an isotools Gene

    Isotools chooses a unified PAS per transcript by default. However,
    transcripts are defined by their internal splice sites, so a single
    transcript may have multiple PAS, not captured by the unified PAS. This
    function first groups transcripts by their last exon, then calls all PAS
    peaks for all transcripts with that last exon, and tests for differential
    APA between groups, without relying on a unified consensus for each
    transcript.

    Output columns are same as those from isotools._transcriptome_stats.altsplice_test, except for:
     * "last_exon_start" - start position of the last exon shared by the transcripts tested
     * "trids" - list of transcript IDs tested
     * "start" - position of the first PAS peak tested
     * "end" - position of the second PAS peak tested

    Column "splice_type" is always "PAS" in this function.

    :param self: isotools.Transcriptome object
    :param groups: Dict mapping sample names to group names
    :param smooth_window: Window size for smoothing function in peak calling
    :param prominence: Minimum peak prominence to retain in peak calling
    :param min_total: Minimum total counts across all samples to consider a PAS
    :param min_alt_fraction: Minimum fraction of counts at an alternative PAS
        to consider it for testing
    :param min_n:
    :param min_sa:
    :param test: Either "auto" to use isotools default test, or a custom test function
    :param **kwargs: Additional keyword arguments passed to iter_genes
    :returns: DataFrame with test results for alternative PAS events
    """
    groupnames, groups_arr, grp_idx = _check_groups(self, groups)
    # Choose appropriate test
    if isinstance(test, str):
        if test == "auto":
            test = (
                TESTS["betabinom_lr"]
                if min(len(group) for group in groups_arr[:2]) > 1
                else TESTS["proportions"]
            )
        else:
            try:
                test = TESTS[test]
            except KeyError as e:
                e.add_note(f"Test name {test} not found")
                raise
    if min_sa < 1:
        min_sa *= sum(len(group) for group in groups_arr[:2])
    # Store results here
    res = []
    for gene in self.iter_genes(**kwargs):
        # For each last exon, get PAS peaks and test for differential usage
        trids_by_last_exon = get_gene_last_exons(gene)
        for last_exon in trids_by_last_exon:
            pileup, coords, smoothed, peaks, peak_assignments = get_gene_terminal_peaks(
                gene=gene,
                trids=trids_by_last_exon[last_exon],
                which="PAS",
                smooth_window=smooth_window,
                prominence=prominence,
            )
            # No peaks found, likely because coverage is too low
            if peaks is None:
                continue
            # TODO: report number of reads not assigned to a called peak
            # Count coverage per PAS peak per group/sample
            group_cov = defaultdict(lambda: defaultdict(int))  # pas, group -> [cov]
            for i in peak_assignments["total"]:
                for g in groups:
                    group_cov[i][g] = [
                        peak_assignments["total"][i][s] for s in groups[g]
                    ]
            # Take pairwise combinations of alternative PAS and test for differential coverage
            for i, j in combinations(group_cov, 2):
                x = [np.array(group_cov[i][g]) for g in groups]
                n = [
                    np.array(group_cov[i][g]) + np.array(group_cov[j][g])
                    for g in groups
                ]
                total_cov = sum([e.sum() for e in n])
                alt_cov = sum([e.sum() for e in x])
                if total_cov < min_total:
                    continue
                if alt_cov / total_cov < min_alt_fraction or alt_cov / total_cov > (
                    1 - min_alt_fraction
                ):
                    continue
                # TODO: this doesn't look right given the definition of min_sa in Isotools docstring
                if sum((ni >= min_n).sum() for ni in n[:2]) < min_sa:
                    continue
                pval, params = test(x[:2], n[:2])
                # pvalue is NaN, comparison is invalid (e.g. all counts zero for both groups in one alternate)
                if np.isnan(pval):
                    continue
                covs = [
                    val for lists in zip(x, n) for pair in zip(*lists) for val in pair
                ]
                res.append(
                    [
                        gene.name,
                        gene.id,
                        gene.chrom,
                        gene.strand,
                        last_exon,
                        trids_by_last_exon[last_exon],
                        peaks["total"][0][i],
                        peaks["total"][0][j],
                        "PAS",
                        pval,
                        x,
                        n,
                        *list(params),
                        *covs,
                    ]
                )
    # Column names
    colnames = [
        "gene",
        "gene_id",
        "chrom",
        "strand",
        "last_exon_start",
        "trids",
        "start",
        "end",
        "splice_type",
        "pvalue",
        "x",
        "n",
    ]
    colnames += [
        groupname + part
        for groupname in groupnames[:2] + ["total"] + groupnames[2:]
        for part in ["_PSI", "_disp"]
    ]
    colnames += [
        f"{sample}_{groupname}_{w}"
        for groupname, group in zip(groupnames, groups_arr)
        for sample in group
        for w in ["_in_cov", "_total_cov"]
    ]
    return pd.DataFrame(res, columns=colnames)
