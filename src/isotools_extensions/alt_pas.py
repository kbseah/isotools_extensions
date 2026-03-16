from collections import defaultdict
from collections.abc import Callable
from itertools import combinations

import isotools._utils
import numpy as np
import pandas as pd
from isotools import Gene
from isotools._transcriptome_stats import TESTS, _check_groups
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu
from statsmodels.stats import multitest


def pileup_to_smoothed(pileup: dict, smooth_window: int = 31) -> tuple:
    """Smooth a coverage pileup.

    :param pileup: Dict of coverage values keyed by genomic coordinate
    :param smooth_window: Window size for smoothing function
    :returns: Pileup values as array, coordinates as array, and smoothed values
    :rtype: tuple
    """
    # X coordinates of the pileup
    # Add flanking buffer of 1*smooth_window to avoid smoothed max falling
    # right at the first or last position and not being found by find_peaks
    coords = range(
        min(pileup.keys()) - smooth_window,
        max(pileup.keys()) + 1 + smooth_window,
    )
    pileup_arr = [pileup.get(pos, 0) for pos in coords]
    smoothed = isotools._utils.smooth(np.array(pileup_arr), smooth_window)
    return (pileup_arr, coords, smoothed)


def get_transcript_terminal_peaks(
    gene: Gene,
    trid: int,
    which: str = "PAS",
    total: bool = False,
    smooth_window: int = 31,
    prominence: int = 2,
) -> tuple[dict, dict, dict, dict]:
    """Get PAS/TSS peaks for an isotools transcript.

    Unlike default isotools behavior, this calls all peaks without choosing a
    unified consensus for each transcript.

    :param gene: isotools.Gene object
    :param trid: Transcript index
    :param which: Either "PAS" or "TSS"
    :param total: Sum pileups for all samples if True, else plot samples separately
    :param smooth_window: Window size for smoothing function
    :param prominence: Minimum peak prominence to retain
    :returns: tuple of dicts representing pileup, coordinates, smoothed pileup,
        and peak positions
    """
    pileup, coords, smoothed, peaks = {}, {}, {}, {}
    if total:
        pileup_sum = {}
        for sample in gene.transcripts[trid][which]:
            for pos, cov in gene.transcripts[trid][which][sample].items():
                pileup_sum[pos] = pileup_sum.get(pos, 0) + cov
        pileup["total"], coords["total"], smoothed["total"] = pileup_to_smoothed(
            pileup_sum,
            smooth_window,
        )
        peaks["total"] = translate_peaks_offset(
            find_peaks(smoothed["total"], prominence=(prominence, None)),
            min(coords["total"]),
        )
    else:
        for sample in gene.transcripts[trid][which]:
            pileup[sample], coords[sample], smoothed[sample] = pileup_to_smoothed(
                gene.transcripts[trid][which][sample],
                smooth_window,
            )
            peaks[sample] = translate_peaks_offset(
                find_peaks(smoothed[sample], prominence=(prominence, None)),
                min(coords[sample]),
            )
    return (pileup, coords, smoothed, peaks)


def assign_to_closest_peak(pos: int, peaks: list) -> tuple:
    """Find closest peak to a given genomic coordinate.

    Assumes peaks and pos have already been converted to genomic coordinates

    :param pos: Genomic coordinate of position to find closest peak to
    :param peaks: List of peak coordinates
    :returns: Tuple of the index of the closest peak, and the genomic
        coordinate distance between pos and the closest peak.
    :rtype: tuple
    """
    dist = [abs(p - pos) for p in peaks]
    closest_index = np.argmin(dist)
    return closest_index, dist[closest_index]


def translate_peaks_offset(peaks: list, offset: int) -> tuple:
    """Translate array indices back to genomic coordinates.

    :param peaks: List of peak positions in relative coordinates
    :param offset: The offset between genomic and relative coordinates.
    :returns: Tuple of the peak positions in genomic coords and a dict with
        peak prominences, left_bases, and right_bases.
    :rtype: tuple
    """
    return (
        np.array([p + offset for p in peaks[0]]),
        {
            "prominences": peaks[1]["prominences"],
            "left_bases": np.array([lb + offset for lb in peaks[1]["left_bases"]]),
            "right_bases": np.array([rb + offset for rb in peaks[1]["right_bases"]]),
        },
    )


def get_gene_terminal_peaks(
    gene: Gene,
    trids: list = None,
    which: str = "PAS",
    smooth_window: int = 31,
    prominence: int = 2,
    # TODO: Arguments to filter transcripts; removing unspliced is important
) -> tuple:
    """Get PAS/TSS peaks for an isotools Gene, summing across all transcripts.

    Unlike default isotools behavior, this calls all peaks without choosing a
    unified consensus for each transcript.

    :param gene: isotools.Gene object
    :param trids: List of transcript IDs to include (as ints); if None, include
        all transcripts
    :param which: Either "PAS" or "TSS"
    :param smooth_window: Window size for smoothing function
    :param prominence: Minimum peak prominence to retain
    :returns: Tuple of pileup, coords, smoothed, peaks, peak_assignments
    :rtype: tuple
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
        pileup_sum,
        smooth_window,
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
        lambda: defaultdict(int),
    )  # peak, sample -> count
    for trid, transcript in enumerate(gene.transcripts):
        if trid in trids:
            for sample in transcript[which]:
                for pos in transcript[which][sample]:
                    closest_index, distance = assign_to_closest_peak(
                        pos,
                        peaks["total"][0],
                    )
                    if distance <= smooth_window:
                        peak_assignments["total"][closest_index][sample] += transcript[
                            which
                        ][sample][pos]
    return pileup, coords, smoothed, peaks, peak_assignments


def get_gene_last_exons(gene: Gene, **kwargs) -> dict:
    """Get last exons for all transcripts of a gene.

    :param gene: isotools.Gene object
    :param **kwargs: Parameters to pass to gene.filter_transcripts()
    :returns: Dict mapping transcript IDs to start positions of last exons
    :rtype: dict
    """
    trids = gene.filter_transcripts(**kwargs)
    last_exons = {}
    for trid, transcript in enumerate(gene.transcripts):
        last_exon_start = (
            transcript["exons"][-1][0]
            if gene.strand == "+"
            else transcript["exons"][0][1]
        )
        if trid in trids:
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
    test: (
        str | Callable
    ) = "auto",  # either string with test name or a custom test function
    **kwargs,
) -> pd.DataFrame:
    """Identify and test alternative PAS within an isotools Gene.

    Isotools chooses a unified PAS per transcript by default. However,
    transcripts are defined by their internal splice sites, so a single
    transcript may have multiple PAS, not captured by the unified PAS. This
    function first groups transcripts by their last exon, then calls all PAS
    peaks for all transcripts with that last exon, and tests for differential
    APA between groups, without relying on a unified consensus for each
    transcript.

    Output columns are same as those from
    isotools._transcriptome_stats.altsplice_test, except for:
     * "last_exon_start" - start position of the last exon shared by the
         transcripts tested
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
    groupnames, groups_arr, _grp_idx = _check_groups(self, groups)
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
            _pileup, coords, smoothed, peaks, peak_assignments = (
                get_gene_terminal_peaks(
                    gene=gene,
                    trids=trids_by_last_exon[last_exon],
                    which="PAS",
                    smooth_window=smooth_window,
                    prominence=prominence,
                )
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
            # Take pairwise combinations of alternative PAS and test for
            # differential coverage
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
                    val
                    for lists in zip(x, n, strict=True)
                    for pair in zip(*lists, strict=True)
                    for val in pair
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
                    ],
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
        for groupname in [*groupnames[:2], "total", *groupnames[2:]]
        for part in ["_PSI", "_disp"]
    ]
    colnames += [
        f"{sample}_{groupname}_{w}"
        for groupname, group in zip(groupnames, groups_arr, strict=True)
        for sample in group
        for w in ["_in_cov", "_total_cov"]
    ]
    return pd.DataFrame(res, columns=colnames)


def transcript_mean_3utr(transcript:dict, groups:dict):
    """Get the mean 3'UTR length per sample group for a transcript.

    :params transcript: Dict of transcript from isotools.Gene object
    :params groups: Dict of samples keyed by group name
    :returns: Mean 3'-UTR length for each group.
    :rtype: dict
    """
    if "ORF" not in transcript:
        print("ORF not in transcript")
        return None
    canonical_3utr = transcript["ORF"][2]["3'UTR"]
    persample_3utr = {}
    if transcript["strand"] == "+":
        for s in transcript["PAS"]:
            persample_3utr[s] = {
                canonical_3utr + k - transcript["exons"][-1][1] : v
                for k,v in transcript["PAS"][s].items()
            }
    elif transcript["strand"] == "-":
        for s in transcript["PAS"]:
            persample_3utr[s] = {
                canonical_3utr + transcript["exons"][0][0] - k : v
                for k,v in transcript["PAS"][s].items()
            }
    out = defaultdict(list)
    for g, samples in groups.items():
        for s in samples:
            if s in persample_3utr:
                out[g].extend(persample_3utr[s].items())
    return {
        g : sum([k*v for k,v in out[g]]) / sum([v for k,v in out[g]])
        for g in out
    }


def transcript_get_3utr(transcript:dict, groups:dict) -> dict[str, dict[int|float, int]]:
    """Get counts per 3'UTR length per sample group for a transcript.

    :params transcript: Transcript dict from isotools gene object
    :params groups: Dict of samples keyed by group
    :returns: Dict of dicts, first key is group name, second key the 3'-UTR
        length, values are counts of transcripts with that UTR length.
    :rtype: dict
    """
    if "ORF" not in transcript:
        raise KeyError("transcript does not have ORF")
    canonical_3utr = transcript["ORF"][2]["3'UTR"]
    out = defaultdict(dict)
    persample_3utr = {}
    if transcript["strand"] == "+":
        for s in transcript["PAS"]:
            persample_3utr[s] = {
                canonical_3utr + k - transcript["exons"][-1][1] : v
                for k,v in transcript["PAS"][s].items()
            }
    elif transcript["strand"] == "-":
        for s in transcript["PAS"]:
            persample_3utr[s] = {
                canonical_3utr + transcript["exons"][0][0] - k : v
                for k,v in transcript["PAS"][s].items()
            }
    for g, samples in groups.items():
        for s in samples:
            if s in persample_3utr:
                for u, count in persample_3utr[s].items():
                   out[g][u] = out[g].get(u, 0) + count
    return out


def transcript_get_lastexon_len(transcript:dict, groups:dict) -> dict[str, dict[int|float, int]]:
    """Get counts per last exon length per group for a transcript.

    :params transcript: Transcript dict from isotools gene object
    :params groups: Dict of samples keyed by group
    :returns: Dict of dicts, first key is group name, second key the last exon
        length, values are counts of transcripts with that length.
    :rtype: dict
    """
    out = defaultdict(dict)
    persample_lens = {}
    if transcript["strand"] == "+":
        for s in transcript["PAS"]:
            persample_lens[s] = {
                k - transcript["exons"][-1][0] : v
                for k,v in transcript["PAS"][s].items()
            }
    elif transcript["strand"] == "-":
        for s in transcript["PAS"]:
            persample_lens[s] = {
                transcript["exons"][0][1] - k : v
                for k,v in transcript["PAS"][s].items()
            }
    for g, samples in groups.items():
        for s in samples:
            if s in persample_lens:
                for u, count in persample_lens[s].items():
                   out[g][u] = out[g].get(u, 0) + count
    return out


def mean_countsdict(countsdict:dict[int|float, int]) -> tuple[float, int]:
    """Calculate mean from a dict of counts of a numeric variable.

    :returns: Arithmetic mean of the variable, and sum of counts.
    :rtype: tuple
    """
    sumlen = sum(k*v for k,v in countsdict.items())
    sumcount = sum(v for v in countsdict.values())
    return sumlen/sumcount, sumcount


def gene_get_3utr_len(
    gene: isotools.gene.Gene, groups: dict, mean=True, **kwargs,
) -> list:
    """Compare 3'-UTR lengths between sample groups for a given gene.

    ORF calling must first be performed on the transcripts. Only transcripts with
    the 'ORF' parameter will be considered.

    :param gene: isotools gene object
    :param group: Dict of samples keyed by group
    :param mean: Summarize mean 3'-UTR lengths? If False, report raw counts.
    :param **kwargs: Arguments for transcripts to pass to .filter_transcripts(). The
        filters are applied before the counts are aggregated by last exon.
    :returns: TBD
    """
    tr_by_last_exon = get_gene_last_exons(gene, **kwargs)
    # key1 last exon splice junction coordinate
    # key2 sample group
    # key3 3'-UTR length
    # value number of transcripts with that length
    utrlen_by_last_exon = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for le, trids in tr_by_last_exon.items():
        for trid in trids:
            utrlen = transcript_get_3utr(gene.transcripts[trid], groups)
            for g, counts in utrlen.items():
                for l, count in counts.items():
                    utrlen_by_last_exon[le][g][l] += count
    if not mean:
        return utrlen_by_last_exon
    out = []
    for le, bygroup in utrlen_by_last_exon.items():
        for g, countsdict in bygroup.items():
            meanlen, sumcount = mean_countsdict(countsdict)
            out.append(dict(zip(
                ["gene", "last_exon_start", "group", "mean_3utr_len", "total_cov"],
                [gene.id, le, g, meanlen, sumcount],
                strict=True,
            )))
    return out


def gene_get_lastexon_len(
    gene: isotools.gene.Gene, groups: dict, **kwargs,
) -> list:
    """Compare last exon lengths between sample groups for a given gene.

    Last exon length is more variable than 3'-UTR length, but this procedure
    should be more reliable for finding differences in PAS sites between
    groups, because it does not rely on ORF predictions, which may be
    inaccurate for fragmentary transcripts.

    :param gene: isotools gene object
    :param groups: Dict of samples keyed by group
    :param **kwargs: Arguments for transcripts to pass to
        .filter_transcripts(). The filters are applied before the counts are
        aggregated by last exon.
    :returns: TBD
    """
    tr_by_last_exon = get_gene_last_exons(gene, **kwargs)
    # key1 last exon splice junction coordinate
    # key2 sample group
    # key3 last exon length
    # value number of transcripts with that length
    lastexon_lengths = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for le, trids in tr_by_last_exon.items():
        for trid in trids:
            lastexon_len = transcript_get_lastexon_len(gene.transcripts[trid], groups)
            # TODO check for negative length!
            for g, counts in lastexon_len.items():
                for l, count in counts.items():
                    lastexon_lengths[le][g][l] += count
    out = defaultdict(dict)
    for le, bygroup in lastexon_lengths.items():
        for g, countsdict in bygroup.items():
            meanlen, sumcount = mean_countsdict(countsdict)
            lengths = [j for k,v in countsdict.items() for j in [k]*v]
            out[le][g] = dict(zip(
                ["trid", "lastexon_len", "mean_lastexon_len", "total_cov"],
                [tr_by_last_exon[le], lengths, meanlen, sumcount],
                strict=True,
            ))
    return out


def gene_test_lastexon_len(
    gene:Gene, groups:dict, pairs:list, test:Callable=mannwhitneyu, **kwargs,
    ) -> list[dict]:
    """Statistically test for difference in last exon length between groups.

    To find genes with potential alternative polyadenylation sites between
    groups of samples, we test for different last exon lengths. We test last
    exon length instead of 3'-UTR length because the latter depends on ORF
    prediction, which may be incorrect or missing for fragmentary transcripts,
    leading to inconsistencies and inaccurate 3'-UTR positions.

    The default test applied is the Mann-Whitney U test for two independent
    samples. This nonparametric test was chosen because we cannot assume that
    the PAS positions are normally distributed or homoscedastic; one reason is
    that in one sample, transcripts with a number of alternative PASs may be
    observed (i.e., we see multiple PAS pileup peaks).

    Because the test is pairwise, and there may be more than two groups of
    interest, a list of pairs to compare should be specified. Exons where one
    or more pairs are missing are skipped.

    :param gene: Isotools2 gene object
    :param groups: Dict of samples keyed by groups.
    :param pairs: List of pairs of groups to compare.
    :param test: Statistical test to apply, result must have the attributes
        `statistic` and `pvalue`.
    :param **kwargs: Arguments to pass to .filter_transcripts()
    :returns: List of dict of results
    :rtype: list
    """
    dd = gene_get_lastexon_len(gene, groups, **kwargs)
    out = []
    for le, gdict in dd.items():
        for i,j in pairs:
            try:
                result = test(gdict[i]["lastexon_len"], gdict[j]["lastexon_len"])
                out.append(dict(zip(
                    ["gene_id","trid","lastexon_start","group1","group1_mean","group1_count","group2","group2_mean","group2_count","stat","pvalue"],
                    [gene.id, gdict[i]["trid"], le, i, gdict[i]["mean_lastexon_len"], gdict[i]["total_cov"], j, gdict[j]["mean_lastexon_len"], gdict[j]["total_cov"], result.statistic, result.pvalue],
                    strict=True,
                )))
            except KeyError:
                pass
    return out


def test_diff_lastexon_len(
    self, groups:dict, pairs:list, alpha:float=0.05, method:str="fdr_bh", **kwargs,
) -> pd.DataFrame:
    """Test for differential last exon length.

    Unlike test_alternative_pas, this function does not attempt to call PAS
    peaks but simply tests for significant difference in the last exon lengths
    between groups for transcripts of a given gene that share the same last
    exon.

    We use last exon length as metric instead of 3'-UTR length because CDS
    annotations can be inconsistent or incorrect when the transcripts are
    fragmentary, as is often the case with long read data.

    :param self: isotools.Transcriptome object
    :param groups: Dict of samples by group name
    :param pairs: Pairs of groups to be compared (should be a list of tuples;
        do not use an itertools.combinations without first converting to list)
    :param alpha: P-value threshold after adjustment for multiple comparisons.
    :param method: Method for p-value adjustment, passed to
        multitest.multipletests
    :param **kwargs: Filter arguments to be passed to gene.filter_transcripts()
    :returns: DataFrame with comparison of mean last exon lengths between
        groups for each gene and group of transcripts, with significance
        testing of difference.
    :rtype: pd.DataFrame
    """
    out = []
    for gene in self.iter_genes():
        out.extend(
            gene_test_lastexon_len(
                gene,
                groups,
                pairs,
                test=mannwhitneyu,
                **kwargs,
            ),
        )
    df = pd.DataFrame(out)
    _mr, padj, _acsidak, _acbonf = multitest.multipletests(
        df["pvalue"], alpha=alpha, method=method,
    )
    df["padj"] = padj
    df["diff_mean"] = df["group1_mean"] - df["group2_mean"]
    df["diff_sign"] = df["diff_mean"] > 0
    return df
