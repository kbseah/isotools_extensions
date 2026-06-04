"""Tests for alternative polyadenylation (APA) using LABRAT-type psi metric.

Main functions are:
* `ale_psi_by_group`: Test alternative last exon APA (ALE-APA).
* `apa_psi_by_group_one_exon`: Test tandem 3'-UTR APA for a single exon in a
    gene of interest (TUTR-APA).
* `apa_psi_by_group`: Applies TUTR-APA test to all last exons for a given gene.
"""

import logging
import numpy as np
import pandas as pd
from isotools import Transcriptome
from isotools_extensions.alt_pas import get_gene_last_exons, get_gene_terminal_peaks


logger = logging.getLogger("isotools")


def gene_coverage_sum_samplewise(
    coverage: np.ndarray, sample_groups: dict, sort_descending: bool = False
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
    counts_sum = np.asarray(
        [coverage[sample_groups[key], :].sum(axis=0) for key in keys]
    )
    return counts_sum, keys


def gene_coverage_sum_transcriptwise(
    coverage: np.ndarray, transcript_groups: dict, sort_descending: bool = False
) -> tuple:
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
    counts_sum = np.asarray(
        [coverage[:, transcript_groups[key]].sum(axis=1) for key in keys]
    ).T
    return counts_sum, keys


def filter_transcripts_by_coverage(
    coverage: np.ndarray,
    mincov_frac: float = 0.01,
) -> tuple:
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
    return coverage[:, mask], np.where(mask == True)[0]


def coverage_matrix_to_psi(
    coverage: np.ndarray,
) -> np.ndarray:
    """Calculate LABRAT-style psi from counts matrix.

    :param coverage: 2D matrix of coverage per sample (row) per ALE/APA
        (columns). The columns must already be sorted from proximal (lowest
        index) to distal (highest index).
    :returns: Array of length `coverage.shape[0]`, of the psi values
        for each sample.
    """
    # m values
    mm = np.array(range(coverage.shape[1])) / (coverage.shape[1] - 1)
    scaled = (coverage * mm).sum(axis=1)
    unscaled = coverage.sum(axis=1)
    # Suppress warnings for invalid division and handle division by zero
    # if sum is zero for some rows (i.e. samples/groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = scaled / unscaled
        psi[~np.isfinite(psi)] = np.nan  # Replace inf and -inf with NaN
    return psi


def bootstrap_count_matrix(obs: list | np.ndarray, rng=None) -> np.ndarray:
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
    transcriptome: Transcriptome,
    gene_id: str,
    groups: dict,
    query: str = "not UNSPLICED",
    n_bootstrap: int = 10,
    mincov_frac: float = 0.01,
) -> dict | None:
    """Calculate psi metric for alternative last exons per sample group.

    :param transcriptome: isotools.Transcriptome object
    :param gene_id: ID for gene of interest, for lookup in `transcriptome`
    :param groups: Sample groups for which to calculate psi; group names as
        keys, lists of sample names as values. Sample names should be present
        in `transcriptome.samples`
    :param query: Filter query passed to isotools.Gene object.
    :param n_bootstrap: Number of resamplings for significance testing.
    :param mincov_frac: Minimum fraction of total (exclusive) for an
        alternative last exon to be included in the psi calculation, to exclude
        exons that may be artefacts or have too low coverage.
    :returns: Results, under key `actual` are the actual psi values, counts,
        and junction coordinates calculated. Under `bootstraps` are the
        resampled psi values and count matrices for significance testing.
    """
    # Get indices of samples and groups
    sample2idx = {s: idx for (idx, s) in enumerate(transcriptome.samples)}
    groups2idx = {g: [sample2idx[s] for s in samples] for g, samples in groups.items()}

    # Get junction coordinates for exons of interest
    trids = transcriptome[gene_id].filter_transcripts(query)
    if len(trids) <= 1:
        logger.debug("No transcripts after filter query for gene %s", gene_id)
        return None
    junc2trid = {
        junc: [t for t in transcripts if t in trids]
        for junc, transcripts in get_gene_last_exons(transcriptome[gene_id]).items()
    }

    # Sum counts per last exon and sort from proximal to distal
    counts_junc, juncs = gene_coverage_sum_transcriptwise(
        transcriptome[gene_id].coverage,
        junc2trid,
        sort_descending=(transcriptome[gene_id].strand == "-"),
    )
    if len(juncs) <= 1:
        logger.debug("No junctions found for gene %s", gene_id)
        return None

    # Sum counts by sample group
    counts_junc_group, groups = gene_coverage_sum_samplewise(counts_junc, groups2idx)

    # Filter only those junctions that represent at least x% of total
    counts_filt, filt_idx = filter_transcripts_by_coverage(
        counts_junc_group, mincov_frac
    )
    if len(filt_idx) <= 1:
        logger.debug(
            "Zero or only one exon after coverage filtering for gene %s",
            gene_id,
        )
        return None
    juncs_filt = [juncs[i] for i in filt_idx]
    psi = coverage_matrix_to_psi(counts_filt)

    out = {}
    out["actual"] = {
        "counts": counts_filt,
        "juncs": {j: junc2trid[j] for j in juncs_filt},
        "psi": psi,
        "juncs_ordered": juncs_filt,
        "groups_ordered": groups,
    }

    # bootstrapping
    out["bootstraps"] = []
    for _i in range(n_bootstrap):
        bs = bootstrap_count_matrix(counts_filt)
        bs_psi = coverage_matrix_to_psi(bs)
        out["bootstraps"].append({"counts": bs, "psi": bs_psi})

    return out


def apa_psi_by_group_one_exon(
    transcriptome: Transcriptome,
    gene_id: str,
    trids: list,
    groups: dict,
    query: str = "not UNSPLICED",
    n_bootstrap: int = 10,
    mincov_frac: float = 0.01,
) -> dict | None:
    """Calculate psi metric for APA per sample group for one exon of one gene.

    This function is primarily meant to be called by `apa_psi_by_group`
    although it is exposed to the user. See the other function for other
    parameter documentation.

    :param trids: List of transcript IDs that contain the exon of interest
    :returns: Results, under key `actual` are the actual psi values, counts,
        and junction coordinates calculated. Under `bootstraps` are the
        resampled psi values and count matrices for significance testing.
    """
    # Peak calling
    _p, _c, _s, peaks, peak_assignments = get_gene_terminal_peaks(
        transcriptome[gene_id], trids=trids
    )
    if peak_assignments is None:  # No peaks, coverage too low
        return None
    # Dict of peak indices by peak coordinates
    peak_idx_sorted = [
        i
        for i, peak in sorted(
            enumerate(peaks["total"][0]),
            key=lambda x: x[1],
            reverse=(transcriptome[gene_id].strand == "-"),
        )
    ]
    # Coverage per sample (rows) per peak (columns)
    counts = np.array(
        [
            [peak_assignments["total"][p][s] for p in peak_idx_sorted]
            for s in transcriptome.samples
        ]
    )
    # Get indices of samples and groups
    sample2idx = {s: idx for (idx, s) in enumerate(transcriptome.samples)}
    groups2idx = {g: [sample2idx[s] for s in samples] for g, samples in groups.items()}

    # Sum counts by sample group
    counts_group, groups = gene_coverage_sum_samplewise(counts, groups2idx)

    # Filter only those peaks that represent at least x% of total
    counts_filt, filt_idx = filter_transcripts_by_coverage(counts_group, mincov_frac)
    if len(filt_idx) <= 1:
        logger.debug(
            "Zero or only one peak after coverage filtering for gene %s",
            gene_id,
        )
        return None
    peak_idx_sorted = [peak_idx_sorted[i] for i in filt_idx]
    psi = coverage_matrix_to_psi(counts_filt)

    out = {}
    out["actual"] = {
        "counts": counts_filt,
        "peaks": {i: peaks["total"][0][i] for i in peak_idx_sorted},
        "psi": psi,
        "peaks_ordered": peak_idx_sorted,
        "groups_ordered": groups,
    }

    # bootstrapping
    out["bootstraps"] = []
    for i in range(n_bootstrap):
        bs = bootstrap_count_matrix(counts_filt)
        bs_psi = coverage_matrix_to_psi(bs)
        out["bootstraps"].append({"counts": bs, "psi": bs_psi})

    return out


def apa_psi_by_group(
    transcriptome: Transcriptome,
    gene_id: str,
    groups: dict,
    query: str = "not UNSPLICED",
    n_bootstrap: int = 10,
    mincov_frac: float = 0.01,
) -> dict | None:
    """Calculate psi metric for APA per sample group for all exons of a gene.

    :param transcriptome: isotools.Transcriptome object
    :param gene_id: ID for gene of interest, for lookup in `transcriptome`
    :param groups: Sample groups for which to calculate psi; group names as
        keys, lists of sample names as values. Sample names should be present
        in `transcriptome.samples`
    :param query: Filter query passed to isotools.Gene object.
    :param n_bootstrap: Number of resamplings for significance testing.
    :param mincov_frac: Minimum fraction of total (exclusive) for an
        alternative last exon to be included in the psi calculation, to exclude
        exons that may be artefacts or have too low coverage.
    :returns: Results keyed by internal splice junction coordinate of each last
        exon. Results of APA for each last exon are structured as for
        `ale_psi_by_group`: under key `actual` are the actual psi values,
        counts, and junction coordinates calculated. Under `bootstraps` are the
        resampled psi values and count matrices for significance testing.
    """
    # Get transcripts corresponding to each last exon
    lastexon2trid = get_gene_last_exons(transcriptome[gene_id])
    out = {}
    for lastexon, trids in lastexon2trid.items():
        res = apa_psi_by_group_one_exon(
            transcriptome,
            gene_id,
            trids,
            groups,
            query=query,
            n_bootstrap=n_bootstrap,
            mincov_frac=mincov_frac,
        )
        if res is not None:
            out[lastexon] = res
    return out


def test_apa_psi(
    self: Transcriptome,
    groups: dict,
    contrast: list | tuple,
    transcript_query: str = "not UNSPLICED",
    n_bootstrap: int = 100,
    mincov_frac: float = 0.01,
    min_total: int = 10,
    min_group: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Test for TUTR-APA with psi metric.

    :param self: isotools.Transcriptome object
    :param groups: Sample groups for which to calculate psi; group names as
        keys, lists of sample names as values. Sample names should be present
        in `self.samples`
    :param contrast: Pair of groups to contrast for delta_psi calculation and
        significance testing.
    :param transcript_query: Filter for transcripts.
    :param n_bootstrap: Number of resamplings for significance testing.
    :param mincov_frac: Minimum fraction of total (exclusive) for an
        alternative last exon to be included in the psi calculation, to exclude
        exons that may be artefacts or have too low coverage.
    :param min_total: Minimum total coverage over all selected samples (for all
        groups combined)
    :param min_group: Minimum coverage per group; all groups must have at least
        this coverage for event to be considered.
    :param **kwargs: Keyword arguments passed to self.iter_genes
    :returns: pandas.DataFrame with test results.
    """
    if len(contrast) != 2:
        raise ValueError("Only two groups can be contrasted")
    outs = []
    for gene in self.iter_genes(**kwargs):
        res = apa_psi_by_group(
            self,
            gene.id,
            groups,
            query=transcript_query,
            n_bootstrap=n_bootstrap,
            mincov_frac=mincov_frac,
        )
        if res is None:
            continue
        for junc, result in res.items():
            if result["actual"]["counts"].sum() < min_total:
                continue
            if any(result["actual"]["counts"].sum(axis=1) < min_group):
                continue
            psi_dict = dict(
                zip(
                    result["actual"]["groups_ordered"],
                    result["actual"]["psi"],
                    strict=True,
                ),
            )
            delta_psi = psi_dict[contrast[0]] - psi_dict[contrast[1]]
            peaks = [
                result["actual"]["peaks"][i] for i in result["actual"]["peaks_ordered"]
            ]
            resample_psi = []
            for bs in result["bootstraps"]:
                bs_psi_dict = dict(
                    zip(result["actual"]["groups_ordered"], bs["psi"], strict=True),
                )
                resample_psi.append(bs_psi_dict[contrast[0]] - bs_psi_dict[contrast[1]])
            p = sum(np.array(resample_psi) > 0) / len(resample_psi)
            out = {
                "gene": gene.name,
                "gene_id": gene.id,
                "chrom": gene.chrom,
                "strand": gene.strand,
                "start": min([*peaks, junc]),
                "end": max([*peaks, junc]),
                "splice_type": "TUTR_APA",
                "pvalue": p,
                "peaks": np.array(peaks).tolist(),  # convert to native python int
                "groups": result["actual"]["groups_ordered"],
                "counts": result["actual"][
                    "counts"
                ].tolist(),  # convert numpy to native python list and int
                "junc": junc,
                "delta_psi": delta_psi,
            }
            out.update(
                {
                    f"{group}_psi": psi_dict[group]
                    for group in result["actual"]["groups_ordered"]
                },
            )
            outs.append(out)
    return pd.DataFrame(outs)


def test_ale_psi(
    self: Transcriptome,
    groups: dict,
    contrast: list | tuple,
    transcript_query: str = "not UNSPLICED",
    n_bootstrap: int = 100,
    mincov_frac: float = 0.01,
    min_total: int = 10,
    min_group: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Test for ALE-APA with psi metric.

    :param self: isotools.Transcriptome object.
    :param groups: Sample groups for which to calculate psi; group names as
        keys, lists of sample names as values. Sample names should be present
        in `self.samples`
    :param contrast: Pair of groups to contrast for delta_psi calculation and
        significance testing.
    :param transcript_query: Filter for transcripts.
    :param n_bootstrap: Number of resamplings for significance testing.
    :param mincov_frac: Minimum fraction of total (exclusive) for an
        alternative last exon to be included in the psi calculation, to exclude
        exons that may be artefacts or have too low coverage.
    :param min_total: Minimum total coverage over all selected samples (for all
        groups combined)
    :param min_group: Minimum coverage per group; all groups must have at least
        this coverage for event to be considered.
    :param **kwargs: Keyword arguments passed to self.iter_genes
    :returns: pandas.DataFrame with test results.
    """
    if len(contrast) != 2:
        raise ValueError("Only two groups can be contrasted")
    outs = []
    for gene in self.iter_genes(**kwargs):
        res = ale_psi_by_group(
            self,
            gene.id,
            groups,
            query=transcript_query,
            n_bootstrap=n_bootstrap,
            mincov_frac=mincov_frac,
        )
        if res is None:
            continue
        if res["actual"]["counts"].sum() < min_total:
            continue
        if any(res["actual"]["counts"].sum(axis=1) < min_group):
            continue
        psi_dict = dict(
            zip(res["actual"]["groups_ordered"], res["actual"]["psi"], strict=True),
        )
        delta_psi = psi_dict[contrast[0]] - psi_dict[contrast[1]]
        juncs = res["actual"]["juncs_ordered"]
        resample_psi = []
        for bs in res["bootstraps"]:
            bs_psi_dict = dict(
                zip(res["actual"]["groups_ordered"], bs["psi"], strict=True),
            )
            resample_psi.append(bs_psi_dict[contrast[0]] - bs_psi_dict[contrast[1]])
        p = sum(np.array(resample_psi) > 0) / len(resample_psi)
        out = {
            "gene": gene.name,
            "gene_id": gene.id,
            "chrom": gene.chrom,
            "strand": gene.strand,
            "start": min(juncs),
            "end": max(juncs),
            "splice_type": "ALE_APA",
            "pvalue": p,
            "juncs": juncs,
            "groups": res["actual"]["groups_ordered"],
            "counts": res["actual"][
                "counts"
            ].tolist(),  # convert numpy to native python list and int
            "delta_psi": delta_psi,
        }
        out.update(
            {
                f"{group}_psi": psi_dict[group]
                for group in res["actual"]["groups_ordered"]
            },
        )
        outs.append(out)
    return pd.DataFrame(outs)
