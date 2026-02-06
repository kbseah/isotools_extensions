from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from isotools._transcriptome_stats import TESTS, _check_groups


def get_exon_nodes(segment_graph, transcript, start_node=None, end_node=None):
    """Get exon nodes for a given transcript from its isotools segment graph

    :param segment_graph: isotools.SegmentGraph object
    :param transcript: Transcript index
    :param start_node: Start from this node. If None, then start from leftmost
        node (based on coordinate, ignores gene strand)
    :param end_node: End with this node. If None, then end at rightmost node
        (based on coordinate, ignores gene strand)
    """
    node = segment_graph._tss[transcript] if start_node is None else start_node
    end = segment_graph._pas[transcript] if end_node is None else end_node
    out = []
    exon_start = node
    while node <= end:
        if transcript in segment_graph[node].suc:
            suc = segment_graph[node].suc[transcript]
            if segment_graph[node].end != segment_graph[suc].start:
                out.append((exon_start, node))
                exon_start = suc
            node = suc
            if node == end:
                out.append((exon_start, node))
                return out
        elif node == end:
            # Monoexonic
            out.append((exon_start, node))
            return out
        else:
            raise ValueError("Invalid exon node")


def get_exon_coords(segment_graph, transcript: int):
    """Get exon coordinates for a given transcript from its segment graph

    Should give identical output to isotool's original
    SegmentGraph._get_all_exons method.

    :param segment_graph: isotools.SegmentGraph object
    :param transcript: Transcript index
    """
    return [
        (segment_graph[ns].start, segment_graph[ne].end)
        for ns, ne in get_exon_nodes(segment_graph, transcript)
    ]


def get_ale_mono(segment_graph):
    """Find alternative last exon (ALE) events in the segment graph (DEPRECATED)

    This procedure assumes that there is only a single ALE and a common splice
    site. Will overlook instances of >1 ALE spliced together. Superseded by
    get_ale_afe.
    """
    if segment_graph.strand == "+":
        last_exons = [
            (trid, segment_graph._get_exon_start(trid, pas), pas)
            for trid, pas in enumerate(segment_graph._pas)
        ]
        # Get splice junctions to second-to-last exon
        # Monoexonic transcripts will be dropped
        last_exon_junctions = [
            (trid, segment_graph[ns].pre[trid], ns, ne)
            for (trid, ns, ne) in last_exons
            if trid in segment_graph[ns].pre
        ]
    else:
        # _pas and _tss attributes are misleadingly named
        last_exons = [
            (trid, segment_graph._get_exon_end(trid, pas), pas)
            for trid, pas in enumerate(segment_graph._tss)
        ]
        last_exon_junctions = [
            (trid, segment_graph[ns].suc[trid], ns, ne)
            for (trid, ns, ne) in last_exons
            if trid in segment_graph[ns].suc
        ]
    # arrange by splice junctions
    last_exon_junction_dict = defaultdict(lambda: defaultdict(list))
    for trid, pre, ns, ne in last_exon_junctions:
        last_exon_junction_dict[pre][ns].append(trid)
    # return all combinations of ALEs sharing the same second-to-last splice junction
    for pre in last_exon_junction_dict:
        for i, j in combinations(last_exon_junction_dict[pre].keys(), 2):
            prim_set = last_exon_junction_dict[pre][i]
            alt_set = last_exon_junction_dict[pre][j]
            prim_node_ids = (pre, i)
            alt_node_ids = (pre, j)
            yield (prim_set, alt_set, prim_node_ids, alt_node_ids, "ALE")


def get_ale_afe(segment_graph, which="ALE"):
    """Report last or first exon (ALE/AFE) events in the segment graph

    Report potential first or last exon events from a gene's segment graph.
    Most tools for ALE/AFE consider only cases where there is only a single
    alternative exon. However there may be multiple alternative exons that
    co-vary. This function first looks for splice junctions immediately before
    (ALE) or after (AFE) all single terminal exons in the segment graph, then
    collects all the exon variants downstream (ALE) or upstream (AFE) of those
    splice junctions. This way, multiple-exon alternatives at transcript ends
    are also captured.

    This function is used within find_ale_afe_pairs which implements additional
    filtering steps and reports pairwise alternative AFE/ALE events.

    :param segment_graph: isotools.SegmentGraph object
    :param which: Either "ALE" or "AFE"
    :returns dict: Nested dictionary where keys of the outer dictionary are the
    splice junction nodes before (ALE) or after (AFE) the alternative exons,
    and keys of the inner dictionary are tuples of exon nodes representing the
    alternative last/first exons. Values of the inner dictionary are lists of
    transcript indices supporting each alternative.
    """
    out = defaultdict(lambda: defaultdict(list))
    if (which == "ALE" and segment_graph.strand == "+") or (
        which == "AFE" and segment_graph.strand == "-"
    ):
        pre_nodes = []
        fragments = segment_graph.find_fragments()
        # Find splice junctions immediately before last exons
        # (second-to-last splice junctions)
        for trid, pas in enumerate(segment_graph._pas):
            ns = segment_graph._get_exon_start(trid, pas)
            # Drop fragments
            if trid in segment_graph[ns].pre and trid not in fragments:
                pre_nodes.append(segment_graph[ns].pre[trid])
        pre_nodes = list(set(pre_nodes))
        # Find all exon variants downstream of each second-to-last splice junction.
        # Include multiple last exons
        for pre_node in pre_nodes:
            for trid in segment_graph[pre_node].suc:
                downstream_nodes = get_exon_nodes(
                    segment_graph, trid, start_node=pre_node, end_node=None
                )
                # check that this node is at a splice junction
                if downstream_nodes[0][0] == downstream_nodes[0][1]:
                    out[pre_node][tuple(downstream_nodes[1:])].append(trid)
    elif (which == "ALE" and segment_graph.strand == "-") or (
        which == "AFE" and segment_graph.strand == "+"
    ):
        suc_nodes = []
        # TODO: use the isotools annotations instead, because some instances of
        # novel transcripts get called fragments too
        fragments = segment_graph.find_fragments()
        # Find splice junctions immediately after first exons
        # (second splice junctions)
        for trid, tss in enumerate(segment_graph._tss):
            ns = segment_graph._get_exon_end(trid, tss)
            # Drop fragments
            if trid in segment_graph[ns].suc and trid not in fragments:
                suc_nodes.append(segment_graph[ns].suc[trid])
        suc_nodes = list(set(suc_nodes))
        # Find all exon variants upstream of each second splice junction.
        # Include multiple first exons
        for suc_node in suc_nodes:
            for trid in segment_graph[suc_node].pre:
                upstream_nodes = get_exon_nodes(
                    segment_graph, trid, start_node=None, end_node=suc_node
                )
                # check that this node is at a splice junction
                if upstream_nodes[-1][0] == upstream_nodes[-1][1]:
                    out[suc_node][tuple(upstream_nodes[:-1])].append(trid)
    # Drop singletons (nothing to contrast)
    out = {i: out[i] for i in out if len(out[i]) > 1}
    return out


def find_ale_afe_pairs(gene, which: str = "ALE"):
    """Generator for ALE/AFE events similar to find_splice_bubbles

    To identify potential ALE and AFE events in a gene, we first identify all
    splice junctions immediately upstream of a terminal exon. Then, for each of
    these splice junctions, we collect all alternative terminal exons spliced
    at that junction. Finally, we yield all pairwise combinations of these
    alternative terminal exons as potential ALE/AFE events. This includes cases
    with multiple exons downstream of that splice junction. Because of
    transcript fragmentation, some of these events may not be true ALE/AFE
    events. We exclude pairs where any exons are shared between the two
    alternatives, as these represent other types of alternative splicing and
    should be captured by the standard find_splice_bubbles function.

    ALE and AFE cannot be uniquely defined with two nodes, so we also return
    the full list of primary and alternate nodes. This addresses a limitation
    of the altsplice_test function in IsoTools, which defines each AS event
    with two coordinates only. For compatibility the first five elements of the
    returned tuple correspond to the fields returned by
    isotools.SegmentGraph.find_splice_bubbles.

    Primary vs. alternate forms are determined by sort order of the node indices.

    :param gene: isotools.Gene object
    :param which: Either "ALE" or "AFE"
    :returns tuple: Tuple with the following elements:
        * primary isoform transcripts (list of transcript indices)
        * alternate isoform transcripts (list)
        * start coordinate (int)
        * end coordinate (int)
        * event type, either "AFE" or "ALE" (str)
        * coordinates in SUPPA2-like format
    """
    events = get_ale_afe(gene.segment_graph, which=which)
    for pre in events:
        for i, j in combinations(events[pre].keys(), 2):
            # Skip cases with overlapping exons, represent other splice events
            if any(exon in i for exon in j):
                continue
            if (which == "ALE" and gene.strand == "+") or (
                which == "AFE" and gene.strand == "-"
            ):
                prim_nodes, alt_nodes = sorted((i, j))
                prim_set = events[pre][prim_nodes]
                alt_set = events[pre][alt_nodes]
                # SUPPA2-like coordinates
                e1 = gene.segment_graph[pre].end
                s2 = gene.segment_graph[prim_nodes[0][0]].start
                e2 = gene.segment_graph[prim_nodes[-1][1]].end
                s3 = gene.segment_graph[alt_nodes[0][0]].start
                e3 = gene.segment_graph[alt_nodes[-1][1]].end
                coord = f"{e1!s}-{s2!s}:{e2!s}:{e1!s}-{s3!s}:{e3!s}"
                yield (
                    prim_set,
                    alt_set,
                    e2,
                    e3,
                    which,
                    coord,
                )
            elif (which == "AFE" and gene.strand == "+") or (
                which == "ALE" and gene.strand == "-"
            ):
                prim_nodes, alt_nodes = sorted((i, j))
                prim_set = events[pre][prim_nodes]
                alt_set = events[pre][alt_nodes]
                # SUPPA2-like coordinates
                s3 = gene.segment_graph[pre].start
                s1 = gene.segment_graph[prim_nodes[0][0]].start
                e1 = gene.segment_graph[prim_nodes[-1][1]].end
                s2 = gene.segment_graph[alt_nodes[0][0]].start
                e2 = gene.segment_graph[alt_nodes[-1][1]].end
                coord = f"{s1!s}:{e1!s}-{s3!s}:{s2!s}:{e2!s}-{s3!s}"
                yield (
                    prim_set,
                    alt_set,
                    s1,
                    s2,
                    which,
                    coord,
                )


def test_ale_afe(
    transcriptome,
    gene,
    groups: dict,
    min_total: int = 100,
    min_alt_fraction: float = 0.01,  # different from Isotools default
    min_n: int = 5,
    min_sa: float = 0.51,
    test="auto",  # either string with test name or a custom test function
):
    """Test for alternative last/first exon (ALE/AFE) events

    Reimplementation of isotools._transcriptome_stats.altsplice_test for ALE
    and AFE events, which are not supported in the current IsoTools version
    (2.0.0). IsoTools reports alternative PAS or TSS events by taking all
    pairwise combinations of PAS/TSS sites for a given gene, does not account
    for splice junctions. ALE events are hence conflated with alternative PAS
    events that do not involve splicing, likewise for AFE and TSS.

    This function identifies ALE and AFE events with relevant splice junctions,
    then applies the same statistical tests as
    isotools._transcriptome_stats.altsplice_test. Alternative PAS testing is
    implemented in isotools_extensions.alt_pas.test_alternative_pas.

    Alternative first/last exon events with multiple exons are supported. See
    docstring for `find_ale_afe_pairs` and `get_ale_afe` for details.

    :param transcriptome: isotools.Transcriptome object
    :param gene: isotools.Gene object
    :param groups: Dictionary mapping group names to lists of sample names
    :param min_total: Minimum total coverage across all samples to consider an
        event
    :param min_alt_fraction: Minimum fraction of reads supporting the
        alternative form (between 0 and 0.5)
    :param min_n: Minimum coverage in each group to consider an event
    :param min_sa: Minimum number of samples with sufficient coverage to
        consider an event. If between 0 and 1, interpreted as fraction of total
        samples.
    :param test: Statistical test to use. Either a string with the test name
        (see isotools._transcriptome_stats.TESTS) or a custom test function
        with signature test(x: list, n: list) -> (pvalue: float, params: tuple).
        If "auto", selects betabinom_lr if there are at least two samples per
        group, otherwise uses proportions test.
    :returns pd.DataFrame: DataFrame with test results for all identified
        ALE/AFE events. Columns are identical with
        isotools._transcriptome_stats.altsplice_test output except for "coord",
        which reports SUPPA2-like coordinates for ALE/AFE events. "splice_type"
        is either "ALE" or "AFE".
    """
    # There should be only two groups
    groupnames, groups, grp_idx = _check_groups(transcriptome, groups)
    sidx = np.array(grp_idx[0] + grp_idx[1])
    if isinstance(test, str):
        if test == "auto":
            test = (
                TESTS["betabinom_lr"]
                if min(len(group) for group in groups[:2]) > 1
                else TESTS["proportions"]
            )
        else:
            try:
                test = TESTS[test]
            except KeyError as e:
                e.add_note(f"Test name {test} not found")
                raise
    if min_sa < 1:
        min_sa *= sum(len(group) for group in groups[:2])
    res = []
    for which in ["ALE", "AFE"]:
        for setA, setB, start, end, splice_type, coord in find_ale_afe_pairs(
            gene, which
        ):
            junction_cov = gene.coverage[:, setB].sum(1)
            total_cov = gene.coverage[:, setA].sum(1) + junction_cov
            if total_cov[sidx].sum() < min_total:
                continue
            alt_fraction = junction_cov[sidx].sum() / total_cov[sidx].sum()
            if alt_fraction < min_alt_fraction or alt_fraction > 1 - min_alt_fraction:
                continue
            x = [junction_cov[grp] for grp in grp_idx]
            n = [total_cov[grp] for grp in grp_idx]
            if sum((ni >= min_n).sum() for ni in n[:2]) < min_sa:
                continue
            pval, params = test(x[:2], n[:2])
            if np.isnan(pval):
                continue
            # TODO: nmdA, nmdB
            covs = [val for lists in zip(x, n) for pair in zip(*lists) for val in pair]
            res.append(
                [
                    gene.name,
                    gene.id,
                    gene.chrom,
                    gene.strand,
                    start,
                    end,
                    which,
                    pval,
                    setA,
                    setB,
                    x,
                    n,
                    coord,
                    *list(params),
                    *covs,
                ]
            )
    colnames = [
        "gene",
        "gene_id",
        "chrom",
        "strand",
        "start",
        "end",
        "splice_type",
        "pvalue",
        "trA",
        "trB",
        "x",
        "n",
        "coord",
    ]
    colnames += [
        groupname + part
        for groupname in groupnames[:2] + ["total"] + groupnames[2:]
        for part in ["_PSI", "_disp"]
    ]
    colnames += [
        f"{sample}_{groupname}_{w}"
        for groupname, group in zip(groupnames, groups)
        for sample in group
        for w in ["_in_cov", "_total_cov"]
    ]
    return pd.DataFrame(res, columns=colnames)
