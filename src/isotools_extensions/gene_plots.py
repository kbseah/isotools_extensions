import matplotlib.pyplot as plt
import numpy as np


def sashimi_figure_altsplice_result(
    self,
    groups: dict,
    diff_splice_result,
    flank=2000,
    start=None,
    end=None,
    query="(FSM or not (RTTS or INTERNAL_PRIMING or FRAGMENT)) and SUBSTANTIAL",
):
    """Sashimi plot and gene tracks figure for isoforms involved in a single
    alternative splicing event

    Augment the sashimi plot with gene tracks for isoforms of interest to help
    evaluate an alternative splicing event. The AS event must have previously
    been called by a function like isotools.Transcriptome.altsplice_test.
    Relevant row from the output DataFrame is passed to the
    `diff_splice_result` parameter for plotting. The relevant columns are trA,
    trB, start, end.

    :param self: isotools.Gene object
    :param groups: Dict of groups to samples
    :param diff_splice_result: Single AS result from the differential splicing
        analysis results table, either as tuple generated from .itertuples()
        method or as pandas Series e.g. from .iloc method
    :param flank: Distance (bp) flanking start/end coordinates of AS event to
        pad plot
    :param start,end: Override default start/end coordinates; if specified,
        'flank' parameter is ignored
    :param query: Query string passed to filter_transcripts
    :returns: (fig, axs) tuple of matplotlib Figure and Axes objects
    """
    if start is None:
        start = diff_splice_result.start - flank
    if end is None:
        end = diff_splice_result.end + flank
    pos = [start, end]

    # Get transcript IDs for transcripts supporting the alternative splice forms
    # and reference transcripts
    transcripts_A = list(
        set(self.filter_transcripts(query=query)).intersection(
            set(diff_splice_result.trA)
        )
    )
    transcripts_B = list(
        set(self.filter_transcripts(query=query)).intersection(
            set(diff_splice_result.trB)
        )
    )
    transcripts_C = self.ref_transcripts

    # Scale plot according to number of gene tracks to plot
    height_ratios = (
        [0.2 + 0.3 * len(transcripts_A), 0.2 + 0.3 * len(transcripts_B)]
        + [0.2 + 0.3 * len(transcripts_C)]
        + [1.5] * len(groups)
    )
    fig_height = sum(height_ratios)
    fig_width = 10
    gs_kw = dict(height_ratios=height_ratios)
    fig, axs = plt.subplots(
        len(height_ratios), figsize=(fig_width, fig_height), gridspec_kw=gs_kw
    )

    # Add gene tracks for transcripts
    self.gene_track(
        x_range=pos, ax=axs[0], reference=False, select_transcripts=transcripts_A
    )
    self.gene_track(
        x_range=pos, ax=axs[1], reference=False, select_transcripts=transcripts_B
    )
    self.gene_track(
        x_range=pos,
        ax=axs[2],
        reference=True,
    )
    axs[3].annotate(diff_splice_result.splice_type, xy=(0, 1), xycoords="axes fraction")
    ax_idx = 3

    # Add sashimi plot zoomed in on AS event
    for group in groups:
        self.sashimi_plot(
            samples=groups[group],
            x_range=pos,
            ax=axs[ax_idx],
            title=group,
            log_y=False,
        )
        # Underline to highlight regions with differential splicing
        # Color red if p_adj < 0.05
        if diff_splice_result is not None:
            axs[ax_idx].vlines(
                x=[diff_splice_result.start, diff_splice_result.end],
                ymin=-10000,
                ymax=10000,
                color="red",
            )
        ax_idx += 1

    fig.tight_layout()
    return (fig, axs)


def domains_figure(
    self,
    groups: dict,
    source="hmmer",
    query="FSM or SUBSTANTIAL",
    transcript_ids=False,
    cov_color="grey",
    ref_transcript_ids=False,
    highlight=None,
    height_factor=0.75,
    **kwargs,
):
    """Generate figure of domains alongside scatterplot of coverage per group

    The domains plot will be plotted alongside a scatterplot where size of plot
    elements is proportional to the reads coverage of the given transcript per
    sample. SQANTI categories of each transcript is also indicated.

    :param self: isotools.Gene object
    :param groups: Transcriptome's dict of groups
    :param source: Source of protein domains, e.g. "annotation", "hmmer" or
        "interpro", for domains added by the functions
        "add_annotation_domains", "add_hmmer_domains" or "add_interpro_domains"
        respectively.
    :param query: Query to filter transcripts by TAGs
    :param transcript_ids: List of transcript IDs; if False, defer to the query
    :param ref_transcript_ids: List of reference transcript ids to plot; if
        False, reference transcripts are not plotted.
    :param highlight: Coordinates to highlight, passed to plot_domains
    :param cov_color: Either a color name string, or a list of color name
        strings for the coverage plot elements (of same length as the list of
        transcript_ids)
    :param height_factor: Adjustment factor for figure height.
    :param **kwargs: Other parameters passed to plot_domains
    :returns: (fig, axs) tuple of matplotlib Figure and Axes objects
    """
    if transcript_ids:
        if isinstance(transcript_ids, list):
            trids = transcript_ids
        else:
            trids = list(range(self.n_transcripts))
    else:
        trids = self.filter_transcripts(query)
    if ref_transcript_ids:
        if isinstance(ref_transcript_ids, list):
            n_ref_transcripts = len(ref_transcript_ids)
        else:
            n_ref_transcripts = self.n_ref_transcripts
    else:
        n_ref_transcripts = 0
    if len(trids) == 0:
        raise ValueError("Empty set after query filter")
    # Check if color array is correct length and type
    if isinstance(cov_color, list):
        assert len(cov_color) == len(
            trids
        ), "cov_color must be same length as number of transcripts to plot"
        assert all(
            [isinstance(j, str) for j in cov_color]
        ), "cov_color must be a list of color name strings"
    # Plot parameters
    fig_height = height_factor * (len(trids) + n_ref_transcripts)
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(10, fig_height),
        sharey=True,
        gridspec_kw=dict(width_ratios=[5, 2]),
    )
    # Plot domains
    self.plot_domains(
        source=source,
        ref_transcript_ids=ref_transcript_ids,
        transcript_ids=trids,
        label="name",
        include_utr=True,
        coding_only=False,
        separate_exons=True,
        highlight=highlight,
        ax=axs[0],
        **kwargs,
    )
    axs[0].set_title("domains")
    # Coverage per isoform per group, as scatter plot blobs
    # transpose: rows represent transcripts
    arr = np.array(
        [
            self.get_infos(trid, ["group_coverage_sum"], None, range(len(groups)))
            for trid in trids
        ]
    )
    # Initialize array for color strings
    cc = []
    # if reference transcripts, add blanks
    # ref transcripts precede observed
    if ref_transcript_ids:
        ngroup = arr.shape[1]
        arr = np.concatenate(
            [
                np.zeros([n_ref_transcripts, ngroup], dtype=int),
                arr,
            ]
        )

    yy = []
    xx = []
    ss = []
    for row in range(len(arr)):
        for col in range(len(arr[row])):
            yy.append(0 - row)  # from top to bottom
            xx.append(col)
            ss.append(10 * arr[row, col] ** 0.5)
            if isinstance(cov_color, str):
                cc.append(cov_color)
            elif isinstance(cov_color, list):
                cc.append(cov_color[row - n_ref_transcripts])
    axs[1].scatter(x=xx, y=yy, s=ss, c=cc, linewidths=0)
    axs[1].set_xlim(min(xx) - 1, max(xx) + 1)
    axs[1].set_title("coverage")
    axs[1].set_xticks(ticks=range(len(groups)), labels=list(groups.keys()))
    # label with SQANTI categories of each transcript
    annot2cat = {
        0: "FSM",
        1: "ISM",
        2: "NIC",
        3: "NNC",
        4: "NOVEL",
    }
    categories = []
    if ref_transcript_ids:
        if isinstance(ref_transcript_ids, list):
            categories.extend(["REF" for i in ref_transcript_ids])
        else:
            categories.extend(["REF"] * self.n_ref_transcripts)
    categories.extend([annot2cat[self.transcripts[i]["annotation"][0]] for i in trids])
    axs_twinx = axs[1].twinx()
    axs_twinx.set_ylim(axs[1].get_ylim())
    axs_twinx.set_yticks(
        ticks=axs[1].get_yticks(),
        labels=categories,
    )
    fig.tight_layout()
    return (fig, axs)


def domains_figure_altsplice_result(
    self,
    groups: dict,
    diff_splice_result,
    source="hmmer",
    query="FSM or SUBSTANTIAL",
    transcript_ids=False,
    ref_transcript_ids=False,
    height_factor=0.75,
    **kwargs,
):
    """Generate figure of domains alongside coverage scatterplot, contrasting
    transcripts involved in AS event

    Refer to isotools.Gene.domains_figure documentation for details

    :param self: isotools.Gene object
    :param groups: Transcriptome's dict of groups
    :param diff_splice_result: Single AS result from the differential splicing
        analysis results table, either as tuple generated from .itertuples()
        method or as pandas Series
    :param source: Source of protein domains, e.g. "annotation", "hmmer" or
        "interpro", for domains added by the functions
        "add_annotation_domains", "add_hmmer_domains" or "add_interpro_domains"
        respectively.
    :param query: Query to filter transcripts by TAGs
    :param transcript_ids: List of transcript IDs; if False, defer to the query
    :param ref_transcript_ids: List of reference transcript ids to plot; if
        False, reference transcripts are not plotted.
    :param height_factor: Adjustment factor for figure height.
    :param **kwargs: Other parameters passed to plot_domains
    :returns: (fig, axs) tuple of matplotlib Figure and Axes objects
    """
    setA = list(
        set(self.filter_transcripts(query)).intersection(diff_splice_result.trA)
    )
    setB = list(
        set(self.filter_transcripts(query)).intersection(diff_splice_result.trB)
    )
    if len(setA) == 0 or len(setB) == 0:
        raise ValueError("Empty set after query filter")
    cov_color = ["darkblue"] * len(setA) + ["darkred"] * len(setB)
    fig, axs = self.domains_figure(
        groups=groups,
        source=source,
        query=query,
        transcript_ids=setA + setB,
        cov_color=cov_color,
        highlight=[diff_splice_result.start, diff_splice_result.end],
        ref_transcript_ids=ref_transcript_ids,
        height_factor=height_factor,
        **kwargs,
    )
    return fig, axs
