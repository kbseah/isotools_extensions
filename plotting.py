#!/usr/bin/env python3

import matplotlib.pyplot as plt
import isotools._utils
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict


def plot_transcript_terminal_pileup(
    gene, trid: int, which="PAS", total: bool = False, show_unified: bool = False
):
    """Plot pileup of PAS or TSS for an isotools transcript

    :param gene: isotools.Gene object
    :param trid: Transcript index
    :param which: Either "PAS" or "TSS"
    :param total: Sum pileups for all samples if True, else plot samples separately
    :param show_unified: Overlay unified TSS/PAS consensus value from IsoTools
    :returns: Figure and axis objects
    """
    try:
        pileups = gene.transcripts[trid][which]
        unified = gene.transcripts[trid][which + "_unified"]
        if total:
            fig, ax = plt.subplots(1, 1, figsize=(8, 2))
            pileup = {}
            for sample in pileups:
                for idx in pileups[sample]:
                    pileup[idx] = pileup.get(idx, 0) + pileups[sample][idx]
            xx = list(pileup.keys())
            yy = list(pileup.values())
            ax.vlines(xx, ymin=[0 for y in yy], ymax=yy)
        else:
            fig, ax = plt.subplots(len(pileups), 1, figsize=(8, 6), sharex=True)
            if show_unified:
                for i, sample in enumerate(unified):
                    xx = list(unified[sample].keys())
                    yy = list(unified[sample].values())
                    ax[i].vlines(xx, ymin=[0 for y in yy], ymax=yy, color="green")
            for i, sample in enumerate(pileups):
                xx = list(pileups[sample].keys())
                yy = list(pileups[sample].values())
                ax[i].vlines(xx, ymin=[0 for y in yy], ymax=yy)
        return (fig, ax)
    except KeyError:
        raise KeyError("Parameter `which` must be either 'PAS' or 'TSS'")


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
    pileup = {}
    coords = {}
    smoothed = {}
    peaks = {}
    if total:
        pileup_sum = {}
        for sample in gene.transcripts[trid][which]:
            for pos, cov in gene.transcripts[trid][which][sample].items():
                pileup_sum[pos] = pileup.get(pos, 0) + cov
        pileup["total"], coords["total"], smoothed["total"] = pileup_to_smoothed(
            pileup_sum, smooth_window
        )
        peaks["total"] = find_peaks(smoothed["total"], prominence=(prominence, None))
    else:
        for sample in gene.transcripts[trid][which]:
            pileup[sample], coords[sample], smoothed[sample] = pileup_to_smoothed(
                gene.transcripts[trid][which][sample], smooth_window
            )
            peaks[sample] = find_peaks(smoothed[sample], prominence=(prominence, None))
    return (pileup, coords, smoothed, peaks)


def plot_transcript_terminal_peaks(
    gene,
    trid,
    which="PAS",
    total=False,
    smooth_window=31,
    show_peaks=True,
    prominence=2,
):
    """Plot smoothed PAS/TSS pileups and called peaks for an isotools transcript

    Unlike default isotools behavior, this calls all peaks without choosing a
    unified consensus for each transcript.

    :param gene: isotools.Gene object
    :param trid: Transcript index
    :param which: Either "PAS" or "TSS"
    :param total: Sum pileups for all samples if True, else plot samples separately
    :param smooth_window: Window size for smoothing function
    :param prominence: Minimum peak prominence to retain
    """
    pileup, coords, smoothed, peaks = get_transcript_terminal_peaks(
        gene=gene,
        trid=trid,
        which=which,
        total=total,
        smooth_window=smooth_window,
        prominence=prominence,
    )
    fig, ax = plt.subplots(
        len(smoothed), 1, figsize=(8, 2 * len(smoothed)), sharex=True
    )
    for idx, s in enumerate(smoothed):
        if len(smoothed) == 1:
            myax = ax
        else:
            myax = ax[idx]
        myax.plot(coords[s], smoothed[s])
        if show_peaks:
            myax.vlines(
                [x + min(coords[s]) for x in peaks[s][0]],
                ymin=[0 for i in peaks[s][0]],
                ymax=peaks[s][1]["prominences"],
                color="red",
            )
    return (fig, ax, pileup, smoothed, peaks)
