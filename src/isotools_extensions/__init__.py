# SPDX-FileCopyrightText: 2026-present Brandon Seah <brandon_seah@tll.org.sg>
#
# SPDX-License-Identifier: MIT

from isotools import Gene, Transcriptome

from .__about__ import __version__
from .ale_afe import test_ale_afe
from .alt_pas import test_alternative_pas, test_diff_lastexon_len
from .domains import add_hmmer_domains_parallel
from .gene_plots import (
    domains_figure,
    domains_figure_altsplice_result,
    plot_gene_terminal_peaks,
    plot_gene_terminal_pileup,
    plot_transcript_terminal_peaks,
    plot_transcript_terminal_pileup,
    sashimi_figure_altsplice_result,
)
from .psi_test import test_ale_psi, test_apa_psi
from .io import to_gtf

# Monkeypatch
Transcriptome.test_alternative_pas = test_alternative_pas
Transcriptome.test_ale_afe = test_ale_afe
Transcriptome.test_diff_lastexon_len = test_diff_lastexon_len
Transcriptome.add_hmmer_domains_parallel = add_hmmer_domains_parallel
Transcriptome.test_ale_psi = test_ale_psi
Transcriptome.test_apa_psi = test_apa_psi
Gene.domains_figure = domains_figure
Gene.domains_figure_altsplice_result = domains_figure_altsplice_result
Gene.sashimi_figure_altsplice_result = sashimi_figure_altsplice_result
Gene.plot_gene_terminal_peaks = plot_gene_terminal_peaks
Gene.plot_gene_terminal_pileup = plot_gene_terminal_pileup
Gene.plot_transcript_terminal_peaks = plot_transcript_terminal_peaks
Gene.plot_transcript_terminal_pileup = plot_transcript_terminal_pileup
Gene._to_gtf = to_gtf
