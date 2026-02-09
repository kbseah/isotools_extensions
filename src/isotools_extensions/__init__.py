# SPDX-FileCopyrightText: 2026-present Brandon Seah <brandon_seah@tll.org.sg>
#
# SPDX-License-Identifier: MIT

from isotools import Transcriptome, Gene
from .alt_pas import test_alternative_pas
from .ale_afe import test_ale_afe
from .gene_plots import (
    domains_figure,
    domains_figure_altsplice_result,
    sashimi_figure_altsplice_result,
)

# Monkeypatch
Transcriptome.test_alternative_pas = test_alternative_pas
Transcriptome.test_ale_afe = test_ale_afe
Gene.domains_figure = domains_figure
Gene.domains_figure_altsplice_result = domains_figure_altsplice_result
Gene.sashimi_figure_altsplice_result = sashimi_figure_altsplice_result
