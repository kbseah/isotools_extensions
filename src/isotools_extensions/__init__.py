# SPDX-FileCopyrightText: 2026-present Brandon Seah <brandon_seah@tll.org.sg>
#
# SPDX-License-Identifier: MIT

from isotools import Transcriptome
from .alt_pas import test_alternative_pas
from .ale_afe import test_ale_afe

# Monkeypatch
Transcriptome.test_alternative_pas = test_alternative_pas
Transcriptome.test_ale_afe = test_ale_afe
