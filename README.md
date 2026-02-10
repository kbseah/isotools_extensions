IsoTools Extensions
===================

Extended functionality for [IsoTools2](https://github.com/HerwigLab/IsoTools2)
package for gene transcript isoform analysis. Specifically, it implements new
tests for alternative last/first exon usage (ALE, AFE) that is separate from
alternative polyadenylation site or transcription start site (PAS, TSS)
testing. A new test for alternative polyadenylation (APA) events is also
provided. In the original IsoTools package the available tests for PAS/TSS
conflated them with ALE/AFE, which are distinct phenomena.

A test for alternative TSS is not implemented because accurate annotation of
5'-terminal ends of mRNA is more challenging with current long read isoform
sequencing technologies than 3'-terminal ends. See discussion by
[Calvo-Roitberg et al., 2024](https://doi.org/10.1101/gr.279559.124).


## Installation

Install with `pip` into a Python virtual environment:

```bash
python -m venv ./my_env
source my_env/bin/activate
cd path/to/isotools_extensions
pip install .
```


## Usage

Import `isotools_extensions` after importing `isotools`.

Extended functions are monkeypatched to `isotools.Gene` and
`isotools.Transcriptome` classes:

 * `Transcriptome.test_alternative_pas` -- Test alternative PAS events
 * `Transcriptome.test_ale_afe` -- Test ALE/AFE events
 * `Gene.domains_figure` -- Plot exons with annotated domains and coverage in
       one figure
 * `Gene.domains_figure_altsplice_result` -- Plot exons with annotated domains
       and coverage in one figure, coverage plots grouped and colored by
       alternative splicing event contrast
 * `Gene.sashimi_figure_altsplice_result` -- Plot sashimi figure and gene
       tracks, grouped and colored by alternative splicing event contrast
 * `Gene.plot_gene_terminal_peaks` -- Plot PAS position peak calls for a single
       gene
 * `Gene.plot_gene_terminal_pileup` -- Plot PAS position coverage pileups for a
       single gene
 * `Gene.plot_transcript_terminal_peaks` -- Plot PAS position peak calls for a
       single transcript
 * `Gene.plot_transcript_terminal_pileup` -- Plot PAS position coverage pileups
       for a single transcript


## TODO

* Report PAS peak calls as features in GTF or GFF format along with their
  downstream A content; should parent features be exon, transcript, or gene?


## License

[MIT](LICENSE)


## Citation

This package uses a [fork](https://github.com/kbseah/IsoTools2/tree/bugfix)
that implements some bug fixes not yet incorporated into the upstream
repository.

Please cite the original IsoTools papers and this repository if you use it.
