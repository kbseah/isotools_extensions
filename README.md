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

This package relies on a [fork](https://github.com/kbseah/IsoTools2) that
implements some bug fixes not yet incorporated into the upstream repository.

Please cite the original IsoTools papers and this repository if you use it.


## TODO

* Simplify ALE/AFE test to consider only single terminal exons
* Consider terminal exons with same distal end to be the same, because of 5'-
  and 3'-fragmentation
* Filter out transcripts annotated as fragments during the test procedure
* Report PAS peak calls as features in GTF or GFF format along with their
  downstream A content; should parent features be exon, transcript, or gene?
* Rewrite extended functions as subclass or mixin methods for cleaner
  inheritance.
