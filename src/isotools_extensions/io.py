import logging

from isotools import Gene

logger = logging.getLogger("isotools")


def get_cds_features(
    self:Gene,
    transcript_id:int,
    reference:bool=False,
) -> list[tuple]|None:
    """Report CDS feature coordinates and frame for a transcript.

    For GTF export; refer to
    https://nbisweden.github.io/AGAT/gxf/#gtf2-and-gff25 for GTF definition.
    Not fully compliant because stop codons are not removed from the CDS
    features.

    :param self: isotools.Gene object.
    :param transcript_id: Index of transcript in isotools.Gene.transcripts.
    :param reference: True if ref_transcript should be reported.
    :returns list: List of tuples, each with start, end coordinates, strand,
        and frame. Frame is defined as for GTF format. Start and
        end-coordinates are in 0-based python-type indices, should be converted
        to 1-based indices for GTF output.
    """
    transcript = self.ref_transcripts[transcript_id] if reference else self.transcripts[transcript_id]
    strand = self.strand
    try:
        # Try CDS key first
        orf_s, orf_e = transcript.get("CDS", transcript.get("ORF"))[0:2]
    except TypeError:
        # No ORF feature predicted in the transcript, skip
        return None
    out = []
    frame = 0 # Initialize frame counter
    for es, ee in transcript.get("exons"):
        cds_s, cds_e = None, None
        if ee < orf_s or orf_e < es:
            # exon entirely within UTRs
            continue
        cds_s = max(es, orf_s) # first cds feature
        cds_e = min(orf_e, ee) # last cds_feature
        # if negative strand, update frame before adding
        if strand == "+":
            out.append((cds_s, cds_e, strand, frame))
            # update frame
            frame = (frame + (cds_e - cds_s)) % 3
        elif strand == "-":
            frame = (frame + (cds_e - cds_s)) % 3
            out.append((cds_s, cds_e, strand, frame))
    # TODO Remove stop codon. Note that start/stop codon may span intron
    # boundary in rare cases
    return out


def to_gtf(
    self,
    transcript_ids:list,
    ref_transcript_ids:list|None=None,
    source:str="isoseq",
    ref_source:str="annotation",
)->list[tuple]:
    """Create the GTF lines of the gene as tuples.

    Drop-in substitute for existing isotools.Gene._to_gtf, to report CDS
    features too. start/stop codons and UTRs not yet implemented.

    Features exported: gene, transcript, exon, CDS. `transcript_id` is gene_id
    appended with transcript index. `exon_id` is transcript_id appended with
    exon index. `cds_id` is transcript_id appended with "CDS" and cds index, to
    avoid overlap with exon_id.

    :param self: isotools.Gene object
    :param transcript_ids: List of indices for isotools transcripts.
    :param ref_transcript_ids: List of reference transcript indices. Default is
        to skip ref transcripts. If supplied, only those ref transcripts not
        already identified as FSM with isotools transcripts are reported.
    :param source: Value for `source` field of GTF for transcripts.
    :param ref_source: Value for `source` field of GTF for ref_transcripts.
    :returns list: List of tuples, each representing a GTF line.
    """
    donotshow = {"transcripts", "short_exons", "segment_graph"}
    info = {"gene_id": self.id, "gene_name": self.name}
    lines = [None]
    starts = []
    ends = []
    ref_fsm = []

    for i in transcript_ids:
        transcript = self.transcripts[i]
        cdss = get_cds_features(self, i, reference=False)
        info["transcript_id"] = f'{info["gene_id"]}_{i}'
        starts.append(transcript["exons"][0][0] + 1)
        ends.append(transcript["exons"][-1][1])
        transcript_info = info.copy()
        if "downstream_A_content" in transcript:
            transcript_info["downstream_A_content"] = (
                f'{transcript["downstream_A_content"]:0.3f}'
            )
        if transcript["annotation"][0] == 0:  # FSM
            refinfo = {}
            for refid in transcript["annotation"][1]["FSM"]:
                ref_fsm.append(refid)
                for k in self.ref_transcripts[refid]:
                    if k == "exons":
                        continue
                    if k == "CDS":
                        if self.strand == "+":
                            cds_start, cds_end = self.ref_transcripts[refid]["CDS"]
                        else:
                            cds_end, cds_start = self.ref_transcripts[refid]["CDS"]
                        refinfo.setdefault("CDS_start", []).append(str(cds_start))
                        refinfo.setdefault("CDS_end", []).append(str(cds_end))
                    else:
                        refinfo.setdefault(k, []).append(
                            str(self.ref_transcripts[refid][k]),
                        )
            for k, vlist in refinfo.items():
                transcript_info[f"ref_{k}"] = ",".join(vlist)
        else:
            transcript_info["novelty"] = ",".join(
                k for k in transcript["annotation"][1]
            )
        lines.append(
            (
                self.chrom,
                source,
                "transcript",
                transcript["exons"][0][0] + 1,
                transcript["exons"][-1][1],
                ".",
                self.strand,
                ".",
                "; ".join(f'{k} "{v}"' for k, v in transcript_info.items()),
            ),
        )
        noncanonical = transcript.get("noncanonical_splicing", [])
        for enr, pos in enumerate(transcript["exons"]):
            exon_info = info.copy()
            exon_info["exon_id"] = f'{info["gene_id"]}_{i}_{enr}'
            if enr in noncanonical:
                exon_info["noncanonical_donor"] = noncanonical[enr][:2]
            if enr + 1 in noncanonical:
                exon_info["noncanonical_acceptor"] = noncanonical[enr + 1][2:]
            lines.append(
                (
                    self.chrom,
                    source,
                    "exon",
                    pos[0] + 1,
                    pos[1],
                    ".",
                    self.strand,
                    ".",
                    "; ".join(f'{k} "{v}"' for k, v in exon_info.items()),
                ),
            )
        # add CDS features
        if cdss is not None:
            for cnr, (cds_s, cds_e, _strand, cds_frame) in enumerate(cdss):
                cds_info = info.copy()
                cds_info["CDS_id"] = f'{info["gene_id"]}_{i}_CDS{cnr}'
                lines.append(
                    (
                        self.chrom,
                        source,
                        "CDS",
                        cds_s + 1,
                        cds_e,
                        ".",
                        self.strand,
                        cds_frame,
                        "; ".join(f'{k} "{v}"' for k, v in cds_info.items()),
                    ),
                )

    if ref_transcript_ids:
        # add reference transcripts not covered by FSM
        for i, transcript in enumerate(self.ref_transcripts):
            if i in ref_fsm:
                continue
            cdss = get_cds_features(self, i, reference=True)
            starts.append(transcript["exons"][0][0] + 1)
            ends.append(transcript["exons"][-1][1])
            info["transcript_id"] = f'{info["gene_id"]}_ref{i}'
            refinfo = info.copy()
            for k in transcript:
                if k == "exons":
                    continue
                if k == "CDS":
                    if self.strand == "+":
                        cds_start, cds_end = transcript["CDS"]
                    else:
                        cds_end, cds_start = transcript["CDS"]
                    refinfo["CDS_start"] = str(cds_start)
                    refinfo["CDS_end"] = str(cds_end)
                else:
                    refinfo.setdefault(k, []).append(str(transcript[k]))
                lines.append(
                    (
                        self.chrom,
                        ref_source,
                        "transcript",
                        transcript["exons"][0][0] + 1,
                        transcript["exons"][-1][1],
                        ".",
                        self.strand,
                        ".",
                        "; ".join(f'{k} "{v}"' for k, v in refinfo.items()),
                    ),
                )
                for enr, pos in enumerate(transcript["exons"]):
                    exon_info = info.copy()
                    exon_id = f'{info["gene_id"]}_ref{i}_{enr}'
                    lines.append(
                        (
                            self.chrom,
                            ref_source,
                            "exon",
                            pos[0] + 1,
                            pos[1],
                            ".",
                            self.strand,
                            ".",
                            f'exon_id "{exon_id}"',
                        ),
                    )
                # add CDS features
                if cdss is not None:
                    for cnr, (cds_s, cds_e, _strand, cds_frame) in enumerate(cdss):
                        cds_info = info.copy()
                        cds_info["CDS_id"] = f'{info["gene_id"]}_{i}_CDS{cnr}'
                        lines.append(
                            (
                                self.chrom,
                                source,
                                "CDS",
                                cds_s + 1,
                                cds_e,
                                ".",
                                self.strand,
                                cds_frame,
                                "; ".join(f'{k} "{v}"' for k, v in cds_info.items()),
                            ),
                        )

    if len(lines) > 1:
        # add gene line
        if "reference" in self.data:
            # add reference gene specific fields
            info.update(
                {
                    k: v
                    for k, v in self.data["reference"].items()
                    if k not in donotshow
                },
            )
        lines[0] = (
            self.chrom,
            source,
            "gene",
            min(starts),
            max(ends),
            ".",
            self.strand,
            ".",
            "; ".join(
                f'{k} "{v}"' for k, v in info.items() if k != "transcript_id"
            ),
        )
        return lines
    return []
