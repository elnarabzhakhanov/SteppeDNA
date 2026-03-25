"""
SteppeDNA - VCF Parsing & Batch Prediction
===========================================

This module processes VCF (Variant Call Format) files — the standard output from
DNA sequencing pipelines. When a patient's genome is sequenced, the sequencer
compares it to the human reference genome and outputs a VCF file listing every
position where the patient differs from the reference. Each line in a VCF file
represents one variant (mutation) and contains:

    CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT  SAMPLE
    13     32340300  .  A    G    99    PASS    .     GT:GQ   0/1:40

    - CHROM: chromosome number (e.g., 13 for BRCA2)
    - POS: genomic position on the chromosome (GRCh38 coordinates)
    - REF: the reference (normal) base at this position
    - ALT: the alternate (mutant) base the patient has
    - FILTER: whether the variant passed quality control ("PASS" = good)
    - FORMAT/SAMPLE: per-sample fields like genotype (GT) and quality (GQ)

This module handles three main tasks:

1. PARSING: Reading VCF lines and extracting the relevant fields.

2. COORDINATE CONVERSION: Converting genomic coordinates to biological coordinates.
   DNA sequencers report positions on the chromosome (genomic coordinates), but
   doctors and biologists think in terms of the gene's coding sequence (cDNA) and
   the resulting protein (amino acid positions). The conversion chain is:

       Genomic position (chr13:32340300)
           → cDNA position (c.7397)        via the gene's CDS exon map
           → Codon position (codon 2466)    by dividing cDNA pos by 3
           → Amino acid change (D2466G)     by translating ref/alt codons

   For genes on the reverse strand (BRCA1, PALB2, RAD51D), the VCF alleles must
   be complemented (A↔T, C↔G) because the VCF always reports the forward strand,
   but the gene is read from the reverse strand.

3. VARIANT CLASSIFICATION: Determining what type of mutation this is and how
   dangerous it is. There are 6 types, split into two tiers:

   TIER 1 — Rule-based (biology dictates the answer, no ML needed):
     - Frameshift: insertion/deletion that shifts the reading frame → almost
       always destroys the protein → Pathogenic (p=0.9999)
     - In-frame indel: insertion/deletion that preserves reading frame → severity
       depends on whether it hits a functional domain
     - Splice-site: mutation at an exon/intron boundary → disrupts mRNA splicing →
       canonical (+/-1-2bp) = Likely Pathogenic, near (+/-3-8bp) = VUS
     - Nonsense: single base change that creates a premature stop codon → truncates
       the protein → Pathogenic (p=0.9999)
     - Synonymous: single base change that does NOT change the amino acid →
       usually benign, but flagged if near a splice site

   TIER 2 — ML ensemble (for missense variants that change one amino acid):
     - Missense: single base change that changes one amino acid to another →
       runs through the full XGBoost + MLP ensemble pipeline with isotonic
       calibration, exactly the same as the /predict single-variant endpoint

4. BATCH PROCESSING: The /predict/vcf endpoint accepts an entire VCF file upload,
   iterates through every variant, classifies each one, and returns all results
   plus a compound heterozygosity warning if applicable.

5. COMPOUND HETEROZYGOSITY DETECTION: When two or more heterozygous (one normal
   copy + one mutant copy) variants are found in the same gene, this is clinically
   important. If the two mutations are on DIFFERENT copies of the gene ("trans"),
   the patient has NO working copy — this is called compound heterozygosity and
   can cause autosomal recessive disease. If both mutations are on the SAME copy
   ("cis"), one copy still works. Determining cis vs trans requires family
   sequencing (trio analysis with parents).
"""

# ─── STANDARD LIBRARY IMPORTS ─────────────────────────────────────────────────
import re        # Regular expressions for chromosome validation
import time      # Performance timing for endpoint latency tracking
import logging   # Structured logging for VCF processing events

# ─── THIRD-PARTY IMPORTS ─────────────────────────────────────────────────────
import numpy as np       # Numerical operations (probability clipping, array handling)
import xgboost as xgb    # XGBoost ML model (one half of our ensemble)
from fastapi.responses import JSONResponse  # HTTP error responses with status codes
from fastapi import APIRouter, UploadFile, File, Form  # FastAPI routing and file upload handling

# ─── INTERNAL IMPORTS (SteppeDNA backend modules) ────────────────────────────
from backend.models import (
    SUPPORTED_GENES, MAX_VCF_SIZE,       # Which genes we support + max upload size
    XGB_WEIGHT, NN_WEIGHT,               # Default ensemble blend weights (XGBoost vs MLP)
    _GENE_ENSEMBLE_WEIGHTS,              # Per-gene custom blend weights (e.g., BRCA1 uses MLP-only)
    _GENE_CALIBRATORS,                   # Per-gene isotonic calibrators for probability calibration
)
from backend.features import get_gene_data, build_feature_vector
from backend.explanations import compute_bootstrap_ci
# CODON_TABLE maps 3-letter DNA codons to amino acids (e.g., "ATG" → "Met")
# COMPLEMENT maps each base to its complement (A↔T, C↔G) for reverse-strand genes
from backend.constants import CODON_TABLE, COMPLEMENT
# Database storage disabled — privacy: no variant data is stored server-side
# from backend.database import record_vcf_upload

logger = logging.getLogger("steppedna")

# RefSeq transcript IDs used for building HGVS c. notation (the standard way to
# describe a mutation relative to the gene's coding DNA sequence).
# Example: NM_000059.4:c.7397A>G means "position 7397 in BRCA2's cDNA, A changed to G"
GENE_TRANSCRIPTS = {
    "BRCA1": "NM_007294.4",
    "BRCA2": "NM_000059.4",
    "PALB2": "NM_024675.4",
    "RAD51C": "NM_058216.3",
    "RAD51D": "NM_002878.4",
}

# FastAPI router — all endpoints defined in this file get the "VCF" tag in the API docs
router = APIRouter(tags=["VCF"])

# _metrics_lock is needed for updating _metrics in main.py.
# We import these from main at call time to avoid circular imports.


def _compute_risk_tier(probability: float) -> str:
    """Unified 5-tier risk logic used by both /predict and /predict/vcf.
    Aligned with ACMG 5-tier classification system.

    The ACMG (American College of Medical Genetics) defines 5 standard tiers
    for classifying genetic variants. We map our ML probability to these tiers:

        Probability > 0.9  -> "pathogenic"          (very likely disease-causing)
        Probability > 0.7  -> "likely_pathogenic"    (probably disease-causing)
        Probability < 0.1  -> "benign"               (very likely harmless)
        Probability < 0.3  -> "likely_benign"         (probably harmless)
        Otherwise          -> "uncertain"             (VUS -- not enough evidence)

    The "uncertain" middle zone (0.3-0.7) is intentionally wide to avoid
    over-confident predictions in cases where the model is unsure.

    Args:
        probability (float): A calibrated pathogenicity probability between 0 and 1,
            where 0 = certainly benign and 1 = certainly pathogenic.

    Returns:
        str: One of "pathogenic", "likely_pathogenic", "uncertain",
            "likely_benign", or "benign".
    """
    if probability > 0.9:
        return "pathogenic"
    elif probability > 0.7:
        return "likely_pathogenic"
    elif probability < 0.1:
        return "benign"
    elif probability < 0.3:
        return "likely_benign"
    return "uncertain"


def parse_vcf_line(line):
    """Parse a single VCF data line into its component fields.

    A VCF data line looks like this (tab-separated):
        13  32340300  .  A  G  99  PASS  .  GT:GQ  0/1:40

    This function extracts the biologically relevant fields and performs
    basic validation. It returns the raw ALT field which may contain
    comma-separated multi-allelic values (e.g. "T,G"). Splitting into
    individual alleles is handled by the /predict/vcf endpoint caller,
    so each allele gets its own prediction while sharing the same
    genotype context.

    Args:
        line (str): A single tab-separated VCF data line (not a header line).
            Must have at least 5 columns: CHROM, POS, ID, REF, ALT.
            Optionally has QUAL (col 5), FILTER (col 6), INFO (col 7),
            FORMAT (col 8), and SAMPLE (col 9+).

    Returns:
        tuple or None:
            On success, returns an 8-element tuple:
                - chrom (str): Normalized chromosome (e.g., "13", not "chr13")
                - pos (int): Genomic position on the chromosome (GRCh38)
                - ref (str): Reference allele, uppercased (e.g., "A")
                - alt (str): Alternate allele(s), uppercased, may be comma-separated
                  for multi-allelic sites (e.g., "T,G")
                - genotype (str or None): Genotype string from the SAMPLE column
                  (e.g., "0/1" for heterozygous, "1/1" for homozygous alt).
                  None if FORMAT/SAMPLE columns are missing.
                - vcf_filter (str): FILTER column value (e.g., "PASS", "LowQual")
                - vcf_warnings (list[str]): Any quality warnings (e.g., low GQ,
                  failed FILTER)
                - multi_sample (bool): True if the VCF has more than one sample
                  column (we only analyze the first sample)
            On failure (malformed line, invalid chromosome, etc.), returns None.
    """
    # VCF lines are tab-separated. The minimum required columns are:
    # CHROM(0), POS(1), ID(2), REF(3), ALT(4). Optional: QUAL(5), FILTER(6), INFO(7), FORMAT(8), SAMPLE(9+)
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None

    # Normalize chromosome: "chr13" → "13", "chrX" → "X"
    chrom = parts[0].lower().replace("chr", "")
    # Only accept valid human chromosomes (1-22, X, Y, MT)
    if not re.match(r'^([1-9]|1[0-9]|2[0-2]|X|Y|MT?)$', chrom, re.IGNORECASE):
        return None
    try:
        pos = int(parts[1])
    except ValueError:
        return None
    ref = parts[3].upper()
    # ALT may contain multiple alleles separated by commas (multi-allelic site),
    # e.g., "T,G" — we return the raw string and split later in the batch loop
    alt = parts[4].upper()

    # FILTER column tells us if the variant passed the sequencer's quality control.
    # "PASS" or "." means it passed; anything else (e.g., "LowQual") is a warning.
    vcf_filter = parts[6] if len(parts) > 6 else "."
    vcf_warnings = []
    if vcf_filter not in {"PASS", ".", ""}:
        vcf_warnings.append(f"VCF FILTER: {vcf_filter} — variant did not pass upstream QC")

    # Parse genotype (GT) and genotype quality (GQ) from FORMAT + SAMPLE columns.
    # FORMAT defines the field order (e.g., "GT:GQ:DP"), SAMPLE has the values (e.g., "0/1:40:30").
    # GT "0/1" means heterozygous (one ref allele + one alt allele).
    # GT "1/1" means homozygous alt (both copies are mutant).
    # GQ is a confidence score for the genotype call (higher = more confident, <20 = low).
    genotype = None
    gq = None
    multi_sample = False
    if len(parts) >= 10:
        # If there are columns beyond the first SAMPLE (col 9), this is a multi-sample VCF.
        # We only analyze the first sample — a limitation we warn the user about.
        if len(parts) > 10:
            multi_sample = True
        format_field = parts[8]
        sample_field = parts[9]
        format_keys = format_field.split(":")
        sample_vals = sample_field.split(":")
        # Extract GT (genotype) field by matching its position in the FORMAT string
        if "GT" in format_keys:
            gt_idx = format_keys.index("GT")
            if gt_idx < len(sample_vals):
                genotype = sample_vals[gt_idx]
        # Extract GQ (genotype quality) — low GQ means the sequencer is not confident
        if "GQ" in format_keys:
            gq_idx = format_keys.index("GQ")
            if gq_idx < len(sample_vals):
                try:
                    gq = int(sample_vals[gq_idx])
                    if gq < 20:
                        vcf_warnings.append(f"Low genotype quality (GQ={gq})")
                except (ValueError, TypeError):
                    pass

    return chrom, pos, ref, alt, genotype, vcf_filter, vcf_warnings, multi_sample


def vcf_variant_to_prediction(chrom, genomic_pos, ref_allele, alt_allele, gene_name="BRCA2", gene_data=None):  # noqa: C901
    """
    Convert a single VCF variant (one genomic position + allele change) into a
    pathogenicity prediction with clinical annotations.

    This is the core classification function. It takes raw VCF-level data (chromosome,
    position, reference allele, alternate allele) and performs the full pipeline:

        1. Validates the variant is on the correct chromosome for the target gene
        2. Checks if it's an INDEL (insertion/deletion) and classifies as frameshift
           or in-frame based on whether the length change is a multiple of 3
        3. If not in coding sequence (CDS), checks if it's near a splice site
        4. Converts genomic coordinates to cDNA and protein coordinates
        5. Handles strand complementing for reverse-strand genes
        6. Looks up the affected codon and translates to amino acids
        7. Classifies as synonymous, nonsense, or missense
        8. For missense: runs the full ML ensemble (XGBoost + MLP + calibration)

    Args:
        chrom (str): Chromosome number, normalized (e.g., "13", not "chr13").
        genomic_pos (int): Genomic position on the chromosome (GRCh38 coordinates).
            This is the POS field from the VCF file.
        ref_allele (str): Reference (normal) allele from the VCF, uppercased.
            Always reported on the forward strand of the chromosome.
        alt_allele (str): Alternate (mutant) allele from the VCF, uppercased.
            Also on the forward strand.
        gene_name (str): Which gene to analyze (default: "BRCA2"). Must be one of
            the 5 supported HR genes: BRCA1, BRCA2, PALB2, RAD51C, RAD51D.
        gene_data (dict or None): Pre-loaded gene data dictionary containing CDS
            sequence, exon maps, ML models, scalers, calibrators, etc. If None,
            it will be loaded via get_gene_data() (cached after first load).

    Returns:
        tuple: (result_dict, error_message) where exactly one is None:
            - On success: (dict, None) — dict contains prediction, probability,
              variant_type, risk_tier, HGVS notation, ACMG evidence codes, etc.
            - On skip/error: (None, str) — str explains why the variant was
              skipped (e.g., "Not on chr13", "Not in BRCA2 CDS", "Ref mismatch")

    Handles 6 variant types (classified in this order):
      - Frameshift (Tier 1 rule): p=0.9999, ACMG PVS1
      - In-frame indel (Tier 1 rule): domain-aware, ACMG PM4
      - Splice-site (Tier 1 rule): canonical +/-1-2bp or near +/-3-8bp
      - Nonsense (Tier 1 rule): p=0.9999, ACMG PVS1
      - Synonymous (Tier 1 rule): splice-proximity aware, ACMG BP7
      - Missense (Tier 2 ML): full XGBoost + MLP ensemble prediction
    """
    # Load gene-specific data (CDS sequence, exon maps, ML models, etc.)
    # This is cached after first load so repeated calls are fast.
    if gene_data is None:
        gene_data = get_gene_data(gene_name)

    # Gene config contains: chromosome number, strand direction (+/-), and functional domains.
    gene_config = gene_data.get("gene_config", {})
    if not gene_config:
        return None, f"Gene configuration not found for {gene_name}"
    target_chrom = str(gene_config.get("chromosome", "13"))
    # Strand tells us which DNA strand the gene is read from.
    # Forward (+): BRCA2, RAD51C — VCF alleles match the coding strand directly.
    # Reverse (-): BRCA1, PALB2, RAD51D — VCF alleles must be complemented (A↔T, C↔G).
    target_strand = gene_config.get("strand", "+")

    # Quick filter: skip variants not on this gene's chromosome
    if chrom != target_chrom:
        return None, f"Not on chr{target_chrom}"

    # genomic_to_cdna is a dictionary mapping each genomic position (chromosome coordinate)
    # to its cDNA position (position within the gene's coding sequence).
    # Only positions that fall within exons (coding regions) are in this map.
    # Intronic positions are NOT in this map — they'll be checked for splice-site proximity.
    genomic_to_cdna = gene_data.get("genomic_to_cdna", {})
    if not genomic_to_cdna:
        return None, "Genomic mapping not loaded"

    # Exon data includes splice site information for intronic variants near exon boundaries
    exon_data = gene_data.get("exon_data", {})

    # ─── INDEL CHECK ───────────────────────────────────────────────────────────
    # An INDEL (insertion/deletion) is when ref and alt have different lengths.
    # Example: REF="ACT" ALT="A" is a 2-base deletion; REF="A" ALT="ATG" is a 2-base insertion.
    # If ref and alt are both length 1, it's a single nucleotide variant (SNV), not an indel.
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        len_diff = abs(len(ref_allele) - len(alt_allele))
        if len_diff % 3 != 0:
            # FRAMESHIFT: The length difference is not a multiple of 3.
            # Since codons are 3 bases long, a non-multiple-of-3 indel shifts the entire
            # reading frame downstream, producing a completely wrong protein from this point
            # onward. This almost always destroys the protein → Pathogenic with very high
            # confidence. ACMG evidence code PVS1 (very strong pathogenic evidence).
            return {
                "cdna_pos": genomic_to_cdna.get(genomic_pos, 0),
                "aa_pos": 0,
                "aa_ref": "Indel",
                "aa_alt": "fs",
                "mutation": "delins",
                "hgvs_p": "p.(fs)",
                "prediction": "Pathogenic",
                "probability": 0.9999,
                "risk_tier": "pathogenic",
                "risk_detail": "Frameshift Truncation",
                "genomic_pos": genomic_pos,
                "variant_type": "frameshift",
                "classification_method": "rule_based",
                "acmg_evidence": {"PVS1": "Frameshift null variant causing premature truncation"},
                "comments": "Frameshift mutation resulting in likely premature truncation."
            }, None
        else:
            # ─── IN-FRAME INDEL: domain-aware rule-based classification ────────
            # The length difference IS a multiple of 3, so the reading frame is preserved.
            # The protein loses or gains whole amino acids, but downstream sequence is intact.
            # Severity depends on WHERE the indel falls: inside a functional domain = worse.
            cdna_pos_approx = genomic_to_cdna.get(genomic_pos, 0)
            # Convert cDNA position to amino acid position: (cDNA-1)//3 + 1 (1-based)
            aa_pos_approx = (cdna_pos_approx - 1) // 3 + 1 if cdna_pos_approx > 0 else 0
            # How many amino acids are inserted or deleted
            aa_change = len_diff // 3

            # Check if the affected amino acid falls within a known functional domain.
            # Functional domains are critical protein regions (e.g., DNA-binding domain,
            # BRCT domain in BRCA1). Indels inside domains are more likely to be damaging.
            in_domain = False
            domain_name = None
            if aa_pos_approx > 0:
                for dname, drange in gene_config.get("domains", {}).items():
                    if drange[0] <= aa_pos_approx <= drange[1]:
                        in_domain = True
                        domain_name = dname
                        break

            if in_domain:
                # In-frame indel INSIDE a functional domain → Likely Pathogenic (PM4 evidence).
                # Deleting/inserting amino acids in a domain likely disrupts its function.
                return {
                    "cdna_pos": cdna_pos_approx,
                    "aa_pos": aa_pos_approx,
                    "aa_ref": "Indel",
                    "aa_alt": f"inframe({aa_change}aa)",
                    "mutation": "delins",
                    "hgvs_p": f"p.({aa_pos_approx}delins)",
                    "prediction": "Likely Pathogenic",
                    "probability": 0.80,
                    "risk_tier": "likely_pathogenic",
                    "risk_detail": "In-Frame Indel in Functional Domain",
                    "genomic_pos": genomic_pos,
                    "variant_type": "inframe_indel",
                    "classification_method": "rule_based",
                    "acmg_evidence": {"PM4": f"In-frame deletion/insertion in {domain_name} domain"},
                    "comments": f"In-frame indel ({aa_change} AA) in {domain_name} domain"
                }, None
            else:
                # In-frame indel OUTSIDE known domains → VUS (uncertain significance).
                # May or may not be damaging — not enough evidence to classify confidently.
                return {
                    "cdna_pos": cdna_pos_approx,
                    "aa_pos": aa_pos_approx,
                    "aa_ref": "Indel",
                    "aa_alt": f"inframe({aa_change}aa)",
                    "mutation": "delins",
                    "hgvs_p": f"p.({aa_pos_approx}delins)",
                    "prediction": "VUS",
                    "probability": 0.50,
                    "risk_tier": "uncertain",
                    "risk_detail": "In-Frame Indel",
                    "genomic_pos": genomic_pos,
                    "variant_type": "inframe_indel",
                    "classification_method": "rule_based",
                    "acmg_evidence": {"PM4": "In-frame deletion/insertion outside known functional domain"},
                    "comments": f"In-frame indel ({aa_change} AA) outside known functional domains"
                }, None

    # At this point, we know it's a single nucleotide variant (SNV): both ref and alt are 1 base.
    # Validate that they are valid DNA bases.
    if ref_allele not in "ACGT" or alt_allele not in "ACGT":
        return None, "Invalid alleles"

    # ─── SPLICE-SITE CHECK (before CDS check) ─────────────────────────────────
    # If the genomic position is NOT in the coding sequence (not in genomic_to_cdna),
    # it might be an intronic variant near an exon boundary (splice site).
    # Splice sites are critical: the cell's machinery uses them to cut out introns
    # and join exons together. Mutations here can cause exon skipping or intron retention,
    # producing a broken protein.
    #   - "canonical" = +/-1-2 bases from the exon boundary (almost always breaks splicing)
    #   - "near" = +/-3-8 bases (might affect splicing, but less certain)
    if genomic_pos not in genomic_to_cdna:
        splice_info = exon_data.get("splice_info", {}).get(genomic_pos)
        if splice_info and splice_info["zone"] == "canonical":
            return {
                "cdna_pos": 0,
                "aa_pos": 0,
                "aa_ref": ref_allele,
                "aa_alt": alt_allele,
                "mutation": f"{ref_allele}>{alt_allele}",
                "hgvs_p": f"g.{genomic_pos} (canonical splice)",
                "prediction": "Likely Pathogenic",
                "probability": 0.95,
                "risk_tier": "likely_pathogenic",
                "risk_detail": "Canonical Splice Site",
                "genomic_pos": genomic_pos,
                "variant_type": "splice_canonical",
                "classification_method": "rule_based",
                "acmg_evidence": {"PVS1": f"Canonical splice site variant ({splice_info['distance']}bp from exon boundary) disrupting mRNA splicing"},
                "comments": f"Canonical splice site variant at {splice_info['distance']}bp from exon boundary"
            }, None
        elif splice_info and splice_info["zone"] == "near":
            return {
                "cdna_pos": 0,
                "aa_pos": 0,
                "aa_ref": ref_allele,
                "aa_alt": alt_allele,
                "mutation": f"{ref_allele}>{alt_allele}",
                "hgvs_p": f"g.{genomic_pos} (near splice)",
                "prediction": "VUS (Splice Proximity)",
                "probability": 0.70,
                "risk_tier": "uncertain",
                "risk_detail": "Near Splice Site",
                "genomic_pos": genomic_pos,
                "variant_type": "splice_near",
                "classification_method": "rule_based",
                "acmg_evidence": {"PP3_splice": f"Near splice site variant ({splice_info['distance']}bp from exon boundary) with potential splicing impact"},
                "comments": f"Near splice site variant at {splice_info['distance']}bp from exon boundary"
            }, None
        else:
            # Not in CDS and not near a splice site → intronic/intergenic variant, skip it
            return None, f"Not in {gene_name} CDS"

    # ─── GENOMIC → cDNA COORDINATE CONVERSION ────────────────────────────────
    # The variant IS in the coding sequence. Look up its cDNA position.
    # cDNA position = position within the gene's mRNA coding sequence (1-based).
    cdna_pos = genomic_to_cdna[genomic_pos]

    # ─── STRAND COMPLEMENTING ─────────────────────────────────────────────────
    # VCF files ALWAYS report alleles on the forward (+) strand of the chromosome.
    # But some genes are on the reverse (-) strand, meaning they're read backwards.
    # For reverse-strand genes, we must complement the bases to get the actual
    # coding-strand alleles:  A↔T, C↔G
    # Example: VCF says REF=A, ALT=G on BRCA1 (reverse strand)
    #          → coding strand has REF=T, ALT=C
    if target_strand == "-":
        cds_ref = COMPLEMENT.get(ref_allele, ref_allele)
        cds_alt = COMPLEMENT.get(alt_allele, alt_allele)
    else:
        # Forward strand: VCF alleles already match the coding strand
        cds_ref = ref_allele
        cds_alt = alt_allele

    # Build the mutation string using coding-strand bases (e.g., "A>G")
    mutation_str = f"{cds_ref}>{cds_alt}"

    # ─── CODON LOOKUP & AMINO ACID TRANSLATION ───────────────────────────────
    # Now we need to figure out which amino acid is changed.
    # The genetic code works in groups of 3 bases called "codons". Each codon
    # encodes one amino acid. So cDNA position 7 is in codon 3 (positions 7,8,9)
    # and is the 1st base of that codon.
    #
    # Steps:
    #   1. Find which codon contains this cDNA position
    #   2. Extract the 3-base reference codon from the CDS sequence
    #   3. Replace the mutated base to get the alternate codon
    #   4. Translate both codons to amino acids using the genetic code table
    cds_seq = gene_data.get("cds")
    if cds_seq is None:
        return None, "CDS not loaded"

    # cDNA is 1-based (first base = position 1), Python strings are 0-based (first char = index 0)
    codon_index = (cdna_pos - 1) // 3  # Which codon (0-based). E.g., cDNA pos 7 → codon 2
    pos_in_codon = (cdna_pos - 1) % 3  # Position within the codon (0, 1, or 2)
    codon_start = codon_index * 3       # Start index of this codon in the CDS string

    if codon_start + 3 > len(cds_seq):
        return None, "Codon out of CDS range"

    # Extract the 3-base reference codon (e.g., "ATG" for Methionine)
    ref_codon = cds_seq[codon_start:codon_start + 3]

    # Sanity check: verify that the base in our CDS at this position matches
    # what the VCF says the reference base is (after strand complementing).
    # If it doesn't match, try without complementing — some gene mappings may
    # already be on the CDS strand.
    if ref_codon[pos_in_codon] != cds_ref:
        if ref_codon[pos_in_codon] == ref_allele:
            cds_ref = ref_allele
            cds_alt = alt_allele
            mutation_str = f"{cds_ref}>{cds_alt}"
        else:
            return None, f"Ref mismatch: CDS has {ref_codon[pos_in_codon]} at cDNA {cdna_pos}, VCF says {ref_allele}"

    # Build the mutant codon by replacing the one changed base
    # Example: ref_codon = "GAT" (Asp), pos_in_codon = 2, cds_alt = "G"
    #          → alt_codon = "GAG" (Glu) → this is a missense D>E change
    alt_codon = list(ref_codon)
    alt_codon[pos_in_codon] = cds_alt
    alt_codon = "".join(alt_codon)

    # Translate both codons to amino acids using the standard genetic code.
    # "Ter" or "*" means a stop codon (translation stops here → truncated protein).
    ref_aa = CODON_TABLE.get(ref_codon.upper(), "Unknown")
    alt_aa = CODON_TABLE.get(alt_codon.upper(), "Unknown")
    aa_pos = codon_index + 1  # 1-based amino acid position in the protein

    if ref_aa == "Unknown" or alt_aa == "Unknown":
        return None, "Unknown codon"

    # ─── SYNONYMOUS VARIANT HANDLING ──────────────────────────────────────────
    # A synonymous (silent) variant changes the DNA but NOT the amino acid, because
    # multiple codons can encode the same amino acid (genetic code redundancy).
    # Example: GAT→GAC both encode Aspartate.
    # These are usually benign, BUT if they're near an exon boundary, they might
    # disrupt splicing even though the amino acid doesn't change.
    if ref_aa == alt_aa:
        # Check if this synonymous variant is near an exon/intron boundary.
        # We do this by looking at neighboring cDNA positions and checking if their
        # genomic coordinates have a "gap" — a gap means there's an intron between them,
        # which means we're at an exon boundary.
        # Example: cDNA positions 100 and 101 might map to genomic 50000 and 60000
        # (a 10,000-base intron in between), indicating an exon boundary.
        cdna_to_genomic = gene_data.get("cdna_to_genomic", {})
        near_boundary = False
        for delta in [-3, -2, -1, 1, 2, 3]:
            adj_cdna = cdna_pos + delta
            adj_gpos = cdna_to_genomic.get(adj_cdna)
            curr_gpos = cdna_to_genomic.get(cdna_pos)
            if adj_gpos is not None and curr_gpos is not None:
                # If the genomic distance between adjacent cDNA positions is larger
                # than expected (more than |delta|+1), there's an intron in between
                if target_strand == "+":
                    if abs(adj_gpos - curr_gpos) > abs(delta) + 1:
                        near_boundary = True
                        break
                else:
                    if abs(adj_gpos - curr_gpos) > abs(delta) + 1:
                        near_boundary = True
                        break

        if near_boundary:
            # Synonymous but near a splice site → VUS. The DNA change doesn't alter the
            # amino acid, but it might disrupt the splicing signal and cause exon skipping.
            return {
                "cdna_pos": cdna_pos,
                "aa_pos": aa_pos,
                "aa_ref": ref_aa,
                "aa_alt": ref_aa,
                "mutation": mutation_str,
                "hgvs_p": f"p.({ref_aa}{aa_pos}=)",
                "prediction": "VUS (Synonymous, Near Splice)",
                "probability": 0.40,
                "risk_tier": "uncertain",
                "risk_detail": "Synonymous Near Splice",
                "genomic_pos": genomic_pos,
                "variant_type": "synonymous",
                "classification_method": "rule_based",
                "acmg_evidence": {},
                "comments": "Synonymous variant near exon boundary; may affect splicing"
            }, None
        else:
            # Synonymous and far from splice site → Likely Benign.
            # ACMG evidence BP7: silent variant with no predicted splice impact.
            return {
                "cdna_pos": cdna_pos,
                "aa_pos": aa_pos,
                "aa_ref": ref_aa,
                "aa_alt": ref_aa,
                "mutation": mutation_str,
                "hgvs_p": f"p.({ref_aa}{aa_pos}=)",
                "prediction": "Likely Benign",
                "probability": 0.05,
                "risk_tier": "likely_benign",
                "risk_detail": "Synonymous",
                "genomic_pos": genomic_pos,
                "variant_type": "synonymous",
                "classification_method": "rule_based",
                "acmg_evidence": {"BP7": "Synonymous variant with no predicted splice impact"},
                "comments": "Synonymous variant far from splice site"
            }, None

    # ─── TIER 1: NONSENSE (STOP-GAIN) CHECK ─────────────────────────────────────
    # A nonsense mutation changes an amino acid codon to a STOP codon ("Ter" or "*").
    # This truncates the protein prematurely — the ribosome stops translating here.
    # In tumor suppressor genes like BRCA1/BRCA2, this is almost always pathogenic
    # because the truncated protein cannot perform its DNA repair function.
    # ACMG evidence: PVS1 (very strong pathogenic — null variant).
    if alt_aa in ["Ter", "*", "fs"]:
        return {
            "cdna_pos": cdna_pos,
            "aa_pos": aa_pos,
            "aa_ref": ref_aa,
            "aa_alt": alt_aa,
            "mutation": mutation_str,
            "hgvs_p": f"p.{ref_aa}{aa_pos}{alt_aa}",
            "prediction": "Pathogenic",
            "probability": 0.9999,
            "risk_tier": "pathogenic",
            "risk_detail": "Truncating",
            "genomic_pos": genomic_pos,
            "variant_type": "nonsense",
            "classification_method": "rule_based",
            "acmg_evidence": {"PVS1": "Pathogenic truncating null variant (Nonsense)"},
            "comments": "Tier 1 Protocol: Pathogenic truncating null variant (Nonsense)"
        }, None

    # ─── TIER 2: MACHINE LEARNING ENGINE ──────────────────────────────────────
    # If we reach here, the variant is a MISSENSE mutation: it changes one amino acid
    # to a different amino acid (e.g., Asp→Glu). Unlike frameshifts or nonsense mutations,
    # missense effects range from completely harmless to severely damaging — it depends on
    # which amino acid changed, where in the protein, evolutionary conservation, etc.
    # This is where our ML ensemble (XGBoost + MLP neural network) makes its prediction.
    #
    # The pipeline mirrors the /predict single-variant endpoint exactly:
    #   1. Build a 120-feature vector (conservation scores, structural features, etc.)
    #   2. Scale features using the trained StandardScaler
    #   3. Get predictions from both XGBoost and MLP models
    #   4. Blend them using gene-specific weights (e.g., BRCA1 uses MLP-only)
    #   5. Calibrate the blended probability using isotonic regression

    # Check that ML models are loaded (they may not be during testing or if loading failed)
    if gene_data.get("ensemble_model") is None or gene_data.get("scaler") is None:
        return None, "ML models missing for gene"

    # Step 1: Build the 120-feature vector for this specific variant.
    # Features include: phyloP conservation, BLOSUM62 substitution score, ESM-2 embeddings,
    # AlphaFold structural features, EVE scores, gnomAD allele frequencies, domain distances, etc.
    raw_vector = build_feature_vector(cdna_pos, ref_aa, alt_aa, mutation_str, aa_pos, gene_name=gene_name)
    if raw_vector.shape[1] == 0:
        return None, "Feature names not available for gene"

    # Step 2: Standardize features (zero mean, unit variance) using the training set's scaler.
    # This is critical — ML models expect the same scaling as during training.
    scaled_vector = gene_data["scaler"].transform(raw_vector)

    calibrator = gene_data.get("calibrator")       # Universal isotonic calibrator (fallback)
    nn_model = gene_data.get("ensemble_model")     # MLP (Multi-Layer Perceptron) neural network
    xgb_model = gene_data.get("booster")           # XGBoost gradient boosted trees

    if calibrator is not None and nn_model is not None and xgb_model is not None:
        # Step 3: Get raw probability predictions from both models
        nn_p = nn_model.predict(scaled_vector, verbose=0).flatten()[0]
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]

        # Step 4: Blend the two model predictions using gene-specific weights.
        # Different genes benefit from different model strengths:
        #   - BRCA1: MLP-only (XGB weight=0, MLP weight=1)
        #   - RAD51C: mostly XGBoost (80/20)
        #   - BRCA2: 60% XGB / 40% MLP
        # Falls back to default weights if gene-specific weights aren't available.
        gene_weights = _GENE_ENSEMBLE_WEIGHTS.get(gene_name.upper())
        if gene_weights:
            _xgb_w = gene_weights["xgb_weight"]
            _mlp_w = gene_weights["mlp_weight"]
        else:
            _xgb_w = XGB_WEIGHT
            _mlp_w = NN_WEIGHT
        blended = (_xgb_w * xgb_p) + (_mlp_w * nn_p)

        # Step 5: Calibrate the blended score using isotonic regression.
        # Isotonic calibration maps raw model scores to well-calibrated probabilities,
        # so that "0.8" truly means "80% chance of being pathogenic."
        # Per-gene calibrators are preferred (trained on gene-specific data);
        # the universal calibrator is the fallback.
        gene_cal = _GENE_CALIBRATORS.get(gene_name.upper())
        active_calibrator = gene_cal if gene_cal is not None else calibrator
        probability = float(active_calibrator.predict([blended])[0])

        # Track how much the two models disagree — large disagreement may indicate
        # the variant is in a region where one model is uncertain
        vcf_disagreement = round(float(abs(xgb_p - nn_p)), 4)
    else:
        # Fallback if models aren't loaded: return 0.5 (maximum uncertainty)
        probability = 0.5
        vcf_disagreement = None

    # Compute bootstrap confidence interval by resampling features with noise.
    # This gives the user a sense of how stable the prediction is.
    vcf_feature_names = gene_data.get("feature_names", [])
    vcf_bootstrap_ci = compute_bootstrap_ci(scaled_vector, vcf_feature_names)

    # Clip probability to [0.5%, 99.5%] to avoid extreme overconfidence.
    # No variant should ever be reported as exactly 0% or 100% pathogenic.
    probability = float(np.clip(probability, 0.005, 0.995))

    # Apply the gene-specific decision threshold to produce a binary label.
    # The threshold (default 0.5, but optimized per gene) is the probability cutoff
    # above which we call a variant "Pathogenic" vs "Benign".
    gene_threshold = gene_data.get("threshold", 0.5)
    label = "Pathogenic" if probability >= gene_threshold else "Benign"
    # Map the continuous probability to one of 5 ACMG risk tiers
    risk = _compute_risk_tier(probability)

    # Detect if ESM-2 protein language model features are all zeros.
    # ESM-2 embeddings are precomputed per-gene and stored in pickle files.
    # If a variant's amino acid position doesn't have a precomputed embedding,
    # the features default to zero — which may reduce prediction accuracy.
    # We warn the user so they know the prediction may be less reliable.
    vcf_warnings = []
    esm2_indices = [i for i, n in enumerate(vcf_feature_names) if n.startswith("esm2_")]
    if esm2_indices and raw_vector.shape[1] > 0:
        esm2_vals = [raw_vector[0][i] for i in esm2_indices if i < raw_vector.shape[1]]
        if all(v == 0.0 for v in esm2_vals):
            vcf_warnings.append("ESM-2 features are zero (no precomputed embedding)")

    # Build the HGVS c. notation — the standard way clinicians describe coding DNA variants.
    # Example: "NM_000059.4:c.7397A>G" means BRCA2 transcript, cDNA position 7397, A changed to G.
    transcript = GENE_TRANSCRIPTS.get(gene_name.upper(), "")
    hgvs_c = f"{transcript}:c.{cdna_pos}{mutation_str}" if transcript else f"c.{cdna_pos}{mutation_str}"

    # Assemble the final result dictionary for this missense variant
    vcf_result = {
        "cdna_pos": cdna_pos,
        "aa_pos": aa_pos,
        "aa_ref": ref_aa,
        "aa_alt": alt_aa,
        "mutation": mutation_str,
        "hgvs_p": f"p.{ref_aa}{aa_pos}{alt_aa}",
        "hgvs_c": hgvs_c,
        "prediction": label,
        "probability": round(probability, 4),
        "risk_tier": risk,
        "genomic_pos": genomic_pos,
        "variant_type": "missense",
        "classification_method": "ml_ensemble",
        "model_disagreement": vcf_disagreement,
    }
    if vcf_bootstrap_ci:
        vcf_result["bootstrap_ci"] = vcf_bootstrap_ci
    if vcf_warnings:
        vcf_result["warnings"] = vcf_warnings
    return vcf_result, None


# ─── BATCH VCF ENDPOINT ────────────────────────────────────────────────────────
# This endpoint accepts a VCF file upload and processes ALL variants in it.
# It's the main way users analyze a patient's sequencing results against a specific gene.
# The user selects which gene to analyze (default: BRCA2), and we classify every
# variant that falls within that gene's coding region.
@router.post("/predict/vcf", summary="Batch VCF prediction",
             description="Upload a VCF file to predict pathogenicity for all missense variants in the targeted gene. Returns per-variant predictions with risk tiers.")
async def predict_vcf(file: UploadFile = File(...), gene: str = Form("BRCA2")):  # noqa: C901
    """
    Batch VCF prediction endpoint — accepts a VCF file upload and classifies
    every variant that falls within the specified gene's coding region.

    This is the main way users analyze an entire patient's sequencing results.
    The flow is:
        1. Validate the gene name and load gene-specific data
        2. Read the uploaded file in chunks (with size limit to prevent abuse)
        3. Parse the VCF header and validate the file format
        4. Iterate through every data line, parse it, and classify each variant
           using vcf_variant_to_prediction()
        5. Handle multi-allelic sites (one VCF line with multiple ALT alleles)
        6. Deduplicate variants (skip if same chrom/pos/ref/alt seen before)
        7. Run compound heterozygosity detection across all results
        8. Return a JSON response with all predictions, skip reasons, and warnings

    Args:
        file (UploadFile): The uploaded VCF file. Must be UTF-8 encoded,
            tab-separated, and under MAX_VCF_SIZE bytes. Can be a standard
            single-sample or multi-sample VCF (only first sample is analyzed).
        gene (str): Which gene to analyze. Default "BRCA2". Must be one of the
            5 supported HR genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D).
            Case-insensitive.

    Returns:
        dict (JSON response) with these fields:
            - total_variants_in_file (int): Total number of data lines in the VCF
            - variants_classified (int): How many variants were successfully classified
            - gene_analyzed (str): Which gene was analyzed (uppercased)
            - predictions (list[dict]): Per-variant prediction results, each containing
              cdna_pos, aa_pos, aa_ref, aa_alt, mutation, hgvs_p, prediction,
              probability, risk_tier, variant_type, classification_method, etc.
            - skipped_count (int): How many variants were skipped
            - skipped_reasons (list[dict]): First 50 skip reasons for debugging
            - variant_type_counts (dict): Counts by type (e.g., {"missense": 3, "synonymous": 5})
            - file_warnings (list[str], optional): File-level warnings (e.g., multi-sample)
            - compound_het_warning (dict, optional): Compound heterozygosity warning if
              2+ heterozygous variants found in the same gene

        On error, returns a JSONResponse with appropriate HTTP status code:
            - 400: Invalid gene name or file format
            - 413: File too large
            - 503: Gene data not loaded
    """
    t_start = time.perf_counter()

    # Validate the requested gene is one we support
    if gene.upper() not in SUPPORTED_GENES:
        return JSONResponse(status_code=400, content={"error": f"Unsupported gene: {gene}. Supported: {', '.join(sorted(SUPPORTED_GENES))}"})
    gene = gene.upper()

    # Load all gene-specific data (CDS, exon maps, ML models, etc.)
    gene_data = get_gene_data(gene)
    if not gene_data.get("genomic_to_cdna"):
        return JSONResponse(status_code=503, content={"error": f"VCF parsing not available -- {gene} genomic mapping not loaded"})

    # Read the uploaded file in 1MB chunks with an early size check.
    # This prevents a malicious user from uploading a huge file and exhausting server memory.
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_VCF_SIZE:
            return JSONResponse(status_code=413, content={"error": f"File too large (>{MAX_VCF_SIZE // (1024*1024)} MB). Maximum is {MAX_VCF_SIZE // (1024*1024)} MB."})
        chunks.append(chunk)
    content = b"".join(chunks)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "Could not read file as UTF-8"})

    lines = text.strip().split("\n")

    # Validate VCF format
    has_vcf_header = any(line.startswith("##fileformat=VCF") for line in lines[:50])
    has_chrom_header = any(line.startswith("#CHROM") for line in lines[:100])
    if not has_vcf_header and not has_chrom_header:
        data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        if data_lines and len(data_lines[0].split("\t")) < 5:
            return JSONResponse(status_code=400, content={"error": "File does not appear to be in VCF format. Expected tab-separated columns with CHROM, POS, ID, REF, ALT."})

    # ─── MAIN VARIANT PROCESSING LOOP ────────────────────────────────────────
    # We iterate through every line in the VCF file, skipping headers (lines
    # starting with "#"), and classify each variant. Multi-allelic sites (where
    # one VCF line lists multiple ALT alleles like "T,G") are split so each
    # allele gets its own prediction.
    results = []        # Successfully classified variants (list of result dicts)
    skipped = []        # Variants that were skipped with reasons (for debugging)
    genotypes = []      # Parallel list of genotype strings for compound het detection
    seen_variants = set()  # Set of (chrom, pos, ref, alt) tuples to deduplicate
    total_data_lines = 0   # Counter for all non-header lines (even if skipped)
    multi_sample_warned = False  # Only warn about multi-sample VCF once
    file_warnings = []     # File-level warnings (not per-variant)

    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM") or line.startswith("#"):
            continue

        total_data_lines += 1
        parsed = parse_vcf_line(line)
        if parsed is None:
            skipped.append({"line": total_data_lines, "reason": "Parse error"})
            continue

        chrom, pos, ref, alt, gt, vcf_filter, line_warnings, multi_sample = parsed

        if multi_sample and not multi_sample_warned:
            file_warnings.append("Multi-sample VCF detected — using first sample only")
            multi_sample_warned = True

        # Handle multi-allelic VCF sites: a single VCF line can list multiple
        # alternate alleles separated by commas (e.g., ALT="T,G" means some reads
        # show T and others show G at this position). We process each allele
        # independently since they represent different possible mutations.
        for single_alt in alt.split(","):
            single_alt = single_alt.strip()
            # Skip star alleles and symbolic alleles
            if single_alt == "*" or single_alt.startswith("<"):
                skipped.append({"line": total_data_lines, "reason": f"Unsupported allele type: {single_alt}", "pos": pos})
                continue

            variant_key = (chrom, pos, ref, single_alt)
            if variant_key in seen_variants:
                skipped.append({"line": total_data_lines, "reason": "Duplicate variant", "pos": pos})
                continue
            seen_variants.add(variant_key)

            result, reason = vcf_variant_to_prediction(chrom, pos, ref, single_alt, gene_name=gene, gene_data=gene_data)
            if result:
                result.setdefault("variant_type", "unknown")  # defensive fallback
                # Attach per-line warnings (FILTER, GQ) to the result
                if line_warnings:
                    existing_warnings = result.get("warnings", [])
                    result["warnings"] = existing_warnings + line_warnings
                results.append(result)
                genotypes.append(gt)
            elif reason:
                skipped.append({"line": total_data_lines, "reason": reason, "pos": pos})

    # ─── COMPOUND HETEROZYGOSITY DETECTION ─────────────────────────────────────
    # Compound heterozygosity occurs when a patient has TWO different heterozygous
    # mutations in the SAME gene. Each "heterozygous" means one normal copy and one
    # mutant copy of that gene region. The critical question is: are the two mutations
    # on the SAME chromosome copy ("cis") or on DIFFERENT copies ("trans")?
    #
    #   - If TRANS (different copies): the patient has NO working copy of the gene.
    #     This is true compound heterozygosity and can cause autosomal recessive disease.
    #   - If CIS (same copy): the other chromosome copy still has a working gene.
    #     This is usually less severe (one good copy is often enough).
    #
    # We CANNOT determine cis vs trans from a standard VCF alone — that requires
    # either long-read sequencing or family trio analysis (sequencing the parents).
    # So we flag the situation and recommend phasing analysis.
    #
    # If genotype data is missing (no FORMAT/SAMPLE columns), we conservatively
    # assume all variants are heterozygous, since that's the most common scenario.
    compound_het_warning = None
    if len(results) >= 2:
        het_variants = []
        has_gt_data = False  # Track whether the VCF included genotype information
        for i, r in enumerate(results):
            gt = genotypes[i] if i < len(genotypes) else None
            if gt is not None:
                has_gt_data = True
            # A variant is heterozygous if GT is 0/1 (or phased equivalents 0|1, 1|0).
            # If no genotype data, we conservatively assume heterozygous.
            is_het = (gt is None or gt in ("0/1", "0|1", "1|0", "1/0"))
            if is_het:
                het_variants.append(r)

        if len(het_variants) >= 2:
            variant_descs = [
                f"{r.get('hgvs_p', '?')} ({r.get('variant_type', 'unknown')}, p={r.get('probability', 0):.2f})"
                for r in het_variants[:5]
            ]
            compound_het_warning = {
                "gene": gene,
                "num_variants": len(het_variants),
                "variants": variant_descs,
                "has_gt_data": has_gt_data,
                "message": (
                    f"{len(het_variants)} heterozygous variants detected in {gene}. "
                    f"Compound heterozygosity may affect pathogenicity -- "
                    f"consider phasing analysis to determine if variants are in cis (same allele) or trans (different alleles). ""True compound heterozygosity confirmation requires family trio sequencing (proband + parents). ""Statistical phasing alone is insufficient for clinical decisions."
                ),
            }

    # ─── LOGGING & METRICS ──────────────────────────────────────────────────
    # Calculate processing time and build a summary of variant types found.
    # This is logged for monitoring and debugging (e.g., "3 missense, 5 synonymous").
    latency_ms = (time.perf_counter() - t_start) * 1000
    n_classified = len(results)
    type_counts = {}  # e.g., {"missense": 3, "synonymous": 5, "frameshift": 1}
    for r in results:
        vt = r.get("variant_type", "unknown")
        type_counts[vt] = type_counts.get(vt, 0) + 1
    logger.info(f"[VCF] {file.filename}: {total_data_lines} total, {n_classified} classified {type_counts}, {len(skipped)} skipped ({latency_ms:.0f}ms)")

    # Update metrics in main module (import lazily to avoid circular dependency)
    try:
        from backend import main as _main_mod
        with _main_mod._metrics_lock:
            _main_mod._metrics["vcf_uploads"] += 1
    except Exception:
        pass

    # ─── BUILD RESPONSE ─────────────────────────────────────────────────────
    # Assemble the final JSON response with all predictions, skip info, and warnings.
    response = {
        "total_variants_in_file": total_data_lines,
        "brca2_missense_found": n_classified,  # kept identical key for front-end backwards compatibility
        "variants_classified": n_classified,
        "gene_analyzed": gene,
        "skipped_count": len(skipped),
        "predictions": results,
        "skipped_reasons": skipped[:50],  # first 50 skipped for debugging
        "variant_type_counts": type_counts,
    }
    if file_warnings:
        response["file_warnings"] = file_warnings
    if compound_het_warning:
        response["compound_het_warning"] = compound_het_warning

    return response
