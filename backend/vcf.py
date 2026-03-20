"""
SteppeDNA - VCF Parsing & Batch Prediction

Handles:
- VCF line parsing
- Single VCF variant to prediction conversion
- /predict/vcf batch endpoint
"""

import re
import time
import logging

import numpy as np
import xgboost as xgb
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form

from backend.models import (
    SUPPORTED_GENES, MAX_VCF_SIZE,
    XGB_WEIGHT, NN_WEIGHT,
    _GENE_ENSEMBLE_WEIGHTS, _GENE_CALIBRATORS,
)
from backend.features import get_gene_data, build_feature_vector
from backend.explanations import compute_bootstrap_ci
from backend.constants import CODON_TABLE, COMPLEMENT
# Database storage disabled — privacy: no variant data is stored server-side
# from backend.database import record_vcf_upload

logger = logging.getLogger("steppedna")

# RefSeq transcript IDs for HGVS c. notation
GENE_TRANSCRIPTS = {
    "BRCA1": "NM_007294.4",
    "BRCA2": "NM_000059.4",
    "PALB2": "NM_024675.4",
    "RAD51C": "NM_058216.3",
    "RAD51D": "NM_002878.4",
}

router = APIRouter(tags=["VCF"])

# _metrics_lock is needed for updating _metrics in main.py.
# We import these from main at call time to avoid circular imports.


def _compute_risk_tier(probability: float) -> str:
    """Unified 5-tier risk logic used by both /predict and /predict/vcf.
    Aligned with ACMG 5-tier classification system."""
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
    """Parse a single VCF data line into a dict with chrom, pos, ref, alt, genotype, filter, gq, and warnings.

    Design: Returns the raw ALT field which may contain comma-separated
    multi-allelic values (e.g. "T,G"). Splitting into individual alleles
    is handled by the /predict/vcf endpoint caller, so each allele gets
    its own prediction while sharing the same genotype context.

    Genotype is extracted from FORMAT+SAMPLE columns if present (cols 8+9),
    otherwise None. Used for compound heterozygosity detection.
    """
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    chrom = parts[0].lower().replace("chr", "")
    # Validate chromosome format
    if not re.match(r'^([1-9]|1[0-9]|2[0-2]|X|Y|MT?)$', chrom, re.IGNORECASE):
        return None
    try:
        pos = int(parts[1])
    except ValueError:
        return None
    ref = parts[3].upper()
    alt = parts[4].upper()

    # Extract FILTER column (col 6, 0-indexed)
    vcf_filter = parts[6] if len(parts) > 6 else "."
    vcf_warnings = []
    if vcf_filter not in {"PASS", ".", ""}:
        vcf_warnings.append(f"VCF FILTER: {vcf_filter} — variant did not pass upstream QC")

    # Parse genotype and GQ from FORMAT + SAMPLE columns if present
    genotype = None
    gq = None
    multi_sample = False
    if len(parts) >= 10:
        if len(parts) > 10:
            multi_sample = True
        format_field = parts[8]
        sample_field = parts[9]
        format_keys = format_field.split(":")
        sample_vals = sample_field.split(":")
        if "GT" in format_keys:
            gt_idx = format_keys.index("GT")
            if gt_idx < len(sample_vals):
                genotype = sample_vals[gt_idx]
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
    Convert a VCF variant to a prediction.

    Returns:
        tuple: (result_dict, error_message)
            - On success: (dict with prediction/probability/variant_type/..., None)
            - On skip: (None, str reason why variant was skipped)

    Handles 6 variant types:
      - Frameshift (Tier 1 rule): p=0.9999, PVS1
      - In-frame indel (Tier 1 rule): domain-aware PM4
      - Splice-site (Tier 1 rule): canonical +/-1-2bp / near +/-3-8bp
      - Nonsense (Tier 1 rule): p=0.9999, PVS1
      - Synonymous (Tier 1 rule): splice-proximity aware, BP7
      - Missense (Tier 2 ML): ensemble prediction
    """
    if gene_data is None:
        gene_data = get_gene_data(gene_name)

    gene_config = gene_data.get("gene_config", {})
    if not gene_config:
        return None, f"Gene configuration not found for {gene_name}"
    target_chrom = str(gene_config.get("chromosome", "13"))
    target_strand = gene_config.get("strand", "+")

    if chrom != target_chrom:
        return None, f"Not on chr{target_chrom}"

    genomic_to_cdna = gene_data.get("genomic_to_cdna", {})
    if not genomic_to_cdna:
        return None, "Genomic mapping not loaded"

    exon_data = gene_data.get("exon_data", {})

    # ─── INDEL CHECK ───────────────────────────────────────────────────────────
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        len_diff = abs(len(ref_allele) - len(alt_allele))
        if len_diff % 3 != 0:
            # Frameshift -- highly pathogenic
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
            cdna_pos_approx = genomic_to_cdna.get(genomic_pos, 0)
            aa_pos_approx = (cdna_pos_approx - 1) // 3 + 1 if cdna_pos_approx > 0 else 0
            aa_change = len_diff // 3

            in_domain = False
            domain_name = None
            if aa_pos_approx > 0:
                for dname, drange in gene_config.get("domains", {}).items():
                    if drange[0] <= aa_pos_approx <= drange[1]:
                        in_domain = True
                        domain_name = dname
                        break

            if in_domain:
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

    if ref_allele not in "ACGT" or alt_allele not in "ACGT":
        return None, "Invalid alleles"

    # ─── SPLICE-SITE CHECK (before CDS check) ─────────────────────────────────
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
            return None, f"Not in {gene_name} CDS"

    cdna_pos = genomic_to_cdna[genomic_pos]

    if target_strand == "-":
        cds_ref = COMPLEMENT.get(ref_allele, ref_allele)
        cds_alt = COMPLEMENT.get(alt_allele, alt_allele)
    else:
        cds_ref = ref_allele
        cds_alt = alt_allele

    # Build the mutation string (CDS-strand)
    mutation_str = f"{cds_ref}>{cds_alt}"

    # Determine amino acid change using the CDS
    cds_seq = gene_data.get("cds")
    if cds_seq is None:
        return None, "CDS not loaded"

    # cDNA is 1-based, Python string is 0-based
    codon_index = (cdna_pos - 1) // 3  # 0-based codon number
    pos_in_codon = (cdna_pos - 1) % 3  # 0, 1, or 2
    codon_start = codon_index * 3

    if codon_start + 3 > len(cds_seq):
        return None, "Codon out of CDS range"

    ref_codon = cds_seq[codon_start:codon_start + 3]

    # Verify the CDS matches what the VCF says (after complement)
    if ref_codon[pos_in_codon] != cds_ref:
        # Try without complementing (in case mapping is already on CDS strand)
        if ref_codon[pos_in_codon] == ref_allele:
            cds_ref = ref_allele
            cds_alt = alt_allele
            mutation_str = f"{cds_ref}>{cds_alt}"
        else:
            return None, f"Ref mismatch: CDS has {ref_codon[pos_in_codon]} at cDNA {cdna_pos}, VCF says {ref_allele}"

    # Build the mutant codon
    alt_codon = list(ref_codon)
    alt_codon[pos_in_codon] = cds_alt
    alt_codon = "".join(alt_codon)

    ref_aa = CODON_TABLE.get(ref_codon.upper(), "Unknown")
    alt_aa = CODON_TABLE.get(alt_codon.upper(), "Unknown")
    aa_pos = codon_index + 1  # 1-based

    if ref_aa == "Unknown" or alt_aa == "Unknown":
        return None, "Unknown codon"

    # ─── SYNONYMOUS VARIANT HANDLING ──────────────────────────────────────────
    if ref_aa == alt_aa:
        # Check if near an exon boundary (splice-proximity)
        cdna_to_genomic = gene_data.get("cdna_to_genomic", {})
        near_boundary = False
        for delta in [-3, -2, -1, 1, 2, 3]:
            adj_cdna = cdna_pos + delta
            adj_gpos = cdna_to_genomic.get(adj_cdna)
            curr_gpos = cdna_to_genomic.get(cdna_pos)
            if adj_gpos is not None and curr_gpos is not None:
                if target_strand == "+":
                    if abs(adj_gpos - curr_gpos) > abs(delta) + 1:
                        near_boundary = True
                        break
                else:
                    if abs(adj_gpos - curr_gpos) > abs(delta) + 1:
                        near_boundary = True
                        break

        if near_boundary:
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

    # ─── TIER 1: RULE INTERCEPTOR ───────────────────────────────────────────────
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

    # Build feature vector and predict
    if gene_data.get("ensemble_model") is None or gene_data.get("scaler") is None:
        return None, "ML models missing for gene"

    raw_vector = build_feature_vector(cdna_pos, ref_aa, alt_aa, mutation_str, aa_pos, gene_name=gene_name)
    if raw_vector.shape[1] == 0:
        return None, "Feature names not available for gene"

    scaled_vector = gene_data["scaler"].transform(raw_vector)

    calibrator = gene_data.get("calibrator")
    nn_model = gene_data.get("ensemble_model")
    xgb_model = gene_data.get("booster")

    if calibrator is not None and nn_model is not None and xgb_model is not None:
        nn_p = nn_model.predict(scaled_vector, verbose=0).flatten()[0]
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]
        # Gene-adaptive ensemble weights (matching /predict endpoint)
        gene_weights = _GENE_ENSEMBLE_WEIGHTS.get(gene_name.upper())
        if gene_weights:
            _xgb_w = gene_weights["xgb_weight"]
            _mlp_w = gene_weights["mlp_weight"]
        else:
            _xgb_w = XGB_WEIGHT
            _mlp_w = NN_WEIGHT
        blended = (_xgb_w * xgb_p) + (_mlp_w * nn_p)
        # Per-gene calibrator (matching /predict endpoint)
        gene_cal = _GENE_CALIBRATORS.get(gene_name.upper())
        active_calibrator = gene_cal if gene_cal is not None else calibrator
        probability = float(active_calibrator.predict([blended])[0])
        vcf_disagreement = round(float(abs(xgb_p - nn_p)), 4)
    else:
        probability = 0.5
        vcf_disagreement = None

    # Bootstrap CI for VCF variants (Item 39)
    vcf_feature_names = gene_data.get("feature_names", [])
    vcf_bootstrap_ci = compute_bootstrap_ci(scaled_vector, vcf_feature_names)

    # Clip to [0.5%, 99.5%]
    probability = float(np.clip(probability, 0.005, 0.995))

    gene_threshold = gene_data.get("threshold", 0.5)
    label = "Pathogenic" if probability >= gene_threshold else "Benign"
    risk = _compute_risk_tier(probability)

    # Detect ESM-2 zero-fill in VCF path
    vcf_warnings = []
    esm2_indices = [i for i, n in enumerate(vcf_feature_names) if n.startswith("esm2_")]
    if esm2_indices and raw_vector.shape[1] > 0:
        esm2_vals = [raw_vector[0][i] for i in esm2_indices if i < raw_vector.shape[1]]
        if all(v == 0.0 for v in esm2_vals):
            vcf_warnings.append("ESM-2 features are zero (no precomputed embedding)")

    transcript = GENE_TRANSCRIPTS.get(gene_name.upper(), "")
    hgvs_c = f"{transcript}:c.{cdna_pos}{mutation_str}" if transcript else f"c.{cdna_pos}{mutation_str}"

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


@router.post("/predict/vcf", summary="Batch VCF prediction",
             description="Upload a VCF file to predict pathogenicity for all missense variants in the targeted gene. Returns per-variant predictions with risk tiers.")
async def predict_vcf(file: UploadFile = File(...), gene: str = Form("BRCA2")):  # noqa: C901
    t_start = time.perf_counter()
    if gene.upper() not in SUPPORTED_GENES:
        return JSONResponse(status_code=400, content={"error": f"Unsupported gene: {gene}. Supported: {', '.join(sorted(SUPPORTED_GENES))}"})
    gene = gene.upper()
    gene_data = get_gene_data(gene)
    if not gene_data.get("genomic_to_cdna"):
        return JSONResponse(status_code=503, content={"error": f"VCF parsing not available -- {gene} genomic mapping not loaded"})

    # Read in chunks with early abort to prevent memory exhaustion
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

    results = []
    skipped = []
    genotypes = []  # track genotypes for compound het detection
    seen_variants = set()
    total_data_lines = 0
    multi_sample_warned = False
    file_warnings = []

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

        # Handle multi-allelic VCF
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
    compound_het_warning = None
    if len(results) >= 2:
        het_variants = []
        has_gt_data = False
        for i, r in enumerate(results):
            gt = genotypes[i] if i < len(genotypes) else None
            if gt is not None:
                has_gt_data = True
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

    latency_ms = (time.perf_counter() - t_start) * 1000
    n_classified = len(results)
    type_counts = {}
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
