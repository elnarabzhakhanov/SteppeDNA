# SteppeDNA Improvements Tracker

> **Total: 46 items** | Bugs: 6 | Missing Features: 10 | Scientific: 6 | Engineering: 9 | UI/UX: 6 | Novelty: 9
> Generated from deep codebase audit, March 2026

---

## 🔴 Bugs (Broken Functionality) — 6 items

| # | Issue | File(s) | Severity |
|---|-------|---------|----------|
| 1 | **Frameshift/Del/Ins/Dup frontend validation fails** — `VALID_AAS` in `app.js` missing 'Fs','Del','Ins','Dup','*' that backend `VALID_AA_CODES` accepts. Selecting frameshift sets aaAlt='Del' which fails frontend validation on submit. | `frontend/app.js` L191 | HIGH |
| 2 | **Multi-allelic VCF ALT not split in single-variant path** — `parse_vcf_line()` takes `parts[4]` as whole string "T,G". The `/predict/vcf` endpoint splits on comma (L1552) but `parse_vcf_line` itself doesn't. | `backend/main.py` | MEDIUM |
| 3 | **BRCA2 3D viewer broken** — AlphaFold v4 fragment URLs may no longer resolve. Non-BRCA2 uses `model_v6` which is suspicious. Need to verify/fix AlphaFold URL pattern and add version fallback. | `frontend/app.js` L1590-1593 | HIGH |
| 4 | **Gene map text overlap** — Stagger-only strategy (even/odd +/-11px) insufficient for narrow domains or small screens. No collision detection, no dynamic repositioning. | `frontend/app.js` L1446-1484 | MEDIUM |
| 5 | **Frontend AA position calculation wrong** — `Math.ceil(cdnaValStr/3)` ignores start codon offset. Backend computes correctly but frontend fallback is lossy. History/PDF may save wrong positions. | `frontend/app.js` L867 | MEDIUM |
| 6 | **SHAP legend vs bar direction mismatch** — Legend reads [Red] Pathogenic [Green] Benign (L-R), but bars show benign LEFT, pathogenic RIGHT. Legend order should be swapped to match. | `frontend/index.html` L253-256 | LOW |

---

## 🟡 Missing Data / Incomplete Features — 10 items

| # | Issue | File(s) | Severity |
|---|-------|---------|----------|
| 7 | **Frontend GENE_DOMAINS incomplete** — BRCA1 missing 3 domains (BARD1_interaction, DNA_binding, PALB2_interaction), PALB2 missing 1 (BRCA2_interaction), RAD51C missing 2 (Holliday_junction, RAD51B/D/XRCC3, NLS), RAD51D missing 1. | `frontend/app.js` L1416-1443 | HIGH |
| 8 | **Feature engineering silently zeros unknown domains** — `check_domains()` returns 0 with no log when domain names don't exist in gene config. Could mean 10-15 features silently zeroed. | `backend/feature_engineering.py` | MEDIUM |
| 9 | **No ensemble disagreement metric** — XGBoost and MLP can disagree (0.9 vs 0.1 = blended 0.58) with no signal to user about model uncertainty. | `backend/main.py` | HIGH |
| 10 | **No duplicate variant detection in batch VCF** — Same variant uploaded twice is processed twice. No deduplication. | `backend/main.py` /predict/vcf | LOW |
| 11 | **ACMG rules only partially tested** — PVS1, PM4, BP7, PP3_splice implemented but lack dedicated unit tests for all code paths. | `tests/test_acmg_rules.py` | MEDIUM |
| 12 | **MAVE features enabled in production** — `use_mave=True` by default. Potential circular reasoning if MAVE variants overlap ClinVar training labels. Need documentation or leakage guard. | `backend/feature_engineering.py` | MEDIUM |
| 13 | **No ClinVar version pinning** — Training data snapshot date/version not recorded. Cannot reproduce exact training set. | Documentation | MEDIUM |
| 14 | **EVE scores fetched but not integrated** — `data/eve_scores.pkl` exists but not used in `engineer_features()`. Deferred to novelty phase (retraining required). | `backend/feature_engineering.py` | HIGH |
| 15 | **gnomAD population-specific AFs unavailable** — Ensembl API only returns single MAF, not population-stratified. Document limitation. | `VALIDATION_REPORT.md` | LOW |
| 16 | **No `.env.example` file** — STEPPEDNA_API_KEY, ENVIRONMENT, ALLOWED_ORIGINS, WORKERS undocumented for deployers. | `.env.example` (new) | LOW |

---

## 🟠 Scientific / Methodological — 6 items

| # | Issue | File(s) | Severity |
|---|-------|---------|----------|
| 17 | **0.978 AUC misleading** — BRCA2 is 52% of test set (AUC 0.983). Overall metric hides PALB2 0.641. Need per-gene breakdown on frontend hero section. | `frontend/index.html`, `app.js` | HIGH |
| 18 | **gnomAD augmentation improvement not independently validated** — No pre/post ablation comparison in VALIDATION_REPORT showing gnomAD-specific contribution vs other changes. | `VALIDATION_REPORT.md` | MEDIUM |
| 19 | **Population bias claimed but unquantified** — "European ancestry" disclaimer exists but no ethnicity breakdown of training data. gnomAD pop AFs all 0.0 due to API limitation. | `VALIDATION_REPORT.md` | MEDIUM |
| 20 | **Isotonic calibration on extreme imbalance** — BRCA1 calibration set ~96.6% pathogenic. Isotonic regression with few benign points produces degenerate calibration. | `VALIDATION_REPORT.md` | MEDIUM |
| 21 | **Temporal validation non-BRCA2 is coin-flip** — 0.51-0.61 AUC for small genes currently buried in report. Needs prominent disclosure. | `VALIDATION_REPORT.md`, frontend | MEDIUM |
| 22 | **Non-BRCA2 predictions may use only 30-40/103 features** — Many features zeroed due to missing ESM-2/MAVE/structural data. Need feature coverage metric. | `backend/main.py`, `frontend/app.js` | HIGH |

---

## 🔵 Code Quality / Engineering — 9 items

| # | Issue | File(s) | Severity |
|---|-------|---------|----------|
| 23 | **`get_gene_data()` LRU cache not thread-safe** — `@functools.lru_cache` with async FastAPI workers creates race condition on cache population. | `backend/main.py` | MEDIUM |
| 24 | **Gene config JSON no schema validation** — Missing keys like `cds_length` cause runtime KeyError. Should validate on startup. | `backend/main.py` lifespan | MEDIUM |
| 25 | **Chromosome regex accepts invalid values** — `r'^[0-9XYMTxymt]{1,2}$'` accepts "23", "99", "MM". Need tighter validation. | `backend/main.py` | LOW |
| 26 | **cDNA max length constants off-by-one** — BRCA2 declared 10257 vs RefSeq 10254. Verify against authoritative source for all genes. | `backend/main.py` | LOW |
| 27 | **No model file checksum validation** — Pickle files loaded without integrity check. SHA256 verification on startup recommended. | `backend/main.py` | MEDIUM |
| 28 | **No model versioning system** — Retrain overwrites models. Need version metadata (training date, feature count, git commit). | `data/model_metadata.json` (new) | LOW |
| 29 | **Single Docker worker bottleneck** — Default `--workers 1`. Increase default or document scaling. | `Dockerfile`, `docker-compose.yml` | LOW |
| 30 | **Rate limiter orphan cleanup edge case** — Stale IPs only cleaned when new request arrives after threshold. Need periodic background cleanup. | `backend/main.py` | LOW |
| 31 | **No audit trail** — Triage CSV appended without timestamps, user IDs, or immutable logging. Need structured JSON prediction log. | `backend/main.py` | MEDIUM |

---

## 🟢 UI/UX Polish — 6 items

| # | Issue | File(s) | Severity |
|---|-------|---------|----------|
| 32 | **Gene map domain labels overlap** (same root cause as #4) — Need collision detection + dynamic repositioning for overlapping labels on narrow screens. | `frontend/app.js` | MEDIUM |
| 33 | **No loading/error state for ClinVar/gnomAD lookups** — Only debounce exists; no spinner or error feedback during async lookups. | `frontend/app.js` | MEDIUM |
| 34 | **localStorage quota exceeded fails silently** — `saveHistory()` doesn't notify user. Should auto-prune oldest entries and show toast. | `frontend/app.js` | LOW |
| 35 | **PDF download race condition** — Double-click not fully protected. Should disable button during generation. | `frontend/app.js` | LOW |
| 36 | **No model card file** — Mentioned in memory but no standalone `MODEL_CARD.md` in repository. | `MODEL_CARD.md` (new) | MEDIUM |
| 37 | **3D viewer auto-spin no pause button** — Rotation may be disorienting; no user control to stop/start spin. | `frontend/app.js`, `index.html` | LOW |

---

## 🟣 Novelty / Novel ML Contributions — 9 items

| # | Feature | Approach | Effort | Impact |
|---|---------|----------|--------|--------|
| 38 | **Gene-Adaptive Ensemble Weighting** | Grid-search per-gene optimal XGB/MLP weight ratios on calibration set. Save to `gene_ensemble_weights.json`. Backend uses gene-specific blend. | 3-4 hrs | HIGH |
| 39 | **Confidence Intervals (Bootstrap)** | Train 50 bootstrap XGBoost models. At prediction: run through all 50 + MLP, compute 5th/95th percentile CI. Shows equity gap numerically (BRCA2 +/-0.03 vs PALB2 +/-0.15). | 4-5 hrs | VERY HIGH |
| 40 | **Cross-Gene Feature Importance Transfer** | Compute SHAP on BRCA2 (data-rich), use as Bayesian priors for small-gene feature weighting. Compare AUC before/after. May be analysis-only. | 4-5 hrs | HIGH |
| 41 | **Data Scarcity Quantification Score** | Per-prediction count of training variants within +/-50 AA, same substitution type, same domain. Classify as HIGH/MODERATE/LOW data support. | 2-3 hrs | HIGH |
| 42 | **Active Learning Priority Ranker** | Rank VUS by query-by-committee (XGB vs MLP disagreement) x gene scarcity x positional novelty. Output top 50 per gene for wet-lab validation. | 3-4 hrs | VERY HIGH |
| 43 | **Contrastive Explanation Pairs** | Find nearest opposite-class training variant by feature distance. Show key feature differences. "This variant differs from nearest benign in: charge change, domain location." | 3-4 hrs | MEDIUM |
| 44 | **EVE Integration as Training Feature** | Add EVE score as 104th feature in `engineer_features()`. Retrain XGB+MLP. Specifically helps PALB2/RAD51C/RAD51D (BRCA2 has 0 EVE scores). | 3-4 hrs | HIGH |
| 45 | **Per-Gene Calibrators** | Train separate IsotonicRegression per gene on calibration subset. Fixes BRCA1 96.6% pathogenic calibration issue. Fallback to universal for <50 samples. | 2-3 hrs | MEDIUM |
| 46 | **Feature Coverage Metric** | Count non-zero features per prediction. Return `{nonzero: N, total: 103, pct: X%}`. Display in frontend results as transparency indicator. | 1 hr | MEDIUM |

---

## Implementation Priority Order

### Batch 1: Bug Fixes (Items 1-6)
No dependencies, immediate user-facing improvements.

### Batch 2: Missing Features + UI/UX (Items 7-16, 32-37)
Builds on bug fixes, completes the frontend.

### Batch 3: Code Quality + Engineering (Items 23-31)
Backend hardening, thread safety, validation.

### Batch 4: Scientific/Methodological (Items 17-22)
Documentation transparency, frontend metric disclosure.

### Batch 5: Novelty — Easy (Items 46, 41, 9, 45)
Feature coverage, data support score, ensemble disagreement, per-gene calibrators.

### Batch 6: Novelty — Medium (Items 38, 44, 43)
Gene-adaptive weights, EVE integration, contrastive explanations.

### Batch 7: Novelty — Hard (Items 39, 42, 40)
Bootstrap CIs, active learning ranker, cross-gene transfer.
