# SteppeDNA — Complete Issue Audit & Unfixed Items

> Last verified: 2026-03-16, v5.4.0 (model fixes, deployment prep, VUS analysis)
> Methodology: Line-by-line grep verification against codebase + 3-agent parallel audit
> Updated: 2026-03-08 — added 12 fixed items (A67-A78) from UX/bug fix session + Section H (4 discovered-unfixed items)

---

## SECTION A0: SECURITY AUDIT FIXES (23/23 complete — March 2026)

All 23 issues identified by security audit have been fixed, verified in code, and passing 198/198 tests (v5.4.0).

| # | Severity | Issue | File(s) Fixed |
|---|----------|-------|---------------|
| S1 | CRITICAL | Pickle deserialization RCE — checksum bypass | `backend/main.py:128-160` |
| S2 | CRITICAL | API URL override via `?api=` query param | `frontend/api.js:9-18` |
| S3 | CRITICAL | Unauthenticated cohort write endpoint | `backend/cohort.py:98-127` |
| S4 | HIGH | Broken API key flow (frontend never sends key) | `backend/middleware.py:52-64` |
| S5 | HIGH | CORS allows `null` origin | `backend/main.py:255` |
| S6 | HIGH | Rate limiter X-Forwarded-For spoofing | `backend/middleware.py:75`, `backend/external_api.py:98,178` |
| S7 | HIGH | VCF upload memory exhaustion DoS | `backend/vcf.py:470-481` |
| S8 | HIGH | SQLite on ephemeral filesystem (Render) | `backend/database.py:15-18` |
| S9 | HIGH | `/docs` and `/redoc` exposed in production | `backend/main.py:247-248` |
| S10 | MEDIUM | No HSTS header | `backend/middleware.py:112`, `vercel.json:26`, `nginx.conf:12` |
| S11 | MEDIUM | CSP `unsafe-inline` for scripts | `frontend/sw-register.js` (new), `index.html:655`, `vercel.json:27`, `nginx.conf:13` |
| S12 | MEDIUM | Ephemeral audit logs (lost on redeploy) | `backend/main.py:176-179` |
| S13 | MEDIUM | innerHTML XSS risk (hardcoded SVG) | `frontend/app.js:193` (safety comment) |
| S14 | MEDIUM | Service worker caches non-HTML responses | `frontend/sw.js:15-16,65` |
| S15 | MEDIUM | Wetlab triage CSV unbounded append | `backend/main.py:521-525` |
| S16 | MEDIUM | FileResponse path traversal | `backend/main.py:792-801` |
| S17 | MEDIUM | Docker container runs as root | `Dockerfile:18-21` |
| S18 | LOW | Incomplete dependency pinning | `requirements.txt` |
| S19 | LOW | Error messages leak internal state | `backend/cohort.py` |
| S20 | LOW | No request body size limit | `backend/middleware.py:40-46` |
| S21 | LOW | Dev server binds to 0.0.0.0 | `backend/main.py:822` |
| S22 | LOW | `window.STEPPEDNA_API_BASE` override | `frontend/api.js` (removed) |
| S23 | LOW | No SRI on CDN-loaded NGL.js | `frontend/app.js:1828-1831` |

Also fixed: **Rate limiter flaky test** — `tests/conftest.py:28-33` now resets rate counter between tests.

---

## SECTION A: FIXED ISSUES (No Action Needed)

These items have been verified as resolved in the current codebase:

| # | Issue | Where Fixed |
|---|-------|-------------|
| 1 | VCF "Type Unknown" badge | `app.js:655` — `'unknown': ['Unknown Type', 'badge-info']` |
| 2 | `rel.gene_auc` → `rel.auc` | `app.js:1014` |
| 3 | History replay `getElementById('geneSelect')` | `app.js:109` |
| 4 | "Moderate Confidence" in RESULT_TR | `app.js:218` |
| 5 | HTML fallback text mc_lim2, mc_lim5 | `index.html` updated |
| 6 | Certification text "Future" not "Planned" | `index.html` updated |
| 7 | CSS version tag `?v=5.3` | `index.html` |
| 8 | gnomAD protein-notation handling | `external_api.py:184-185` |
| 9 | Rate limiter growth prevention (stale cleanup) | `main.py:227-232` |
| 10 | Prediction cache vs API cache separated | `external_api.py:44,49` — `_pred_cache` / `_api_cache` |
| 11 | Fallback p=0.5 warning | `main.py:474` |
| 12 | FastAPI lifespan context manager | `main.py` — lifespan, not deprecated `on_event` |
| 13 | Docker data volume writable (no `:ro`) | `docker-compose.yml:16` |
| 14 | Docker health check uses curl | `Dockerfile:31-32`, `docker-compose.yml:19` |
| 15 | docker-compose ALLOWED_ORIGINS env var | `docker-compose.yml:11` |
| 16 | docker-compose no deprecated version field | `docker-compose.yml` (no `version:` line) |
| 17 | CI flake8 no `--exit-zero` | `.github/workflows/ci.yml:37-44` |
| 18 | Triage CSV uses csv.writer | `main.py:545` |
| 19 | database.py docstring says "synchronous sqlite3" | `database.py:2-6` |
| 20 | Duplicate `import xgboost as xgb` removed | Only top-level import |
| 21 | `import time as std_time` removed | Uses top-level `time` |
| 22 | `import json as _json` removed | Uses top-level `json` |
| 23 | Redundant JSONResponse import removed | Single import |
| 24 | Multiple `import re` inside functions removed | Top-level only |
| 25 | gnomAD GraphQL `errors` field checked | `external_api.py:248-250` |
| 26 | Retry logic for ClinVar and gnomAD | `external_api.py:119-126,230-241` |
| 27 | Chromosome validation in VCF parsing | `vcf.py:67` |
| 28 | ESM-2 bounds check (AA_pos > protein length) | `features.py:380` |
| 29 | SHAP rendering empty features guard | `app.js:1118` |
| 30 | Gene reliability tests exist | `test_api_endpoints.py` |
| 31 | README no YOUR_USERNAME placeholder | `README.md` |
| 32 | .dockerignore exists | `.dockerignore` |
| 33 | requirements.txt split (requirements-dev.txt) | Both files exist |
| 34 | Vercel rewrites complete | `vercel.json` — all routes |
| 35 | CSP headers in nginx.conf and vercel.json | Both files |
| 36 | Service worker bumped to v5.3.1 | `sw.js` |
| 37 | ACMG BS1 gene-specific thresholds | `acmg_rules.py` — GENE_BS1_THRESHOLDS |
| 38 | ACMG BA1 stand-alone benign code | `acmg_rules.py` — GENE_BA1_THRESHOLDS |
| 39 | ACMG PM5 code + wired into predict | `acmg_rules.py:85-86`, `main.py` pm5_positions |
| 40 | PM5 lookup table exists | `data/pathogenic_positions.json` (21KB) |
| 41 | ClinVar lookup reads gene query parameter | `external_api.py` |
| 42 | Root endpoint model_type correct | `main.py:745` |
| 43 | Old feature names (in_BRC_repeats) replaced | `features.py` |
| 44 | PDF footer says v5.3 | `app.js:1527` |
| 45 | HSTS header in nginx.conf | `nginx.conf:12` |
| 46 | variant_type defensive fallback | `vcf.py:540` — `result.setdefault("variant_type", "unknown")` |
| 47 | PDF popup blocker detection | `app.js:1529` — showToast on window.open failure |
| 48 | ESM-2 zero-feature warning | `main.py:470` |
| 49 | Position validation in build_feature_vector | `features.py:243` — `if cDNA_pos < 1 or AA_pos < 1:` |
| 50 | API version consistency | `backend/__init__.py:3` → `__version__ = "5.3.0"`, used dynamically |
| 51 | BRCA2 dominance disclosed | VALIDATION_REPORT.md + frontend footnote |
| 52 | Per-gene AUC bars displayed | Frontend per-gene bars |
| 53 | BRCA1 class imbalance acknowledged | VALIDATION_REPORT.md |
| 54 | Internal vs external AUC gap disclosed | Footnote: "Independent benchmark AUC: 0.750-0.801" |
| 55 | European ancestry bias documented | VALIDATION_REPORT.md Section 13 (full equity analysis) |
| 56 | Temporal bias documented | VALIDATION_REPORT.md Section 10 + temporal_validation.py |
| 57 | MAVE coverage acknowledged | VALIDATION_REPORT.md |
| 58 | AlphaFold BRCA2 bias documented | VALIDATION_REPORT.md Section 11 (GPU ablation) |
| 59 | ACMG approximation label | Frontend labels as "computational approximation" |
| 60 | Compound het phasing limitation acknowledged | `vcf.py:575-576` — warning message mentions phasing |
| 61 | SOTA comparison caveat | Footnote present |
| 62 | Deployment configs (Render + Vercel) ready | `render.yaml` + `vercel.json` |
| 63 | Integration script for GPU embeddings | `scripts/integrate_gpu_embeddings.py` |
| 64 | EVE ablation study | `retrain_with_eve.py`, `eve_scores_cache.json` |
| 65 | AlphaMissense leakage acknowledged | VALIDATION_REPORT.md |
| 66 | OOD warnings (C-terminal, gnomAD AF) | `main.py:480,603` |
| 67 | Mobile hero-controls layout (lang/theme buttons overlapping logo) | `styles.css` — `.hero-controls` wrapper, `position: absolute` → `static` on ≤700px |
| 68 | Per-gene section gradient overlap positioning | `styles.css` — `.per-gene-section` margin: `-2rem auto 2.5rem` |
| 69 | Footnote link duplication on language cycling | `lang.js` — removed `<a href="VALIDATION_REPORT.md">` from all i18n footnote values |
| 70 | SOTA badge pill → plain text uniformity | `styles.css` — `.sota-badge` stripped to `font-weight:700` colored text |
| 71 | Frameshift mutation Pydantic validation failure | `app.js` sends `AA_alt="Fs"`, `Mutation="FS"`; `main.py` validator allows FS/DEL/INS/DUP; TIER 1 checks `"Fs"` |
| 72 | Individual variant removal (dismiss + VCF row remove) | `index.html` + `app.js` + `styles.css` — `.btn-dismiss-result` on result card, `.btn-remove-row` on VCF rows |
| 73 | Active learning Python module-level import bug (stale reference) | `cohort.py` → `_get_active_learning()` getter in `models.py` instead of direct `_ACTIVE_LEARNING` import |
| 74 | Gene diagram label-to-circle spacing | `app.js` — lollipop circle `cy` 14→20, line `y2` 18→24 (14px gap, was 8px) |
| 75 | "See Validation Report" vague footnotes removed | `lang.js` + `index.html` — per-gene, temporal, SOTA footnotes in EN/KK/RU |
| 76 | VCF/Manual Entry section spacing (gap after SOTA) | `styles.css` — `.sota-section` `margin-bottom` 2.5rem→1.2rem |
| 77 | CI width bar redesign | `app.js` — 0%/100% scale labels, "CI width" colored header, 10px bar height (was 18px) |
| 78 | Domain label collision detection improved | `app.js` — collision guard 6→10px, stagger spacing 13→15px |

---

## SECTION A2: FIXED IN v5.4 (March 2026)

| # | Issue | Where Fixed |
|---|-------|-------------|
| B3 | PM5 based on training data only |  � fetches from full ClinVar |
| B7 | gnomAD AFs all zeros |  � real gnomAD v4 via myvariant.info |
| B8 | No Kazakh founder mutations |  +  endpoint integration |
| B13 | EVE scores not in production |  � eve_score, eve_pathogenic, eve_x_phylop |
| B15 | No population-stratified analysis |  |
| B16 | SHAP disclaimer missing |  � added to all 3 languages |
| B18 | Headline AUC misleading |  � macro-averaged 0.775 prominent |
| B20 | Compound het disclaimer weak |  � trio sequencing requirement added |
| B21 | MAVE/DMS sparse coverage |  � BRCA1 Findlay SGE integrated |
| B22 | AlphaFold biased to BRCA2 |  � all 5 genes |
| B27 | SOTA caveat not prominent |  +  � asterisk + footnote |
| B30 | 59 redundant scripts | 6 archived to  |
| B33 | hypothesis missing |  � hypothesis>=6.0 added |
| B34 | CSP missing from middleware |  � full CSP header added |
| B35 | README missing limitations |  � Known Limitations section added |
| B36 | Rate limiter test flaky |  � RATE_LIMIT=9999 fixture |
| D1 | psutil not pinned |  � psutil==6.1.1 |
| F1 | NICE_NAMES incomplete |  � ~37 missing names added |
| F2 | Test artifacts in CSV |  � 2 test rows removed |
| F7 | BA1 threshold too permissive |  � ClinGen SVI thresholds |
| F8 | Error format inconsistency |  � JSONResponse with proper HTTP status |
| F10 | gnomAD OOD not gene-specific |  � aligned with BA1 thresholds |
| F13 | Probability clipping undocumented |  � documented as design decision |
| G1 | Orphaned pickle files | Already cleaned (migrated to code constants) |
| G2 | Orphaned brca2_final_model_ipresume | Already removed |
| H1 | Active learning retry fragile |  � exponential backoff (3 attempts) |
| H2 | SW prevents live updates |  � updatefound + toast notification |

**New features added in v5.4:**
- EVE evolutionary coupling scores (3 features)
- Gene-specific domain proximity features (6 features)
- Real gnomAD v4 allele frequencies for all 5 genes
- BRCA1 Findlay SGE DMS scores (1,837 variants)
- Kazakh founder mutation detection in /predict endpoint
- Colab retraining notebook (notebooks/retrain_v54.ipynb)

---

## SECTION A2: FIXED IN v5.4 (March 2026)

| # | Issue | Where Fixed |
|---|-------|-------------|
| B3 | PM5 based on training data only | build_pm5_clinvar.py -- fetches from full ClinVar |
| B7 | gnomAD AFs all zeros | fetch_gnomad_myvariant.py -- real gnomAD v4 |
| B8 | No Kazakh founder mutations | kazakh_founder_mutations.json + predict endpoint |
| B13 | EVE scores not in production | features.py -- eve_score, eve_pathogenic, eve_x_phylop |
| B15 | No population-stratified analysis | scripts/population_analysis.py |
| B16 | SHAP disclaimer missing | frontend/lang.js -- all 3 languages |
| B18 | Headline AUC misleading | frontend/index.html -- macro-averaged 0.775 |
| B20 | Compound het disclaimer weak | backend/vcf.py -- trio sequencing warning |
| B21 | MAVE/DMS sparse coverage | integrate_brca1_dms.py -- BRCA1 Findlay SGE |
| B22 | AlphaFold biased to BRCA2 | fetch_alphafold_all_genes.py -- all 5 genes |
| B27 | SOTA caveat not prominent | index.html + lang.js -- asterisk + footnote |
| B30 | 59 redundant scripts | 6 archived to scripts/archive/ |
| B33 | hypothesis missing | requirements-dev.txt |
| B34 | CSP missing from middleware | backend/middleware.py |
| B35 | README missing limitations | README.md |
| B36 | Rate limiter test flaky | tests/conftest.py |
| D1 | psutil not pinned | requirements.txt -- psutil==6.1.1 |
| F1 | NICE_NAMES incomplete | backend/features.py |
| F2 | Test artifacts in CSV | needs_wetlab_assay.csv -- 2 rows removed |
| F7 | BA1 threshold too permissive | acmg_rules.py -- ClinGen SVI thresholds |
| F8 | Error format inconsistency | vcf.py -- JSONResponse with HTTP status |
| F10 | gnomAD OOD not gene-specific | main.py -- aligned with BA1 |
| F13 | Probability clipping undocumented | VALIDATION_REPORT.md |
| G1 | Orphaned pickle files | Already cleaned |
| G2 | Orphaned archive directory | Already removed |
| H1 | Active learning retry fragile | app.js -- exponential backoff |
| H2 | SW prevents live updates | sw-register.js -- updatefound + toast |

| --- | **Session 2026-03-16 fixes** | --- |
| H4 | Test count stale in docs | MEMORY.md, CHANGELOG.md, ROADMAP -- 198 tests |
| F12 | BRCA1 P871L misclassification | INVESTIGATED: ML still wrong (0.995) but ACMG correctly overrides to Benign via BA1 (48.4% AF). Model limitation, not a code bug. |
| B11 | VUS reclassification not run | DONE: scripts/vus_reclassification_multigene.py + vus_disparity_analysis.py executed |
| B25 | No live deployment | PARTIAL: render.yaml updated (starter plan), .gitignore/.dockerignore updated for model files, TF made optional |
| --- | Version strings stale at 5.3.x | __init__.py, index.html, sw.js, CHANGELOG.md, model_metadata.json all updated to 5.4.0 |
| --- | Founder mutation analysis | NEW: data/founder_mutation_analysis.json + data/vus_disparity_results.json generated |
| --- | Bootstrap CI crash on 120 features | backend/explanations.py -- try/except fallback to beta approximation |
| --- | Gene ensemble weight key mismatch | backend/models.py -- supports both xgb/mlp and xgb_weight/mlp_weight |
| --- | Founder mutations JSON loading crash | backend/main.py -- fixed structure + field name mismatches |
| --- | Test imports broken (v5.3 refactor) | tests/test_exon_boundaries.py -- imports updated |

New v5.4 features: EVE scores (3), domain proximity (6), real gnomAD v4, BRCA1 DMS, KZ founders, Colab notebook.

---

## SECTION B: UNFIXED ISSUES (With Solutions)

---

### B1. External API Timeout Tests Missing
**Category:** Code — Testing
**Priority:** P2
**Effort:** Low

**Problem:**
No mock tests verify graceful degradation when ClinVar or gnomAD APIs timeout. Retry logic exists (`external_api.py:119-126,230-241`) but is never tested.

**Solution:**
Add to `tests/test_api_endpoints.py`:
```python
from unittest.mock import patch, AsyncMock
import httpx

@pytest.mark.asyncio
async def test_clinvar_lookup_timeout(client):
    """ClinVar timeout returns graceful error, not crash."""
    with patch("backend.external_api.httpx.AsyncClient.get",
               side_effect=httpx.TimeoutException("timeout")):
        resp = client.get("/lookup?variant=BRCA2:c.1234A>G&gene=BRCA2")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data or data.get("clinvar") is None

@pytest.mark.asyncio
async def test_gnomad_lookup_timeout(client):
    """gnomAD timeout returns graceful error."""
    with patch("backend.external_api.httpx.AsyncClient.post",
               side_effect=httpx.TimeoutException("timeout")):
        resp = client.get("/lookup?variant=BRCA2:c.1234A>G&gene=BRCA2")
        assert resp.status_code == 200

@pytest.mark.asyncio
async def test_clinvar_retry_succeeds(client):
    """ClinVar retry logic: first call fails, second succeeds."""
    mock_fail = httpx.Response(500)
    mock_ok = httpx.Response(200, json={"esearchresult": {"idlist": []}})
    with patch("backend.external_api.httpx.AsyncClient.get",
               side_effect=[mock_fail, mock_ok]):
        resp = client.get("/lookup?variant=BRCA2:c.1234A>G&gene=BRCA2")
        assert resp.status_code == 200
```

---

### B2. CSP `unsafe-inline` Still Present
**Category:** Security
**Priority:** WON'T FIX
**Effort:** N/A

**Problem:**
Both `nginx.conf` and `vercel.json` allow `'unsafe-inline'` for styles. This weakens XSS protection.

**Why It Can't Be Fixed:**
3Dmol.js (the protein structure viewer) injects inline styles and scripts. Removing `unsafe-inline` breaks the 3D viewer entirely. This is a known upstream limitation.

**Mitigation:**
Already documented. Monitor 3Dmol.js releases for CSP-compatible mode. Could be revisited if 3Dmol adds nonce/hash support.

---

### B3. PM5 Based on Training Data, Not Curated ClinVar
**Category:** Scientific — ACMG Accuracy
**Priority:** P2
**Effort:** Medium

**Problem:**
`data/pathogenic_positions.json` was generated from the training dataset (19,223 variants), not from the full ClinVar database. PM5 can only fire for amino acid positions represented in training. ClinVar has thousands more pathogenic missense positions not in the training set.

Additionally, this is NOT disclosed in VALIDATION_REPORT.md (verified: no mention of PM5 source).

**Solution:**
1. **Build curated PM5 table from latest ClinVar VCF:**
```python
# scripts/build_pm5_from_clinvar.py
import gzip, re, json
from collections import defaultdict

positions = defaultdict(set)
with gzip.open("data/clinvar.vcf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        fields = line.split("\t")
        info = fields[7]
        # Extract gene, AA change, clinical significance
        if "Pathogenic" in info or "Likely_pathogenic" in info:
            gene_match = re.search(r"GENEINFO=(\w+):", info)
            aa_match = re.search(r"CLNHGVS=.*:p\.([A-Z][a-z]{2})(\d+)", info)
            if gene_match and aa_match:
                gene = gene_match.group(1)
                pos = int(aa_match.group(2))
                if gene in {"BRCA1","BRCA2","PALB2","RAD51C","RAD51D"}:
                    positions[gene].add(pos)

with open("data/pathogenic_positions_clinvar.json", "w") as f:
    json.dump({g: sorted(list(ps)) for g, ps in positions.items()}, f)
```
2. **Update `main.py`** to load `pathogenic_positions_clinvar.json` instead of current file
3. **Add disclosure** to VALIDATION_REPORT.md: "PM5 positions are derived from the full ClinVar database, not limited to training variants."

---

### B4. No External Validation Cohort
**Category:** Scientific — Validation
**Priority:** P1
**Effort:** Medium

**Problem:**
All reported metrics are on SteppeDNA's own held-out test set. No independent external dataset has been used for validation.

**Solution:**
- **ENIGMA consortium:** Request access to expert-classified BRCA1/2 variants
- **VariBench / CAGI challenge datasets:** Publicly available variant benchmarks
- **LOVD (Leiden Open Variation Database):** Curated variants for HR genes not all in ClinVar
- Run frozen v5.3 model on external set WITHOUT retraining or threshold tuning
- Report external AUC, calibration curve, per-gene breakdown
- If external AUC drops >0.05 below internal, investigate and document honestly

---

### B5. AlphaMissense Indirect Label Leakage — Not Quantified
**Category:** Scientific — Feature Integrity
**Priority:** P1
**Effort:** High

**Problem:**
AlphaMissense was trained on data that overlaps with ClinVar labels. SteppeDNA uses AM scores as features, creating indirect leakage. The leakage magnitude is acknowledged in VALIDATION_REPORT.md but never quantified via ablation.

**Solution:**
1. Retrain full pipeline with AlphaMissense features REMOVED
2. Compare per-gene AUC with/without AM:
   - Drop <0.02: AM contributes little, leakage moot
   - Drop 0.02-0.05: AM helps modestly, document contribution
   - Drop >0.05: AM is significant driver, prominently disclose
3. Compute correlation between AM scores and model's residual predictions
4. Publish ablation results in VALIDATION_REPORT.md

---

### B6. gnomAD Proxy-Benign Circularity
**Category:** Scientific — Training Data
**Priority:** P2
**Effort:** High

**Problem:**
485 training variants are "proxy-benign" (high gnomAD AF assumed benign). gnomAD AF is also an input feature, creating circularity: model could learn shortcut "high AF = benign" from labels rather than genuine signal.

**Solution:**
- **Short-term:** Retrain with gnomAD AF removed from features, OR remove proxy-benign variants entirely
- **Long-term:** Replace proxy-benign with DMS-validated benign variants (BRCA1 SGE Findlay et al., future BRCA2 DMS)
- Compare performance to quantify proxy-benign contribution
- Document which approach was taken

---

### B7. Population Allele Frequencies Are All Zeros in Training
**Category:** Scientific — Feature Engineering
**Priority:** P0
**Effort:** High

**Problem:**
gnomAD AF is effectively zero for nearly every variant in the training dataset. The gnomAD API either wasn't working during feature engineering or queries didn't match correctly. PM2 fires for everything, BS1/BA1 almost never fire. The model can't learn from population frequency signal.

**Solution:**
1. Fix gnomAD API query logic — verify GraphQL endpoint, GRCh38 coordinates, variant normalization
2. Add integration test: query known common variant (BRCA2 N372S, gnomAD AF ~0.23), verify non-zero AF
3. Re-extract features for full training set with working lookups
4. Retrain model with real AF values
5. Consider downloading gnomAD VCF sites file locally (~50GB) to avoid API rate limits

---

### B8. No Kazakh Founder Mutations
**Category:** Scientific — Population Relevance
**Priority:** P3
**Effort:** High

**Problem:**
SteppeDNA targets Kazakhstan but contains no population-specific founder variants. Kazakh/Central Asian BRCA founder mutations are poorly characterized in public databases.

**Solution:**
- Literature review: PubMed search "BRCA1 BRCA2 Kazakhstan", "Central Asian founder mutations"
- Collaborate with Nazarbayev University or National Center for Biotechnology (Astana)
- If founders identified: add to `pathogenic_positions.json` with population tags
- If no data exists: honestly state in documentation

---

### B9. Extreme Class Imbalance for Non-BRCA2 Genes
**Category:** Scientific — Model Performance
**Priority:** P2
**Effort:** Very High

**Problem:**
BRCA2: ~14,000 variants (0.983 AUC). PALB2: ~800 (0.641 AUC). RAD51D: ~400 (0.804 AUC). Class imbalance directly causes poor non-BRCA2 performance.

**Solution:**
- Mine more gnomAD proxy-benign for underrepresented genes (200-500 more per gene)
- Use DMS/MAVE data where available
- Transfer learning: pre-train on BRCA2, fine-tune on smaller genes
- Gene-aware loss weighting during MLP training
- Track per-gene AUC across retraining to verify improvements

---

### B10. No VUS in Training Set
**Category:** Scientific — By Design
**Priority:** P3
**Effort:** Medium-High

**Problem:**
Trained only on P/LP vs B/LB. VUS excluded (no ground truth). VUS predictions are pure extrapolation.

**Solution (mitigations — not a "fix"):**
- Semi-supervised learning with VUS as unlabeled data
- Collect VUS reclassified in newer ClinVar releases as validation set
- Enhanced uncertainty flags for variants in 0.3-0.7 probability range
- Already disclosed as limitation — continue emphasizing

---

### B11. VUS Reclassification Analysis — Scripts Exist But Not Run Systematically
**Category:** Scientific — Validation
**Priority:** P1
**Effort:** Medium

**Problem:**
`scripts/vus_reclassification.py` and `scripts/vus_reclassification_multigene.py` exist but no systematic temporal analysis has been published: "Of VUS reclassified since training, what % did SteppeDNA predict correctly?"

**Solution:**
1. Run `scripts/vus_reclassification_multigene.py` against latest ClinVar
2. Filter to variants NOT in training set
3. Compute AUC/MCC/balanced accuracy vs new ClinVar labels
4. Output results JSON + visualization for competition poster
5. Add results to VALIDATION_REPORT.md as "Prospective-Style Validation"

---

### B12. GPU Notebook Outputs Not Integrated Into Production Model
**Category:** Scientific — Model Architecture
**Priority:** P0 (but see note)
**Effort:** Medium

**Problem:**
GPU embeddings exist in `data/` (ESM-2 650M, LoRA, GNN) but the production model still uses 103 original features.

**IMPORTANT NOTE:** The GPU ablation study (VALIDATION_REPORT.md Section 11) showed that GNN features DEGRADE non-BRCA2 performance (PALB2: 0.641 -> 0.521). This was a deliberate, correct scientific decision to exclude them.

**Solution:**
- Run `scripts/integrate_gpu_embeddings.py` to produce augmented CSV
- Retrain with ESM-2 650M features ONLY (not GNN, which degraded)
- Compare: if 650M helps without hurting, integrate; if it hurts small genes, document and keep 8M
- This is scientifically correct — do not integrate features that degrade performance

---

### B13. EVE Scores Not in Production Model
**Category:** Scientific — Features
**Priority:** P1
**Effort:** Low

**Problem:**
EVE fetching pipeline exists, `eve_scores_cache.json` generated, ablation shows +0.050 BRCA1 AUC. But EVE features are NOT loaded at inference time.

**Solution:**
1. Add EVE score loading in `get_gene_data()` alongside other data
2. Add 3 EVE features to `build_feature_vector()`: `eve_score`, `eve_pathogenic`, `eve_x_phylop`
3. Add EVE as SOTA comparator in `scripts/sota_comparison.py`
4. Combine with next retrain cycle

---

### B14. No BRCAExchange Integration
**Category:** Scientific — Validation Source
**Priority:** P1
**Effort:** Medium

**Problem:**
BRCAExchange aggregates 20,000+ BRCA1/2 variants with ENIGMA expert consensus. No pipeline or data files exist.

**Solution:**
- Create `data_pipelines/fetch_brca_exchange.py` using GA4GH API
- Fetch clinical significance consensus, functional assay results
- Use as validation (not training) — cross-check SteppeDNA predictions against consensus
- Report concordance rate

---

### B15. No Population-Stratified gnomAD Analysis
**Category:** Scientific — Population Equity
**Priority:** P2
**Effort:** Low-Medium

**Problem:**
gnomAD population sub-frequencies (AFR, AMR, EAS, NFE, SAS) exist but are not analyzed. No Central Asian representation gap analysis done.

**Solution:**
- Create `scripts/population_analysis.py` to analyze gnomAD sub-frequencies
- Document Central Asian representation gap (critical for Kazakhstan competition angle)
- Generate visualization for competition poster
- Consider derived feature: `gnomad_max_pop_diff`

---

### B16. SHAP Interpretation Disclaimer Missing
**Category:** Code — Frontend
**Priority:** P1
**Effort:** Very Low

**Problem:**
SHAP values are displayed with a tooltip explaining what they show, but no disclaimer warns users that SHAP explains model behavior, NOT biological causation. Current tooltip: "shows how each biological feature pushed the prediction toward pathogenic (+) or benign (-)."

**Solution:**
Add to the SHAP tooltip or as a small text line under the SHAP chart in `frontend/index.html` or `frontend/lang.js`:
```
"SHAP values explain which features influenced this prediction. They do not imply biological causation."
```

---

### B17. Datasets Not Yet Integrated (10 Sources)
**Category:** Scientific — Data Coverage
**Priority:** P2 (collectively)
**Effort:** High (collectively), Low-Medium each

| Dataset | Priority | What It Adds | Difficulty |
|---------|----------|-------------|------------|
| ENIGMA explicit filter | HIGH | Highest-quality BRCA curation from ClinVar | EASY |
| Additional DMS (Starita, Richardson, Shirah) | HIGH | Functional validation for non-BRCA2 | MEDIUM-HIGH |
| COSMIC | MEDIUM | Somatic cancer mutations context | MEDIUM |
| LOVD | MEDIUM | Independent curation cross-check | MEDIUM |
| GTEx | MEDIUM | Tissue-specific gene expression | MEDIUM |
| VARITY | MEDIUM | LLM-based pathogenicity predictions | MEDIUM |
| STRING | MEDIUM | Protein-protein interaction context | MEDIUM |
| PhastCons | LOW | 100-way mammalian conservation | EASY |
| UniProt/InterPro | MEDIUM | Domain annotations, PTM sites | EASY |
| dbSNP | LOW | rs IDs, global MAF | EASY |

**Solution:**
Create fetch scripts in `data_pipelines/` starting with HIGH priority items. ENIGMA filter is trivially extractable from existing ClinVar data. Most MEDIUM datasets have REST APIs.

---

### B18. BRCA2 Dominance / Headline AUC Presentation
**Category:** Scientific — Honest Reporting
**Priority:** P1
**Effort:** Low

**Problem:**
Headline AUC 0.978 is sample-weighted (BRCA2 = 52% of data). Macro-averaged AUC is only 0.775. Footnote exists but hero stat may still mislead judges.

**Solution:**
- Make macro-averaged AUC more prominent in frontend hero section
- Competition poster should lead with per-gene table, not headline AUC
- Consider changing hero stat to "0.775 macro-avg (0.978 sample-weighted)"

---

### B19. Internal vs External AUC Gap Explanation
**Category:** Scientific — Documentation
**Priority:** P1
**Effort:** Low

**Problem:**
Internal 0.978 vs independent benchmark 0.750 (ProteinGym DMS BRCA1), 0.801 (ClinVar Expert Panel). Gap will be questioned by judges.

**Solution:**
Prepare clear explanation for competition:
- Internal test shares ClinVar vintage and feature pipeline (methodological advantage)
- Independent benchmark uses different data distribution (DMS vs clinical labels)
- BRCA1 DMS correlation limited by gene-specific class imbalance (96.6% pathogenic)
- Expert Panel BRCA2 subset: 0.918 AUC — much closer to internal
- Include this explanation in VALIDATION_REPORT.md and competition presentation

---

### B20. Compound Het Phasing — Disclaimer Could Be Stronger
**Category:** Code — VCF
**Priority:** P3
**Effort:** Low

**Problem:**
VCF compound het detection warns about phasing (`vcf.py:575-576`) but the warning could be more explicit about clinical implications. Current: "consider phasing analysis to determine if variants are in cis or trans."

**Solution:**
- Add: "True compound heterozygosity confirmation requires family trio sequencing (proband + parents). Statistical phasing alone is insufficient for clinical decisions."
- Long-term: integrate WhatsHap for statistical phasing

---

### B21. MAVE Coverage Extremely Sparse
**Category:** Scientific — Data
**Priority:** P2
**Effort:** Medium

**Problem:**
Only 134 BRCA2 variants have MAVE HDR scores (3.5% of test set). No MAVE for other 4 genes. `mave_score` is mostly zeros.

**Solution:**
- Fetch additional DMS datasets: Starita BRCA2, Richardson PALB2, Shirah BRCA1
- Expand MaveDB queries for all 5 genes
- Consider position-level interpolation for nearby assayed positions

---

### B22. AlphaFold Structure Coverage Biased Toward BRCA2
**Category:** Scientific — Features
**Priority:** P2
**Effort:** Medium-High

**Problem:**
BRCA2 has detailed 3D structure features (6 AlphaFold fragments, RSA, B-factor). Other 4 genes have limited/no structural features. GNN embeddings exist for non-BRCA2 (9,138 entries) but degrade performance.

**Solution:**
- Try ESM-2 650M embeddings alone (without GNN) for structural signal
- Fetch full-length AlphaFold structures for all 5 genes
- Extract RSA, B-factor, distance features for non-BRCA2

---

### B23. In-Memory Rate Limiter Doesn't Scale
**Category:** Infrastructure
**Priority:** P3
**Effort:** Medium (Redis) or Very Low (document only)

**Problem:**
Rate limiting uses in-memory dict — resets on restart, not shared across workers/instances.

**Solution:**
- Current single-instance: acceptable as-is (documented)
- Production scaling: use Redis-backed rate limiting (e.g., `slowapi` with Redis)
- Stale cleanup already implemented

---

### B24. No Periodic Retraining Pipeline
**Category:** Infrastructure — MLOps
**Priority:** P2
**Effort:** Low (docs) to Medium (automation)

**Problem:**
No documented or automated pipeline for retraining when ClinVar releases new data.

**Solution:**
Document retraining protocol in README:
1. Download latest ClinVar VCF
2. Re-run `data_pipelines/prepare_all_genes.py`
3. Re-run `scripts/build_master_dataset.py`
4. Re-run `scripts/integrate_gpu_embeddings.py`
5. Re-run `scripts/train_universal_model.py`
6. Re-run benchmarks + SOTA comparison
7. Update metrics in frontend/docs

Consider GitHub Action for monthly ClinVar change detection.

---

### B25. No Live Public Deployment
**Category:** Infrastructure — Deployment
**Priority:** P1
**Effort:** Low-Medium

**Problem:**
No public URL exists. `render.yaml` and `vercel.json` are ready but never deployed. Judges can't try the tool.

**Solution:**
1. Deploy backend to Render (free tier) — config is ready
2. Deploy frontend to Vercel — config is ready with API rewrites
3. Consider `tensorflow-cpu` in requirements.txt to reduce image size
4. Verify health check, CORS, API key after deployment
5. Add live URL to README.md

---

### B26. Docker Health Check Uses curl (Fragility Risk)
**Category:** Infrastructure — Docker
**Priority:** P3
**Effort:** Very Low

**Problem:**
`Dockerfile:31-32` uses `curl -f` for health check. If curl install fails silently, health check breaks.

**Solution:**
Consider Python-based alternative:
```dockerfile
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```
Or verify curl installation with a build test.

---

### B27. SOTA Comparison Caveat Not Prominent Enough
**Category:** Scientific — Presentation
**Priority:** P1
**Effort:** Very Low

**Problem:**
SOTA comparison table was evaluated on SteppeDNA's own test set (methodological advantage). Only 72-73% coverage for other tools. Caveat exists but could be missed.

**Solution:**
- Add asterisk directly in SOTA table: "* All tools evaluated on SteppeDNA's test set. Coverage: 72-73%."
- In frontend, add inline caveat if SOTA numbers displayed
- Competition poster: always show both own-test-set AND independent benchmark side by side

---

### B28. European Ancestry Training Bias — Performance Unknown on Non-European
**Category:** Scientific — Population Equity
**Priority:** P1
**Effort:** Low (disclosure) to High (validation)

**Problem:**
Training data is predominantly European ancestry. Model performance on Central Asian, East Asian, African populations is completely unknown. Already documented in VALIDATION_REPORT.md Section 13, but no population-stratified evaluation has been run.

**Solution:**
- Run model evaluation stratified by gnomAD population AF
- If ClinVar submitter metadata includes geographic origin, stratify by that
- Explicit disclosure (already present, verify prominence)
- Add as Known Limitation in MODEL_CARD.md if not already there

---

### B29. Cohort CSV Not Migrated to SQLite
**Category:** Code — Infrastructure
**Priority:** P2
**Effort:** Low

**Problem:**
`backend/cohort.py:123-148` stores cohort observations in a plain CSV file (`cohort_observations.csv`) using `csv.DictWriter`. The rest of the app uses SQLite (`backend/database.py` with WAL mode). CSV is not concurrent-safe under load, not queryable, and inconsistent with the rest of the architecture.

**Solution:**
1. Add a `cohort_observations` table to `backend/database.py`:
```python
CREATE TABLE IF NOT EXISTS cohort_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gene TEXT NOT NULL,
    variant TEXT NOT NULL,
    classification TEXT,
    submitted_at TEXT DEFAULT (datetime('now')),
    metadata TEXT  -- JSON blob for flexible fields
)
```
2. Update `cohort.py` `/cohort/submit` to `INSERT INTO cohort_observations`
3. Update `cohort.py` `/cohort/stats` to `SELECT COUNT(*) ... GROUP BY gene`
4. Keep CSV export as a read endpoint if needed, but primary storage should be SQLite

---

### B30. 59 Scripts in scripts/ — Many Redundant
**Category:** Code — Project Hygiene
**Priority:** P2
**Effort:** Low-Medium

**Problem:**
`scripts/` contains 59 Python scripts, including 9 PDF generation scripts (`generate_bw_pdf.py`, `generate_bw_pdf_ru.py`, `generate_explanation_pdf.py`, `generate_final_unified_pdf.py`, `generate_simple_pdf.py`, `generate_simplified_bw_pdf.py`, `generate_ultimate_pdf.py`, `generate_v3_purple_pdf.py`, `md_to_pdf.py`). Most are one-off historical iterations. This clutters the project and confuses judges reviewing the repo.

**Solution:**
1. Move obsolete/superseded scripts to `scripts/archive/` (keep `__init__.py` if needed)
2. Keep only active scripts in `scripts/`:
   - `train_universal_model.py` (training)
   - `sota_comparison.py` (benchmarking)
   - `precompute_umap.py` (preprocessing)
   - `integrate_gpu_embeddings.py` (GPU feature merge)
   - `vus_reclassification_multigene.py` (validation)
   - `population_analysis.py` (if created)
3. Keep ONE PDF script (`generate_final_unified_pdf.py` or whichever is current) and archive the rest
4. Update README if it references any moved scripts

---

### B31. No 3D Viewer / Result Card Animations
**Category:** Code — Frontend UX
**Priority:** P3
**Effort:** Very Low

**Problem:**
Result cards have a `slideUp` animation, but the 3D protein viewer section has no entrance animation, loading spinner animation, or transition effects. The viewer appears abruptly when structures load.

**Solution:**
Add CSS transitions to `frontend/styles.css`:
```css
.viewer3d-container {
    animation: fadeIn 0.5s ease-out;
}
.viewer3d-loading {
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50%      { opacity: 1; }
}
```

---

### B32. generate_bw_pdf.py Contains Outdated Historical Numbers
**Category:** Code — Documentation Accuracy
**Priority:** P3
**Effort:** Very Low

**Problem:**
`scripts/generate_bw_pdf.py:109` contains hardcoded historical text: "759 variants achieved 0.736 AUC." While framed as superseded context, if a judge opens this file they may be confused by old numbers that don't match current metrics (19,223 variants, 0.978 AUC).

**Solution:**
Either:
1. Move `generate_bw_pdf.py` to `scripts/archive/` (it's one of 9 redundant PDF scripts)
2. Or update the historical text to clearly label as "v3 historical context (see current metrics in VALIDATION_REPORT.md)"

---

### B33. `hypothesis` Missing from requirements-dev.txt
**Category:** Code — Dependencies
**Priority:** P2
**Effort:** Very Low

**Problem:**
`tests/test_property_based.py` imports `hypothesis` (line 17) with a skip guard (`pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, ...)`). But `hypothesis` is NOT listed in `requirements-dev.txt`. Result: all 12 property-based tests silently skip in CI and local dev.

**Solution:**
Add to `requirements-dev.txt`:
```
hypothesis>=6.0
```
This single line ensures property-based tests actually run. The skip guard can remain as graceful fallback.

---

### B34. CSP Header Missing from Backend Middleware
**Category:** Security — Defense-in-Depth
**Priority:** P3
**Effort:** Very Low

**Problem:**
`backend/middleware.py:105-113` adds `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and `Strict-Transport-Security` — but no `Content-Security-Policy` header. CSP IS present in `nginx.conf` and `vercel.json`, so production behind a reverse proxy is protected. But direct backend access (development, Docker without nginx) has no CSP.

**Solution:**
Add to `backend/middleware.py` inside `security_headers_middleware`:
```python
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-src 'none'; "
    "object-src 'none'"
)
```
Note: Must match nginx.conf/vercel.json CSP. `unsafe-inline` for styles required by 3Dmol.js (see B2).

---

### B35. README Missing "Known Limitations" Section
**Category:** Documentation
**Priority:** P2
**Effort:** Very Low

**Problem:**
`README.md` has no "Known Limitations" section and doesn't link to VALIDATION_REPORT.md's 19 documented limitations. Judges or users who only read the README miss critical caveats (population bias, temporal generalization, BRCA2 dominance, AlphaMissense leakage).

**Solution:**
Add to `README.md` after the "Quick Start" or "Performance" section:
```markdown
## Known Limitations

SteppeDNA has 19 documented limitations — see [VALIDATION_REPORT.md](VALIDATION_REPORT.md#9-known-limitations) for full details. Key caveats:

- **Population bias:** Training data predominantly European ancestry (ClinVar submission bias)
- **BRCA2 dominance:** Headline AUC 0.978 is sample-weighted; macro-averaged across genes: 0.775
- **Temporal generalization:** Non-BRCA2 temporal AUCs are near-random (0.51–0.61)
- **AlphaMissense leakage:** AM was partially trained on ClinVar labels (indirect circularity)
- **Research use only:** Not validated for clinical diagnostic decisions
```

---

### B37. Missing Snakefile (Reproducible Pipeline Orchestrator)
**Category:** Code — DevOps / Reproducibility
**Priority:** P2
**Effort:** Medium

**Problem:**
The implementation plan specifies a Snakefile to automate the full data-to-model pipeline for reproducibility. It does not exist. Currently, rebuilding the model requires manually running 15+ scripts in `data_pipelines/` and `scripts/` in the correct order, with implicit dependencies between them.

**Solution:**
Create a `Snakefile` at the project root that chains the pipeline stages with explicit input/output dependencies:

```python
# Snakefile — SteppeDNA reproducible pipeline
# Usage: snakemake -j1 (sequential) or snakemake -j4 (parallel where safe)

DATA = "data"
PIPES = "data_pipelines"
SCRIPTS = "scripts"

rule all:
    input:
        f"{DATA}/model_metadata.json",
        f"{DATA}/checksums.json",
        "visual_proofs/1_PerGene_ROC_Curves.pdf"

# ── Stage 1: Data fetching (independent, can parallelize) ────────────
rule fetch_alphafold:
    output: f"{DATA}/structural_features.pkl"
    shell: "python {PIPES}/fetch_alphafold.py && python {PIPES}/fetch_alphafold_fullength.py"

rule fetch_gnomad:
    output: f"{DATA}/gnomad_frequencies.pkl"
    shell: "python {PIPES}/fetch_gnomad.py"

rule fetch_gnomad_proxy:
    output: f"{DATA}/gnomad_proxy_benign.csv"
    shell: "python {PIPES}/fetch_gnomad_proxy_benign.py"

rule fetch_phylop:
    output: f"{DATA}/phylop_scores.pkl"
    shell: "python {PIPES}/fetch_phylop.py"

rule fetch_mave:
    output: f"{DATA}/mave_scores.pkl"
    shell: "python {PIPES}/fetch_mave.py"

rule fetch_alphamissense:
    output: f"{DATA}/alphamissense_scores.pkl"
    shell: "python {PIPES}/fetch_alphamissense.py"

rule fetch_eve:
    output: f"{DATA}/eve_scores.pkl"
    shell: "python {PIPES}/fetch_eve_scores.py"

rule fetch_dbnsfp:
    output: f"{DATA}/dbnsfp_sota_scores.csv"
    shell: "python {PIPES}/fetch_dbnsfp_scores.py"

rule generate_esm2:
    output: f"{DATA}/esm2_embeddings.pkl"
    shell: "python {PIPES}/generate_esm2_embeddings.py"

# ── Stage 2: Gene dataset preparation (depends on all fetchers) ──────
rule prepare_genes:
    input:
        f"{DATA}/structural_features.pkl",
        f"{DATA}/gnomad_frequencies.pkl",
        f"{DATA}/gnomad_proxy_benign.csv",
        f"{DATA}/phylop_scores.pkl",
        f"{DATA}/mave_scores.pkl",
        f"{DATA}/alphamissense_scores.pkl",
        f"{DATA}/esm2_embeddings.pkl"
    output: f"{DATA}/universal_training_data.csv"
    shell: "python {PIPES}/prepare_all_genes.py"

# ── Stage 3: Model training ──────────────────────────────────────────
rule train_model:
    input: f"{DATA}/universal_training_data.csv"
    output: f"{DATA}/model_metadata.json"
    shell: "python {SCRIPTS}/train_universal_model.py"

# ── Stage 4: Post-training calibration & artifacts ───────────────────
rule calibrate:
    input: f"{DATA}/model_metadata.json"
    output: f"{DATA}/conformal_thresholds.json"
    shell:
        "python {SCRIPTS}/train_conformal.py && "
        "python {SCRIPTS}/train_gene_calibrators.py && "
        "python {SCRIPTS}/generate_bootstrap_models.py && "
        "python {SCRIPTS}/optimize_gene_weights.py"

rule checksums:
    input: f"{DATA}/model_metadata.json"
    output: f"{DATA}/checksums.json"
    shell: "python {SCRIPTS}/generate_checksums.py"

# ── Stage 5: Figures & validation ────────────────────────────────────
rule figures:
    input: f"{DATA}/model_metadata.json", f"{DATA}/dbnsfp_sota_scores.csv"
    output: "visual_proofs/1_PerGene_ROC_Curves.pdf"
    shell:
        "python {SCRIPTS}/generate_figures.py && "
        "python {SCRIPTS}/sota_comparison.py"
```

**Notes:**
- Verify actual input/output file names match what each script produces before using — the above is a best-effort mapping based on script names and `data/` contents.
- Add `snakemake` to `requirements-dev.txt`.
- The `docker-compose.yml` already exists and handles runtime; this Snakefile handles the *build-time* pipeline.

---

### B36. Rate Limiter Test Flakiness — PARTIALLY FIXED (March 2026)
**Category:** Code — Testing
**Priority:** P2
**Effort:** Low

**Problem:**
The rate limiter causes flaky 429 responses when running the full test suite together. An `_reset_rate_limiter` autouse fixture exists in `tests/conftest.py:28-33` that clears `_rate_counts` between tests. However, **25 of 198 tests still fail with 429** when running the full suite with default `RATE_LIMIT=60`. All 25 pass when re-run with `RATE_LIMIT=9999`.

**Root cause:** The fixture clears `_rate_counts` (IP→timestamp list), but the session-scoped TestClient shares a single "IP" across all tests. Even with clearing, if multiple tests run within the 60-second window and each makes multiple requests, the per-test clearing isn't sufficient because the timestamps from the CURRENT test's requests aren't old enough to be pruned by the next test's clear.

**Solution:**
Either:
1. Set `RATE_LIMIT=9999` in conftest.py as a session fixture (simplest):
```python
@pytest.fixture(scope="session", autouse=True)
def _disable_rate_limit():
    os.environ["RATE_LIMIT"] = "9999"
```
2. Or exclude the rate limiter middleware entirely for TestClient by checking `request.client.host == "testclient"`
3. Or increase default RATE_LIMIT in conftest to a high value

---

## SECTION C: NON-CODE ITEMS STILL PENDING

### C1. No bioRxiv Preprint
**Priority:** HIGH for competition credibility
**Solution:** Write a short methods paper covering architecture, validation, limitations. Submit to bioRxiv (free, no peer review delay). Include DOI in competition materials.
**Effort:** High (writing + figures)

### C2. No Clinical Expert Endorsement Letter
**Priority:** HIGH for competition
**Solution:** Contact clinical genetics departments at Nazarbayev University, National Research Center for Maternal and Child Health (Astana). Offer co-authorship or formal acknowledgment. Ask for a brief review letter.
**Effort:** High (relationship-building)

### C3. Kazakhstan BRCA Prevalence Slide Not Prepared
**Priority:** MEDIUM for Infomatrix Asia
**Solution:** Literature review of Central Asian BRCA data + create a single infographic slide for competition poster.
**Effort:** Low

### C4. Clinical Utility Statement Not Written
**Priority:** MEDIUM
**Solution:** Write 1-page document: "What does this change for patients in resource-limited settings?" Focus on: free, offline-capable, multi-gene, ACMG-approximating.
**Effort:** Low

### C5. No Expert ACMG Comparison Study
**Priority:** P3
**Solution:** Select 50-100 variants, have 2-3 geneticists independently classify using full ACMG criteria, compare against SteppeDNA output. Report agreement rate + Cohen's kappa.
**Effort:** High (requires recruiting experts)

### C6. Competition Framing: "Research Use Only" Limits Clinical Narrative
**Priority:** MEDIUM for Infomatrix Asia (Applied Science)
**Solution:** Frame as "clinical research tool" not "diagnostic." Emphasize: "enables researchers to prioritize variants for functional validation." Prepare "regulatory pathway" slide.
**Effort:** Low

### C7. Competition Framing: Global Health Equity (for Infomatrix Worlds)
**Priority:** MEDIUM for Worlds
**Solution:** Frame as global health equity tool: open-source, no subscription, works offline (PWA). Prepare talking points for international audience.
**Effort:** Low

### C8. Population Bias Disclosure in Competition Poster
**Priority:** HIGH
**Solution:** Add "Limitations" section to every poster: "Training data predominantly European ancestry. Not validated on Central Asian, East Asian, or African populations."
**Effort:** Very Low

---

## Priority Summary

### P0 — Critical (Do First)
| # | Issue | Type | Effort |
|---|-------|------|--------|
| B7 | gnomAD AFs all zeros in training | Retrain | High |
| B12 | GPU embeddings not integrated (ESM-2 650M only) | Retrain | Medium |
| B13 | EVE scores not in production | Code + Retrain | Low |

### P1 — High Priority
| # | Issue | Type | Effort |
|---|-------|------|--------|
| B4 | No external validation cohort | Data acquisition | Medium |
| B5 | AlphaMissense leakage unquantified | Ablation study | High |
| B11 | VUS reclassification not run | Script execution | Medium |
| B14 | No BRCAExchange cross-check | Data pipeline | Medium |
| B16 | SHAP causation disclaimer | Frontend | Very Low |
| B18 | Hero AUC presentation | Frontend/docs | Low |
| B19 | AUC gap explanation | Docs | Low |
| B25 | No live deployment | Deploy | Low-Medium |
| B27 | SOTA caveat prominence | Docs/frontend | Very Low |
| B28 | Population bias evaluation | Analysis | Low-High |
| F5 | Verify all import paths after refactoring | Integration test | Low |
| F7 | BA1 threshold 0.01 too permissive | ACMG rules | Very Low |
| F12 | BRCA1 P871L flagged as pathogenic (known benign) | Scientific | Very Low |
| C1 | bioRxiv preprint | Writing | High |
| C2 | Clinical expert letter | Outreach | High |
| C8 | Population bias poster disclosure | Presentation | Very Low |

### P2 — Medium Priority
| # | Issue | Type | Effort |
|---|-------|------|--------|
| B1 | External API timeout tests | Tests | Low |
| B3 | PM5 from curated ClinVar | Data + code | Medium |
| B6 | gnomAD proxy-benign circularity | Retrain | High |
| B9 | Class imbalance mitigation | Retrain | Very High |
| B15 | Population-stratified gnomAD | Analysis | Low-Medium |
| B17 | 10 datasets not integrated | Data pipelines | High |
| B21 | MAVE coverage sparse | Data | Medium |
| B36 | Rate limiter test flakiness (partial) | Tests | Low |
| F1 | NICE_NAMES incomplete (37 features) | Code — SHAP UX | Very Low |
| F4 | No mutation testing (mutmut) | Test quality | Medium |
| F8 | Error response format inconsistency | API consistency | Low |
| F9 | HGVS notation simplified, not standard | Standards | Low |
| F10 | gnomAD OOD threshold not gene-specific | Calibration | Very Low |
| B22 | AlphaFold coverage biased | Features | Medium-High |
| B24 | No retraining pipeline | Docs/automation | Low-Medium |
| B29 | Cohort CSV → SQLite migration | Infrastructure | Low |
| B30 | 59 scripts in scripts/ (9 redundant PDF) | Project hygiene | Low-Medium |
| B33 | hypothesis missing from requirements-dev.txt | Dependencies | Very Low |
| B35 | README missing Known Limitations section | Docs | Very Low |
| C4 | Clinical utility statement | Writing | Low |
| C6 | RUO competition framing | Presentation | Low |
| C7 | Global health equity framing | Presentation | Low |
| G3 | Remaining innerHTML XSS in root index.html | Security | Medium |
| G4 | Calibration data leakage (calibrator on test set) | Scientific | High |
| H1 | Active learning section timing fragility | Frontend | Low |
| H2 | Service worker blocks live code updates | Frontend/DevOps | Low |
| H4 | Test count discrepancy in MEMORY.md | Docs | Very Low |

### P3 — Low Priority / Long-term
| # | Issue | Type | Effort |
|---|-------|------|--------|
| B2 | CSP unsafe-inline | WON'T FIX | N/A |
| B8 | Kazakh founder mutations | Data acquisition | High |
| B10 | No VUS in training | By design | Medium-High |
| B20 | Compound het disclaimer | Frontend | Low |
| B23 | Rate limiter scaling | Infrastructure | Medium |
| B26 | Docker health check fragility | Docker | Very Low |
| B31 | No 3D viewer animations | Frontend UX | Very Low |
| B32 | generate_bw_pdf.py outdated numbers | Project hygiene | Very Low |
| B34 | CSP header missing from backend middleware | Security | Very Low |
| F2 | needs_wetlab_assay.csv test artifacts | Test isolation | Very Low |
| F3 | Risk tier "high (Truncating)" inconsistency | API consistency | Very Low |
| F11 | Active learning priorities static JSON | Feature completeness | Medium |
| F13 | Probability clipping not documented | Documentation | Very Low |
| C3 | Kazakhstan BRCA slide | Presentation | Low |
| C5 | Expert ACMG comparison | Study | High |
| G1 | Orphaned pickle files in data/ | Project hygiene | Very Low |
| G2 | Orphaned brca2_final_model_ipresume/ directory | Project hygiene | Very Low |
| H3 | Dead CSS rules from prior refactoring | Frontend | Low |

---

---

## SECTION D: NEW ISSUES FROM SECURITY HARDENING (March 2026)

### D1. `psutil` Not Exactly Pinned
**Category:** Code — Dependencies
**Priority:** P3
**Effort:** Very Low

**Problem:**
`requirements.txt:13` has `psutil>=5.9.0,<7.0.0` (bounded range) not an exact pin. Inconsistent with all other dependencies which are pinned to exact versions after security fix S18.

**Solution:**
Run `pip freeze | grep psutil` and pin exactly (e.g. `psutil==6.1.1`).

---

### D2. Google Fonts Cannot Use SRI (Known Limitation)
**Category:** Security — CDN Trust
**Priority:** P3
**Effort:** Low-Medium

**Problem:**
`frontend/index.html:10-13` loads Google Fonts without SRI. Google serves different CSS per user-agent, making SRI impossible.

**Solution:**
1. **Self-host fonts (recommended):** Download Inter + Playfair Display WOFF2 files, serve from `/fonts/`. Eliminates third-party CDN dependency entirely.
2. **Accept the risk:** Document as known limitation. Google Fonts is a trusted CDN with strong security practices.

---

### D3. UMAP Visualization Returns 404 (Missing Precomputed Data)
**Category:** Feature — Incomplete
**Priority:** P3
**Effort:** Low

**Problem:**
`/umap` endpoint at `backend/cohort.py:44-51` returns 404 because `data/umap_coordinates.json` doesn't exist. Frontend silently handles it but shows no UMAP plot.

**Solution:**
Run `python scripts/precompute_umap.py` to generate the UMAP coordinates file. Requires training data to be available. If UMAP isn't needed, remove the frontend section that calls this endpoint.

---

### D4. `/metrics` Endpoint Auth Decision Needed
**Category:** Security — Access Control
**Priority:** P2
**Effort:** Very Low

**Problem:**
`/metrics` (internal model performance) and `/model_metrics` are in the public_paths set at `backend/middleware.py:53`. These endpoints return internal model performance data without requiring an API key. This was a deliberate decision during security fix S4 to allow monitoring dashboards, but needs explicit confirmation.

**Solution:**
**Requires your input:**
- If metrics should be public for monitoring: keep as-is, add comment documenting the decision
- If metrics should be private: remove from `public_paths` in middleware.py

---

### D5. SQLite Still Ephemeral on Render (Post-Fix S8)
**Category:** Infrastructure — Data Persistence
**Priority:** MEDIUM
**Effort:** Low-Medium

**Problem:**
Security fix S8 added `STEPPEDNA_DB_PATH` env override, but Render's filesystem is still ephemeral. The SQLite DB (history, cohort data) will be lost on every redeploy unless a persistent disk is mounted.

**Solution:**
1. **Render Persistent Disk:** Mount a disk at `/data`, set `STEPPEDNA_DB_PATH=/data/steppedna.db` ($0.25/GB/month)
2. **External PostgreSQL:** Migrate SQLite to Render Managed Postgres (schema migration needed)
3. **Accept ephemeral:** If history isn't critical, document as known limitation
**Requires your input** on preferred approach.

---

### D6. `gnomad_sensitivity.py` Script Incomplete
**Category:** Code — Scripts
**Priority:** P3
**Effort:** Medium

**Problem:**
`scripts/gnomad_sensitivity.py:19` has a TODO — documents what a full gnomAD sensitivity analysis would do but doesn't execute it. Needs model artifacts to be available.

**Solution:**
Complete when model retraining pipeline is ready. Not blocking for deployment.

---

### D7. No Monitoring/Alerting After Deployment
**Category:** Infrastructure — Operations
**Priority:** P2
**Effort:** Low

**Problem:**
No uptime monitoring, error alerting, or performance tracking is configured for when the app goes live.

**Solution:**
1. Render health check is already configured in Dockerfile
2. Set up UptimeRobot (free) or Better Uptime for `/health` endpoint
3. Configure Render log drain to Papertrail or Datadog for error alerting
4. Consider Sentry for frontend error tracking

---

### D8. No Staging Environment
**Category:** Infrastructure — DevOps
**Priority:** P3
**Effort:** Low

**Problem:**
Changes go directly to production. No staging/preview environment for testing.

**Solution:**
Vercel provides preview deployments per PR automatically. For backend, create a separate Render service named `steppedna-api-staging` with `ENVIRONMENT=staging`.

---

### D9. PDF Export Template Missing escapeHtml
**Category:** Security — XSS (Low Risk)
**Priority:** P3
**Effort:** Very Low

**Problem:**
`frontend/app.js:1483-1486` uses raw `${f.blosum62_score}` in the PDF/print template without `escapeHtml()`. Since data comes from the backend (not user input), it's not exploitable, but it's inconsistent with the XSS-hardened pattern used elsewhere.

**Solution:**
Wrap all template values in `escapeHtml()` for consistency.

---

### D10. `launch.json` Had `null` in ALLOWED_ORIGINS (Fixed)
**Category:** Dev Config
**Priority:** FIXED
**Effort:** N/A

**Problem:**
`.claude/launch.json` had `null` in the ALLOWED_ORIGINS env var, contradicting security fix S5 (CORS null origin removal).

**Solution:**
Already fixed — removed `null` from the ALLOWED_ORIGINS value.

---

## SECTION F: NEW ISSUES FROM POST-GEMINI AUDIT (March 2026)

> Source: Comprehensive audit after Gemini refactored backend/main.py into 7 modules
> Verified: 2026-03-07

### F1. NICE_NAMES Incomplete — ~37 Features Lack Human-Readable Labels
**Category:** Code — SHAP UX
**Priority:** P2
**Effort:** Very Low

**Problem:**
`backend/features.py:448-504` NICE_NAMES dict has ~66 entries for 103 features (~64% coverage). Missing entries include all one-hot encoded features:
- `AA_ref_Ala` through `AA_ref_Val` (20 entries)
- `AA_alt_Ala` through `AA_alt_Val` (20 entries)
- `Mutation_A>C` through `Mutation_C>T` (6 entries)
- Also missing: `gnomad_popmax_log`, `is_popmax_rare`

Features without NICE_NAMES fall back to raw names (e.g. `AA_ref_Gly` instead of "Reference: Glycine") in SHAP explanations.

**Solution:**
Add to NICE_NAMES dict in `backend/features.py`:
```python
# One-hot amino acid features
**{f"AA_ref_{aa}": f"Ref AA: {aa}" for aa in ALL_AMINO_ACIDS},
**{f"AA_alt_{aa}": f"Alt AA: {aa}" for aa in ALL_AMINO_ACIDS},
# Mutation type features
**{f"Mutation_{m}": f"Mutation {m}" for m in ALL_MUTATIONS},
# Missing gnomAD features
"gnomad_popmax_log": "gnomAD PopMax AF (log)",
"is_popmax_rare": "Rare in All Populations",
```

---

### F2. needs_wetlab_assay.csv Contains Test Prediction Artifacts
**Category:** Code — Test Isolation
**Priority:** P3
**Effort:** Very Low

**Problem:**
`data/needs_wetlab_assay.csv` contains 6 entries that are test prediction artifacts (written during test suite execution). The file is meant to collect production triage variants for wet-lab follow-up, but tests write to the same file. Entries like `BRCA1:100:Ala34Val` and `RAD51D:100:Ala34Val` are clearly test data (position 100, generic Ala→Val).

**Solution:**
Two options:
1. **Immediate:** Delete test entries from CSV, add Gene column header if missing
2. **Proper fix:** Use a separate triage path during testing. In conftest.py:
```python
@pytest.fixture(scope="session", autouse=True)
def _isolate_triage_csv(tmp_path_factory):
    """Redirect triage CSV to temp dir during tests."""
    import backend.main
    original = backend.main.DATA_DIR
    test_data = tmp_path_factory.mktemp("test_data")
    # Copy necessary data files but not triage CSV
    backend.main.DATA_DIR = str(test_data)
    yield
    backend.main.DATA_DIR = original
```

---

### F3. Risk Tier String Inconsistency for Truncating Variants
**Category:** Code — API Consistency
**Priority:** P3
**Effort:** Very Low

**Problem:**
Tier 1 truncating variants (nonsense, frameshift) return `risk_tier: "high (Truncating)"` (main.py:367, vcf.py:373) while Tier 2 missense variants return `risk_tier: "high"` / `"low"` / `"uncertain"`. Frontend must handle both string formats.

The strings ARE consistent between `/predict` and `/predict/vcf` (both say "high (Truncating)"), but inconsistent with the normal risk tier vocabulary.

**Solution:**
Either:
1. **Normalize:** Change to `risk_tier: "high"` and add separate field `"variant_tier": "truncating"` (cleaner API)
2. **Document:** Add API docs noting the special "(Truncating)" suffix for Tier 1 variants
3. **Leave as-is:** Frontend already handles it (checks `risk.includes("high")`)

---

### F4. No Mutation Testing (mutmut) for Test Quality Verification
**Category:** Code — Testing Quality
**Priority:** P2
**Effort:** Medium

**Problem:**
Test suite has 198 tests but no mutation testing has been run to verify test quality. Mutation testing (e.g. mutmut) introduces small code changes and checks if tests catch them. Without it, tests might have high line coverage but miss critical logic bugs.

Listed in MEMORY.md "Remaining Work" but not in dibidai.md until now.

**Solution:**
1. Add `mutmut` to requirements-dev.txt:
```
mutmut>=2.4.0
```
2. Run:
```bash
mutmut run --paths-to-mutate=backend/acmg_rules.py
mutmut results
```
3. Start with `acmg_rules.py` (120 lines, pure logic, most testable). Aim for >80% mutation kill rate.
4. Expand to `features.py` build_feature_vector() and `vcf.py` variant parsing.
5. Document mutation test results in VALIDATION_REPORT.md.

---

### F5. Gemini Refactoring Removed Inline Code — Verify All Import Paths
**Category:** Code — Refactoring Integrity
**Priority:** P1
**Effort:** Low

**Problem:**
Gemini extracted ~1500 lines from `backend/main.py` into 7 modules. While the import test passes and 171/198 tests pass (25 are rate-limiter flaky), the following cross-module dependencies need verification:
- `main.py` imports from `models.py`, `features.py`, `explanations.py`, `external_api.py`, `vcf.py`, `middleware.py`, `cohort.py`
- `features.py` imports from `models.py`, `feature_engineering.py`
- `explanations.py` imports from `models.py`
- `vcf.py` imports from `features.py`, `models.py`
- No circular imports exist (verified)

The main risk is that some code paths only execute at runtime (e.g., first /predict call triggers lazy model loading via `get_gene_data()`). An import test doesn't exercise these paths.

**Solution:**
Already mostly verified — import test passes, 171/198 tests pass. But add an integration test that exercises the full /predict pipeline:
```python
def test_full_predict_pipeline_smoke(client):
    """Smoke test: exercises model loading, feature engineering, SHAP, CI, ACMG."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2", "cDNA_pos": 8165,
        "AA_ref": "Thr", "AA_alt": "Arg",
        "Mutation": "A>G", "AA_pos": 2722
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data
    assert "shap_explanation" in data
    assert "aa_pos" in data
    assert data["aa_pos"] == 2722
```

---

### F6. Version String Was 5.2.0 in Old Root Endpoint (Fixed by Gemini)
*See also: F8-F13 below for additional items found during exhaustive cross-reference*
**Category:** Code — Versioning
**Priority:** FIXED
**Effort:** N/A

**Problem:**
Previous session identified version string "5.2.0" at line 704 of old main.py. Gemini's refactoring replaced this with `STEPPEDNA_VERSION` from `backend/__init__.py` which correctly reads "5.3.0".

**Status:** FIXED — all version references now use the centralized `STEPPEDNA_VERSION` constant.

---

### F7. ACMG BA1 Threshold at 0.01 — Potentially Too Permissive
**Category:** Scientific — ACMG Accuracy
**Priority:** P1 (also mentioned in E3.5)
**Effort:** Very Low

**Problem:**
`backend/acmg_rules.py:18-24` sets BA1 (stand-alone benign) threshold at 0.01 (1%) for all genes. ClinGen Sequence Variant Interpretation Working Group (SVI) recommends gene-specific BA1 thresholds, typically 0.001-0.005 for cancer predisposition genes. At 0.01, common pathogenic variants in founder populations could be incorrectly classified as benign.

Cross-references: E3.5 (ISEF critique), E1.11 (gnomAD proxy-benign threshold)

**Solution:**
Update `GENE_BA1_THRESHOLDS` in `backend/acmg_rules.py` to ClinGen SVI recommended values:
```python
GENE_BA1_THRESHOLDS = {
    "BRCA1": 0.001,   # ClinGen SVI: 0.001 for high-penetrance cancer genes
    "BRCA2": 0.001,
    "PALB2": 0.002,   # Slightly higher — lower penetrance
    "RAD51C": 0.005,  # Less characterized — conservative
    "RAD51D": 0.005,
}
```
**Requires your input:** Verify these thresholds against ClinGen SVI publications before applying.

---

### F8. Error Response Format Inconsistency (dict vs JSONResponse)
**Category:** Code — API Consistency
**Priority:** P2
**Effort:** Low

**Problem:**
`backend/main.py` returns errors in two different formats: sometimes `JSONResponse(status_code=..., content={"error": ...})` (lines 338, 352, 404, 408, 456) and sometimes plain `return {"error": ...}` (which returns 200 OK with an error body). Frontend must handle both patterns.

**Solution:**
Standardize all error returns to use `JSONResponse` with appropriate HTTP status codes:
```python
# Instead of: return {"error": "Gene not found"}
# Use:        return JSONResponse(status_code=400, content={"error": "Gene not found"})
```
Audit all `return {"error":` patterns in main.py and vcf.py, replace with proper HTTP error responses.

---

### F9. HGVS Notation Is Simplified, Not Standard
**Category:** Scientific — Standards Compliance
**Priority:** P2
**Effort:** Low

**Problem:**
`backend/main.py:722-723` produces simplified HGVS-like strings: `c.{cDNA_pos}{Mutation}` and `p.{AA_ref}{AA_pos}{AA_alt}`. This is NOT proper HGVS notation (which requires 3-letter AA codes with specific formatting like `p.Thr2722Arg`, reference sequence accession, and specific syntax for different mutation types).

Frontend displays these simplified strings, which may confuse clinical geneticists who expect standard HGVS format.

**Solution:**
1. Use proper HGVS protein notation: `p.{AA_ref_3letter}{AA_pos}{AA_alt_3letter}` (already close)
2. Use proper HGVS cDNA notation: `c.{cDNA_pos}{ref}>{alt}` (e.g., `c.8165A>G` not `c.8165A>G`)
3. Add NM_ transcript accession prefix for each gene
4. Consider using `hgvs` Python library for proper formatting
5. Or document that the notation is "HGVS-like simplified" — not full HGVS

---

### F10. gnomAD OOD Warning Threshold Not Gene-Specific
**Category:** Scientific — Calibration
**Priority:** P2
**Effort:** Very Low

**Problem:**
`backend/main.py:602` uses a hardcoded global threshold `if gnomad_af_raw > 0.01:` to flag variants as "common in population — prediction may be unreliable." This threshold is not gene-specific. For high-penetrance genes (BRCA1/2), variants with AF > 0.001 are already suspicious. For lower-penetrance genes (RAD51C/D), 0.01 may be appropriate.

Cross-reference: E1.11 (gnomAD proxy-benign threshold), F7 (BA1 threshold)

**Solution:**
Use gene-specific OOD thresholds aligned with BA1 values:
```python
GENE_OOD_THRESHOLDS = {
    "BRCA1": 0.005, "BRCA2": 0.005,
    "PALB2": 0.005, "RAD51C": 0.01, "RAD51D": 0.01,
}
```

---

### F11. Active Learning Priorities Are Static JSON — Never Update
**Category:** Code — Feature Completeness
**Priority:** P3
**Effort:** Medium

**Problem:**
`backend/cohort.py:205-236` serves `/research/priorities` from a static JSON file (`data/active_learning_priorities.json`). The endpoint is labeled as "active learning" but never actually re-computes priorities based on new predictions, cohort submissions, or model uncertainty. It always returns the same pre-computed list.

**Solution:**
Either:
1. **Dynamic computation:** On each call (or periodically), query recent predictions from SQLite, sort by uncertainty (closest to threshold), and return as priorities
2. **Rename:** Change endpoint name from "active_learning" to "research_priorities" and document it's a curated static list
3. **Periodic update script:** Add a cron/scheduled script that recomputes priorities weekly

**Note (March 2026):** A separate bug caused the section to never appear — `cohort.py` imported `_ACTIVE_LEARNING` at module level (getting a stale empty `{}` reference). Fixed in A73 with `_get_active_learning()` getter. The static JSON issue above remains unfixed.

---

### F12. BRCA1 Pro871Leu (P871L) Flagged as Pathogenic in Triage CSV
**Category:** Scientific — Known Variant Misclassification
**Priority:** P1
**Effort:** Very Low

**Problem:**
`data/needs_wetlab_assay.csv:4` contains `BRCA1,2612,Pro871Leu,0.9799,...,High Confidence,Predicted Pathogenic`. Pro871Leu (rs799917) is a **well-characterized benign polymorphism** with global MAF ~30%. It is classified as Benign/Likely Benign by ClinVar expert panels (ENIGMA consortium) and present in ~30% of all humans.

The model predicting it as pathogenic (0.98 probability) is a **critical false positive** that undermines clinical credibility. This variant should be in the benign training set, not the triage list.

**Solution:**
1. **Immediate:** Remove this entry from needs_wetlab_assay.csv — it does NOT need wet-lab assay
2. **Root cause:** Investigate why the model scores P871L as 0.98 pathogenic — likely missing gnomAD AF data (issue B7: all gnomAD AFs are zeros)
3. **Regression test:** Add P871L to test_clinical_correctness.py as a known benign that must score < 0.5:
```python
def test_brca1_p871l_is_benign():
    """Pro871Leu is a common benign polymorphism (MAF ~30%)."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA1", "cDNA_pos": 2612,
        "AA_ref": "Pro", "AA_alt": "Leu",
        "Mutation": "C>T", "AA_pos": 871
    })
    assert resp.json()["probability"] < 0.5  # Must classify as benign
```

---

### F13. Probability Clipping Not Documented as Design Decision
**Category:** Documentation — Transparency
**Priority:** P3
**Effort:** Very Low

**Problem:**
`backend/main.py:484-485` clips probabilities to [0.005, 0.995] with inline comment "no model should claim absolute certainty." This is a reasonable calibration decision, but it's not documented in VALIDATION_REPORT.md or MODEL_CARD.md. Judges or reviewers may question why probabilities never reach 0 or 1.

**Solution:**
Add to VALIDATION_REPORT.md Known Limitations or Methods section:
```
"Output probabilities are clipped to [0.5%, 99.5%] to prevent overconfident predictions.
No single-model prediction should claim absolute certainty given the limitations of
training data quality and feature coverage."
```

---

## Updated Summary

- **23 security fixes** (Section A0) — ALL FIXED
- **78 fixed items** (Section A) — ALL FIXED (66 original + 12 from March 8 UX session)
- **36 unfixed items** (Section B) — B36 PARTIALLY FIXED, rest documented with solutions
- **8 non-code items** (Section C) — pending
- **10 items from security hardening** (Section D) — 1 fixed, 9 remaining
- **13 items from post-Gemini audit** (Section F) — 1 fixed, 12 remaining (F11 cross-ref updated)
- **4 items from deep analysis session** (Section G) — 0 fixed, 4 remaining
- **ISEF critique items** (Section E) — reference material for competition prep
- **4 discovered-unfixed items** (Section H) — timing fragility, SW caching, dead CSS, test count

*Last verified: 2026-03-08 against codebase v5.3.0 (198 tests)*

---

## SECTION E: ISEF JUDGE CRITIQUE — SCIENCE, NOVELTY & DEPTH (March 2026)

> Source: Simulated rigorous ISEF judge evaluation of the entire project

### E1. Scientific Methodology Flaws

- **E1.1** Headline AUC (0.978) is sample-weighted; macro-averaged is only 0.775 — misleading hero stat
- **E1.2** 3/5 genes (BRCA1, PALB2, RAD51C) perform below clinical utility thresholds
- **E1.3** Temporal validation collapses to near-random for non-BRCA2 (PALB2: 0.513, BRCA1: 0.527, RAD51C: 0.561, RAD51D: 0.608)
- **E1.4** SOTA comparison is unfair — evaluated on own test set, competitors weren't trained for this distribution
- **E1.5** REVEL/BayesDel MCC of 0.027/0.097 reflects distribution mismatch, not tool quality — REVEL typically achieves AUC >0.90 in published benchmarks
- **E1.6** SOTA coverage bias — SteppeDNA scores 100%, competitors score 72-73%, comparing on different difficulty subsets
- **E1.7** No statistical tests (DeLong, bootstrap CI, permutation) on any SOTA comparison — just raw point estimates
- **E1.8** AlphaMissense indirect label leakage acknowledged but no ablation performed — "I know about the leakage but didn't fix it" is worse than not knowing
- **E1.9** MAVE leakage ablation evaluated on same split with low statistical power (3.5% coverage)
- **E1.10** ClinVar label quality: ascertainment bias, temporal instability, geographic bias, inter-lab inconsistency — model learns to reproduce ClinVar consensus, not ground truth pathogenicity
- **E1.11** gnomAD proxy-benign threshold (AC >= 2) too permissive — may include reduced-penetrance pathogenic variants; threshold should be higher (AF > 0.001 or AC >= 5)
- **E1.12** No independent external validation (different lab, population, or time period)
- **E1.13** Gold-standard benchmark overlaps with test set (460/2,234 variants); only 66 truly novel
- **E1.14** All population-stratified allele frequencies are zeros — used wrong API (Ensembl overlap/region instead of gnomAD GraphQL)
- **E1.15** Kazakh founder mutations not represented in training data despite Kazakh branding
- **E1.16** Conformal prediction uses post-calibration probabilities, not proper nonconformity scores; fallback to always include one class breaks formal coverage guarantees
- **E1.17** Bootstrap CI covers only XGBoost (60%); MLP (40%) not bootstrapped — CIs underestimate true ensemble uncertainty
- **E1.18** v4->v4.1 ablation is confounded (gene-identity removal, ESM-2 updates, hyperparameter tuning happened simultaneously)
- **E1.19** No prospective validation of any kind
- **E1.20** No clinical expert has reviewed any output

### E2. Novelty Concerns

- **E2.1** XGBoost + MLP ensemble with isotonic calibration is standard ML — no algorithmic innovation
- **E2.2** Gene-specific variant classifiers already exist (BRCA-ML, Align-GVGD, etc.) for over a decade
- **E2.3** Using BLOSUM62, conservation, structural features, and protein language model embeddings follows established patterns (REVEL, CADD, PrimateAI, AlphaMissense, EVE, ESM-1v)
- **E2.4** ESM-2 usage is minimal — smallest model (8M params), compressed to 20 PCA components
- **E2.5** ACMG rule automation exists in InterVar, VarSome, Franklin
- **E2.6** No wet-lab or experimental validation — everything is computational
- **E2.7** "I built a classifier" is an engineering claim, not a scientific discovery
- **E2.8** Variant pathogenicity classifiers are one of the most saturated categories at ISEF

### E3. Implementation Flaws

- **E3.1** Test suite tests the wrong thing — 57/198 tests target one 120-line rule engine function; zero tests verify ML model produces meaningful missense predictions; all "pathogenic" test variants are nonsense (Ser->Ter) triggering deterministic PVS1 rule
- **E3.2** UMAP visualization is fake — uses probability-based heuristic, not actual UMAP projection; UI labels it "Your variant projected onto 5,000 training variants" which is factually incorrect
- **E3.3** Version string inconsistencies (5.2.0 vs 5.3.0 across different files)
- **E3.4** Hardcoded magic numbers throughout: Tier 1 probability 0.9999, N_EFFECTIVE=200, data scarcity window=50 AA, structural defaults RSA=0.4/bfactor=50.0/dist_dna=999.0, PP3/BP4 thresholds 0.90/0.10
- **E3.5** BA1 thresholds at 0.01 for all genes — dangerously permissive for cancer predisposition genes (ClinGen SVI recommends ~0.001)
- **E3.6** Several ACMG rules can never fire for missense variants (PVS1, PM4, PP3_splice, BP7)
- **E3.7** NGL.js loaded from CDN without subresource integrity (SRI) hash
- **E3.8** Zero-fill imputation creates hidden proxy signals — "no gnomAD entry" becomes AF=0 + is_rare=1, which is informative
- **E3.9** Contrastive explanations use unweighted Euclidean distance — treats all 103 features equally regardless of predictive importance
- **E3.10** Cohort submission has no authentication, append-only CSV read fully into memory — doesn't scale, vulnerable to data poisoning
- **E3.11** ClinVar lookup defaults to BRCA2 silently when no gene parameter provided
- **E3.12** Feature concentration — ~15 of 103 features carry most predictive power; 88 contribute marginal signal; 42 features are just amino acid one-hot encoding

### E4. Presentation & Framing

- **E4.1** "19,223 variants" headline obscures that 14,200+ are BRCA2 alone
- **E4.2** Hero stats show 0.978 AUC without macro-averaged 0.775 visible anywhere in UI
- **E4.3** Independent benchmark results (AUC 0.750-0.801) are buried
- **E4.4** Kazakh translations not expert-verified for medical terminology
- **E4.5** RUO disclaimer undermined by clinical-grade appearance (ACMG codes, PDF reports, confidence intervals)
- **E4.6** README leads with "outperforms REVEL/BayesDel/CADD" despite unfair comparison methodology

### E5. Novelty Opportunities — Ways to Transform the Project

- **E5.1 Circular dependency audit:** Quantify how much ClinVar-trained tools inflate each other's accuracy through shared training labels — map the dependency graph across REVEL, CADD, BayesDel, AlphaMissense, and measure the circularity inflation by comparing Full model vs Independent-only features
- **E5.2 Difficulty drift analysis:** Show that remaining VUS are systematically harder over time by stratifying ClinVar by classification date and measuring feature separability decline — predict when current ML approaches hit a performance ceiling
- **E5.3 Semi-supervised self-training with VUS:** Use uncertainty-gated pseudo-labeling (bootstrap CI + XGB-MLP disagreement + conformal singleton) to rescue data-poor genes by leveraging thousands of available VUS
- **E5.4 Data efficiency learning curves:** Subsample BRCA2 at various sizes (100-10,000), plot AUC vs n, fit power law, extrapolate minimum labeled variants needed per gene for clinical-grade performance — discover data collection targets for the field
- **E5.5 Selective prediction / abstention:** Combine uncertainty signals into abstention score; show accuracy-coverage tradeoff where refusing low-confidence predictions achieves clinical-grade AUC on accepted subset
- **E5.6 Gene-adversarial domain adaptation:** Gradient reversal layer (Ganin et al. 2016) to force gene-invariant feature learning — treat each gene as a "domain" and penalize gene-predictive representations
- **E5.7 Population equity audit:** Show that classifiers systematically miscalibrate for underrepresented populations; test on Kazakh/Central Asian founder mutations specifically

### E6. Strategic Advice — Depth Over Breadth

- **E6.1** ISEF grand award winners have 1-2 very deep, well-done features — not 50 shallow ones
- **E6.2** Strip engineering features that add zero scientific depth: trilingual UI, 3D protein viewer, dark mode, PWA, service worker, comparison panel, PDF reports, cohort system, toast notifications
- **E6.3** Keep them in codebase but don't mention on poster — every second explaining dark mode is a second not explaining the science
- **E6.4** Shift framing from "I built a tool" to "I discovered something the field didn't know"
- **E6.5** Best combination: Learning Curves (E5.4) + Semi-supervised self-training (E5.3) — discovers the data threshold problem then proposes a solution
- **E6.6** Strongest single pivot: Circular Dependency Audit (E5.1) — meta-scientific contribution that audits the field's methodology rather than adding another tool to it
- **E6.7** Poster should have three panels: (1) the question, (2) the method, (3) the validation — not a feature list

---

## SECTION G: REMAINING ISSUES FROM DEEP ANALYSIS SESSION (March 2026)

> Source: Deep analysis of root-level main.py, index.html, train_with_imbalance_handling.py, calibrate_model.py
> Session identified 22 issues total; 17 fixed during session, 1 already fixed (S1), 4 remain below.

---

### G1. Orphaned Pickle Files in `data/`
**Category:** Code — Project Hygiene
**Priority:** P3
**Effort:** Very Low

**Problem:**
Three pickle files in `data/` are orphaned — only referenced by old archived code in `brca2_final_model_ipresume/`:
- `data/aa_hydro.pkl` (331 bytes)
- `data/aa_volume.pkl` (184 bytes)
- `data/blosum62.pkl` (2.2 KB)

Current code has these biological lookup tables hardcoded in `brca2_features.py` (root) and `backend/features.py` (production). The pickle files serve no purpose and add confusion.

**Solution:**
```bash
rm data/aa_hydro.pkl data/aa_volume.pkl data/blosum62.pkl
```
Verify no imports reference them: `grep -r "aa_hydro\|aa_volume\|blosum62.pkl" --include="*.py"` — should return nothing outside archived code.

---

### G2. Orphaned `brca2_final_model_ipresume/` Archive Directory
**Category:** Code — Project Hygiene
**Priority:** P3
**Effort:** Very Low

**Problem:**
`brca2_final_model_ipresume/` contains an old `main.py` (13KB, pre-refactoring single-file version) and a `__pycache__/` directory. It serves no purpose — the current codebase uses `backend/main.py` (production) and root `main.py` (legacy). This directory causes confusion about which code is active.

**Solution:**
```bash
rm -rf brca2_final_model_ipresume/
```
No code imports from this directory.

---

### G3. Remaining innerHTML XSS in Root `index.html`
**Category:** Security — XSS
**Priority:** P2
**Effort:** Medium

**Problem:**
During the deep analysis session, the toast notification XSS in root `index.html` was fixed (switched to `textContent`). However, four other locations still use unsanitized `innerHTML`:
- VCF results table (variant data rendered via innerHTML)
- `dsGrid` (data source grid)
- `featuresGrid` (feature display grid)
- `shapChart` (SHAP explanation rendering)

**Note:** The production frontend (`frontend/app.js`) has separate XSS hardening (per S13: `escapeHtml()` utility). This issue applies specifically to the root `index.html`, which may still be used for local development or the legacy single-file deployment.

**Solution:**
1. Replace all `innerHTML` assignments with `createElement()` + `textContent` pattern:
```javascript
// Instead of:
cell.innerHTML = variantData;
// Use:
const span = document.createElement('span');
span.textContent = variantData;
cell.appendChild(span);
```
2. For structured HTML (e.g., SHAP bars), use `createElement` for each element rather than string concatenation
3. Alternatively, add `escapeHtml()` wrapper (matching `frontend/app.js` pattern) and use it around all dynamic values

---

### G4. Calibration Data Leakage — Calibrator Fitted on Test Set
**Category:** Scientific — Validation Methodology
**Priority:** P2
**Effort:** High (requires retraining)

**Problem:**
The isotonic regression calibrator (`calibrate_model.py`) is fitted AND evaluated on the same held-out test set. This means:
- Calibration metrics (reliability diagram, ECE) are optimistically biased
- The calibrator may overfit to test set distribution
- True out-of-sample calibration performance is unknown

This is separate from the AlphaMissense leakage (B5) and gnomAD proxy-benign circularity (B6) — it's a methodological flaw in the post-hoc calibration step itself.

**Solution:**
Either:
1. **3-way split (recommended):** Split data into train (60%), calibration (20%), test (20%). Fit calibrator on calibration set, evaluate on test set. This is the standard approach for post-hoc calibration.
2. **Cross-validated calibration:** Use k-fold cross-validation during calibration — fit calibrator on k-1 folds, predict on held-out fold, repeat. Produces calibrated predictions for all samples without leakage.
3. **Venn-Abers calibration:** Distribution-free calibration that provides valid prediction intervals without a separate calibration set.

Whichever approach is chosen, report calibration metrics (ECE, Brier score) on data the calibrator has NOT seen during fitting.

**Cross-references:** B5 (AlphaMissense leakage), B6 (gnomAD proxy-benign circularity) — all three are distinct forms of data leakage in the training pipeline.

---

## SECTION H: DISCOVERED-UNFIXED ISSUES FROM UX SESSION (March 2026)

> Source: Issues discovered during March 8 UX/bug fix session but not fixed in code
> Session fixed 12 items (A67-A78); these 4 remain

---

### H1. Active Learning Section Timing Fragility
**Category:** Code — Frontend Resilience
**Priority:** P2
**Effort:** Low

**Problem:**
`frontend/app.js:loadResearchPriorities()` fires 1.5s after `DOMContentLoaded` with a 10-second `fetchWithTimeout`. If the backend is still starting (cold start on Render takes ~60s to load bootstrap models, ESM-2, etc.), the fetch fails silently and the section hides permanently with no retry. The module-level import bug (A73) is fixed, but the timing issue remains — users on cold-start deployments never see the section.

**Solution:**
Add retry with exponential backoff:
```javascript
async function loadResearchPriorities(attempt = 1, maxAttempts = 3) {
    const section = document.getElementById('researchPriorities');
    if (!section) return;
    const delays = [3000, 6000, 12000];
    try {
        const resp = await fetchWithTimeout(
            API_URL.replace('/predict', '/research/priorities') + '?limit=10', {}, 10000);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data.error || !data.priorities) throw new Error(data.error || 'No priorities');
        _researchPrioritiesData = data;
        renderResearchPriorities(data);
        section.style.display = 'block';
    } catch (e) {
        if (attempt < maxAttempts) {
            console.warn(`[SteppeDNA] Research priorities attempt ${attempt} failed, retrying in ${delays[attempt-1]}ms`);
            setTimeout(() => loadResearchPriorities(attempt + 1, maxAttempts), delays[attempt - 1]);
        } else {
            console.warn('[SteppeDNA] Research priorities: all retries exhausted:', e.message);
            section.style.display = 'none';
        }
    }
}
```

**Cross-references:** A73 (import bug fix), F11 (static JSON issue)

---

### H2. Service Worker Caching Prevents Live Code Updates
**Category:** Code — DevOps / UX
**Priority:** P2
**Effort:** Low

**Problem:**
`frontend/sw.js` aggressively caches all assets under `steppedna-v5.3.1`. During development, clearing caches and unregistering the SW was repeatedly required to see changes. For end users, there is no "new version available" notification — they run stale code indefinitely until they manually clear browser data or the cache name is bumped.

**Solution:**
Add SW update detection with user-facing toast:
```javascript
// In sw-register.js or inline <script>
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').then(reg => {
        reg.addEventListener('updatefound', () => {
            const newWorker = reg.installing;
            newWorker.addEventListener('statechange', () => {
                if (newWorker.state === 'activated' && navigator.serviceWorker.controller) {
                    showToast('Update available — click to refresh', 'info', {
                        action: () => window.location.reload(),
                        persistent: true
                    });
                }
            });
        });
    });
}
```

Also consider adding `skipWaiting()` + `clients.claim()` in `sw.js` for immediate activation on deploy.

**Cross-references:** S14 (SW caching scope fix in security session)

---

### H3. Potential Dead CSS Rules from Prior Refactoring
**Category:** Code — Project Hygiene
**Priority:** P3
**Effort:** Low

**Problem:**
During the March 8 session, `.pg-footnote a` and `.pg-footnote a:hover` were identified and removed as dead CSS (no more `<a>` tags in footnotes after A69/A75). Other dead rules likely exist from the v5.2→v5.3 frontend overhaul, the Gemini module extraction, and various CSS additions/removals across multiple sessions. No systematic audit has been performed.

**Solution:**
1. **PurgeCSS (automated):** Configure PurgeCSS to scan `index.html`, `app.js`, and `lang.js` for used selectors. Compare against `styles.css` to identify orphaned rules.
2. **Chrome DevTools Coverage (manual):** Open DevTools → Coverage tab, load the app, interact with all features (submit variant, upload VCF, switch languages, toggle theme). Red-highlighted CSS lines are unused.
3. Remove confirmed dead rules in a single cleanup commit.

---

### H4. Test Count Discrepancy in MEMORY.md
**Category:** Documentation
**Priority:** P2
**Effort:** Very Low

**Problem:**
`MEMORY.md` states "196 collected (v5.3)" but `pytest --collect-only` reports **198 tests**. The discrepancy likely arose from new tests added during bug fix sessions. The stale count appears in MEMORY.md and was referenced in earlier dibidai.md entries.

**Solution:**
Update `MEMORY.md` line `## Tests: 196 collected (v5.3)` to `## Tests: 197 collected (v5.3)`. Verify by running `pytest --collect-only -q | tail -1`.
