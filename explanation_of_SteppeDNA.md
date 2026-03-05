# Comprehensive Explanation of SteppeDNA (v3)

**SteppeDNA** is a bioinformatics and machine learning platform designed to classify missense mutations in Homologous Recombination (HR) DNA repair pathway genes as either **Pathogenic** (disease-causing, specifically increasing the risk of breast / ovarian / pancreatic cancer) or **Benign** (harmless).

**Project Scale:** 154 files | ~13,097 Lines of Code

## 1. The Core Problem and What We Solve
When patients get genetic testing (e.g., a breast cancer gene panel), the results often return as **Variants of Uncertain Significance (VUS)**. This means the laboratory identified a mutation but lacks the clinical or experimental data to know if it is dangerous. Wet-lab functional assays take months. SteppeDNA solves this by instantly analyzing the biochemical, evolutionary, structural, and population-level implications of a variant to predict its pathogenicity using an ensemble machine learning model.

### Evolution of Architecture
*   **v1 (The "0.99 Cheat" Era):** Focused only on BRCA2 and heavily utilized MAVE (Multiplexed Assay of Variant Effect) scores as features. It resulted in a massive 0.99 ROC-AUC, but essentially only worked because it was being fed direct lab results as a feature.
*   **v2 (The "0.91 / 0.77 Blind" Era):** We dropped the MAVE features completely to test the model's true predictive power. On held-out non-MAVE data, it reported 0.91 AUC. However, on strict true-independent held-out MAVE structures, the real AUC was ~0.77.
*   **v3 (The "0.73 Universal" Era - Current):** SteppeDNA expanded to encompass 5 full genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D). The model now has to learn generalized, multi-protein structural rules. The 5-Fold Cross Validated, non-leaked true ROC-AUC is **0.736**. This represents a highly capable generalized model rather than an overfit specific-gene model.

## 2. External Models, Databases, and Tools Implemented
SteppeDNA does not reinvent basic biology; it aggregates massive amounts of external computational data:

1.  **AlphaMissense (Google DeepMind):** Deep learning pathogenicity predictions based on AlphaFold structures. We map the AM score directly to the AA change.
2.  **MAVE (Multiplexed Assay of Variant Effect):** Wet-lab survival score datasets acting as high-confidence biochemical priors when available.
3.  **PhyloP (PhyloP100way):** Evolutionary conservation scores tracking an amino acid's immutability across 100 vertebrate species over millions of years.
4.  **gnomAD (Genome Aggregation Database):** Massive cohort database. Used to determine Allele Frequency (AF). If a variant is common in the global population, it is highly likely to be benign.
5.  **SpliceAI (Illumina):** Deep neural network predicting if a mutation accidentally disrupts an RNA splicing site rather than just changing the amino acid.
6.  **ESM-2 (Meta AI):** An 8-Million parameter Protein Language Model (LLM for biology). We extract a 64-dimensional latent embedding space and perform PCA to capture the "grammar" of the protein sequence around the mutation.
7.  **ClinVar (NIH/NCBI):** The ground-truth database used to label our training rows as Pathogenic (1) or Benign (0).

## 3. Detailed Feature Engineering (128 Features)
Every prediction utilizes a 128-dimensional tensor array per variant.

*   **Physicochemical Features:**
    *   `volume_diff`: Absolute volume change between reference and alternate amino acid (checks for steric clashes).
    *   `hydro_diff`, `ref_hydro`, `alt_hydro`, `hydro_delta`: Tracks shift in hydrophobicity, checking if a hydrophobic core is being incorrectly exposed to water.
    *   `charge_change`, `nonpolar_to_charged`: Tracks ionic disruption.
*   **Substitution Matrix:**
    *   `blosum62_score`: Log-odds score of the amino acid swap occurring naturally via evolution.
*   **Structural & Geometric Features:**
    *   `rsa`: Relative Solvent Accessibility (is the residue buried?).
    *   `is_buried`: Boolean flag derived from RSA.
    *   `bfactor`: Local flexibility / thermal motion confidence of the specific spatial region.
    *   `dist_dna`: Distance in Angstroms from the mutated residue to the nearest DNA-binding interface.
    *   `dist_palb2`: Distance to the PALB2 binding domain.
    *   `is_dna_contact`: Boolean threshold (distance < 5A).
    *   `ss_helix`, `ss_sheet`: Secondary Structure boolean flags.
*   **Interaction Features (Cross-products):**
    *   `conserv_x_blosum`: Heavy penalty if a highly conserved spot gets a very unnatural amino acid swap.
    *   `buried_x_blosum`: Penalty for disrupting the hydrophobic core.
*   **Gene Specific Functional Domain Flags:**
    *   `in_BRC_repeats`, `in_DNA_binding`, `in_OB_folds`, `in_NLS`, `in_PALB2_bind`
*   **ESM-2 Embeddings:**
    *   `esm2_pca_0` through `esm2_pca_19`: The top 20 principal components of the 64-dim ESM latent space.
*   **Categorical Encodings:**
    *   `Mutation_A>C`, `Mutation_T>G`, etc.: The explicit nucleic acid shift.
    *   `AA_ref_Met`, `AA_alt_Lys`, etc.: One-hot encodings of the specific amino acids.

## 4. The Machine Learning Architecture
To prevent single-model failure modes, SteppeDNA utilizes a **Heterogeneous Deep Ensemble**.

1.  **XGBoost Classifier (Weight: 60%):** A tree-based model trained with 300 estimators (max depth 6). XGBoost is an expert at managing sparse tabular data and making hard thresholds.
2.  **Deep Neural Network (MLP) (Weight: 40%):** A 3-layer deep network (128x64x32 nodes) built in TensorFlow. The NN excels at extracting non-linear topological relationships from the continuous ESM-2 latent space embeddings.
3.  **Isotonic Regression Calibrator:** The raw output of both models is blended, and passed through an Isotonic Calibrator. This ensures the output is a true biological "probability" (0.0 to 1.0) rather than an arbitrary ML confidence score.

### Data Leakage & Overfitting Prevention
*   **Strict Isolation:** Scaling (`StandardScaler`) is fitted *exclusively* on the Training set. The holdout test set is never seen by the scaler prior to evaluation.
*   **SMOTE (Synthetic Minority Over-sampling Technique):** The dataset is heavily imbalanced towards Benign. SMOTE is applied *only* on the training set to prevent synthetic data leakage into test evaluations.
*   **Optimal Thresholding:** We use the Precision-Recall curve to find the exact threshold that maximizes the F1-score mathematically, taking human bias out of the threshold pick.

## 5. ACMG / AMP Clinical Rule Engine
The backend runs automated Evidence Rules based on the American College of Medical Genetics guidelines:
*   **PM1 (Moderate Pathogenic):** Triggers if the `dist_dna` <= 5A or `dist_palb2` <= 5A, or if the variant falls in a BRC repeat. Meaning: "Structural Disruption."
*   **BS1 (Strong Benign):** Triggers if `gnomad_af` > 0.05 (5%). Meaning: "Population frequency is too high for a highly penetrant disorder."
*   **PP3 (Supporting Pathogenic):** Triggers if model probability >= 0.90.
*   **BP4 (Supporting Benign):** Triggers if model probability <= 0.10.

## 6. Real-Time Explainability
SteppeDNA does not just return a prediction; it utilizes **SHAP (SHapley Additive exPlanations)**. The backend dynamically calculates exactly which of the 128 features pushed the prediction towards Benign or Pathogenic, creating a visual waterfall graph that medical geneticists can actually interpret.

---

> [!NOTE] 
> Below is a placeholder for a screenshot of the SteppeDNA Version 3 User Interface in action, showing the glassmorphism design, variant input structure, and the SHAP output graph.

![SteppeDNA System Interface Placeholder](placeholder.jpg)

---
