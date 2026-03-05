from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# --- STYLE CONFIG ---
COLOR_PRIMARY = (0, 0, 0)      # Pure Black
COLOR_TEXT = (30, 30, 30)      # Almost Black text for readability
COLOR_SUBTITLE = (80, 80, 80)  # Dark Gray
FONT_FAMILY = "times"          # Elegant Academic Serif Font

class SteppeDNAPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font(FONT_FAMILY, "B", 24)
            self.set_text_color(*COLOR_PRIMARY)
            self.cell(0, 20, "SteppeDNA v5.0 - Comprehensive Deep Dive",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            self.set_draw_color(*COLOR_PRIMARY)
            self.set_line_width(0.8)
            self.line(self.get_x(), self.get_y()-5, self.get_x()+190, self.get_y()-5)
            
            self.set_font(FONT_FAMILY, "I", 12)
            self.set_text_color(*COLOR_SUBTITLE)
            self.multi_cell(0, 10, "A granular breakdown of the true architecture, performance metrics (0.978 ROC-AUC), exact 103 feature set, external datasets, and data leakage prevention mechanisms.")
            self.ln(5)
            self.set_draw_color(200, 200, 200) # Light gray line
            self.set_line_width(0.2)
            self.line(self.get_x(), self.get_y(), self.get_x()+190, self.get_y())
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"SteppeDNA Comprehensive Deep Dive | Validation Metrics | Page {self.page_no()}", align="C")

    def add_section_header(self, text):
        self.ln(8)
        self.set_font(FONT_FAMILY, "B", 18)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*COLOR_PRIMARY)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y() - 1, self.get_x() + 190, self.get_y() - 1)
        self.ln(4)

    def add_sub_header(self, text):
        self.ln(4)
        self.set_font(FONT_FAMILY, "B", 14)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*COLOR_TEXT)
        self.ln(1)

    def add_bullet(self, label, text):
        self.ln(1)
        processed_text = f"- {label}: {text}" if label else f"- {text}"
        self.set_left_margin(15)
        self.set_font(FONT_FAMILY, "", 12)
        self.set_text_color(*COLOR_TEXT)
        self.multi_cell(0, 6, processed_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_left_margin(10) # Reset to default margin
        self.ln(1)
        
    def draw_placeholder_box(self):
        self.ln(10)
        start_x = self.get_x() + 10
        start_y = self.get_y()
        self.set_draw_color(150, 150, 150)
        self.set_line_width(0.5)
        
        width = 170
        height = 90
        
        self.rect(start_x, start_y, width, height)
        
        self.set_xy(start_x, start_y + (height/2) - 10)
        self.set_font(FONT_FAMILY, "B", 14)
        self.set_text_color(*COLOR_SUBTITLE)
        self.cell(width, 10, "[ UI SCREENSHOT PLACEHOLDER ]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_font(FONT_FAMILY, "I", 11)
        self.set_text_color(100, 100, 100)
        self.cell(width, 10, "Insert latest Glassmorphism UI screenshot showing Pathogenicity prediction and SHAP charts here.", align="C")
        
        self.set_xy(10, start_y + height + 10)

def generate_pdf():
    pdf = SteppeDNAPDF()
    pdf.set_margins(10, 15, 10)
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, "", 12)
    pdf.set_text_color(*COLOR_TEXT)

    # SEC 1: Project Scope
    pdf.add_section_header("1. Core Project Scope & Scale")
    pdf.multi_cell(0, 6, "SteppeDNA predicts whether missense mutations in 5 core HR-pathway genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D) are pathogenic or benign. Doctors use this to interpret 'Variants of Uncertain Significance' (VUS) instantly to prevent cancer, rather than waiting months for lab assays.")
    pdf.add_sub_header("Project Dimensions")
    pdf.add_bullet("Genes Analyzed", "5 total (BRCA1, BRCA2, PALB2, RAD51C, RAD51D).")
    pdf.add_bullet("Training Data", "19,223 variants (18,738 ClinVar + 485 gnomAD proxy-benign).")
    pdf.add_bullet("Total Features", "103 engineered biochemical, structural, and evolutionary features per variant.")
    pdf.add_bullet("Architecture", "XGBoost (60%) + MLP (40%) ensemble with isotonic calibration.")

    # SEC 2: The True Performance Metrics
    pdf.add_section_header("2. Model Evaluation Evolution & Validation Metrics")
    pdf.multi_cell(0, 6, "Machine Learning in biology is prone to extreme data leakage if evaluated incorrectly. SteppeDNA's evaluation explicitly prevents this.")
    
    pdf.add_sub_header("Early Versions (v1-v3)")
    pdf.multi_cell(0, 6, "Early versions focused solely on BRCA2 with MAVE features (~0.99 AUC, but data leakage). Removing MAVE dropped to 0.77 AUC on holdout. The v3 pan-gene model on 759 variants achieved 0.736 AUC. These iterations informed the architecture but are superseded.")

    pdf.add_sub_header("The Present: v5.0 Universal Pan-Gene Model (0.978 AUC)")
    pdf.multi_cell(0, 6, "SteppeDNA v5.0 combines all 5 genes into a 19,223-variant dataset with gnomAD proxy-benign augmentation. Evaluated through rigorous 60/20/20 train/calibration/test split with gene x label stratification:")
    pdf.add_bullet("ROC-AUC", "0.978")
    pdf.add_bullet("MCC", "0.881")
    pdf.add_bullet("Balanced Accuracy", "94.1%")
    pdf.add_bullet("10-Fold CV", "0.9797 +/- 0.0031 (95% CI: 0.9777-0.9814)")
    pdf.add_bullet("SOTA Comparison", "SteppeDNA 0.978 > REVEL 0.725 > BayesDel 0.721 > CADD 0.539")
    pdf.multi_cell(0, 6, "This 0.978 AUC reflects the true, un-leaked capacity of the model validated on independent test data and gold-standard benchmarks (ProteinGym DMS + ClinVar Expert Panel).")

    pdf.add_page()
    
    # SEC 3: Data Leakage & Overfitting Prevention
    pdf.add_section_header("3. Overfitting & Data Leakage Prevention Mechanisms")
    pdf.multi_cell(0, 6, "To achieve the 0.978 AUC with no data leakage, several strict software architectures were implemented:")
    pdf.add_bullet("SMOTE Strict Isolation", "Because benign variants vastly outnumber pathogenic ones, we use SMOTE to generate synthetic pathogenic data. Crucially, SMOTE is ONLY applied to the training split. Applying it before splitting causes massive data leakage where the test set contains interpolations of the training data.")
    pdf.add_bullet("Stratified Split", "60/20/20 train/calibration/test split with gene x label stratification. 10-fold CV confirms stability (0.9797 +/- 0.0031).")
    pdf.add_bullet("Sparse Feature Pruning", "One-hot encoded amino acid strings (e.g. 'Mutation_A>C') are actively dropped during true blind tests because they act as unique identifiers that tree models use to memorize specific variants.")

    # SEC 4: External Models & Datasets
    pdf.add_section_header("4. External Datasets & Tool Integrations")
    pdf.add_bullet("ClinVar", "The NIH's ground-truth database. Provides Pathogenic/Benign labels for 18,738 of the 19,223 training variants.")
    pdf.add_bullet("AlphaMissense", "Google DeepMind's specialized deep learning predictions for mutations.")
    pdf.add_bullet("ESM-2 (Meta AI)", "esm2_t6_8M_UR50D protein language model. Per-gene embeddings compressed to 20 PCA components, capturing hidden evolutionary 'grammar'.")
    pdf.add_bullet("PhyloP (100-way)", "Measures evolutionary conservation across 100 vertebrate species over 400M years.")
    pdf.add_bullet("gnomAD", "The Genome Aggregation Database v4 (~1.6M alleles). Provides 485 proxy-benign variants (AC >= 2) and allele frequency features.")
    pdf.add_bullet("SpliceAI", "Illumina's Deep neural network detecting if a mutation disrupts an RNA splicing site.")
    pdf.add_bullet("MAVE", "Multiplexed Assays of Variant Effect. Ground-truth wet-lab data.")

    pdf.add_page()

    # SEC 5: Feature Engineering Breakdown
    pdf.add_section_header("5. Feature Engineering Pipeline (103 Features)")
    pdf.multi_cell(0, 6, "Every variant is converted into a 103-dimensional numerical tensor before model inference (gene-identifying features removed to prevent shortcut learning):")
    pdf.add_sub_header("Physicochemical Physics (Size, Charge, Water affinity)")
    pdf.add_bullet("Volume Diff", "Volumetric shift. Replacing tiny Glycine with massive Tryptophan creates localized 3D stress.")
    pdf.add_bullet("Hydrophobicity", "Tracks the oily/water-loving nature. Flipping these incorrectly exposes core residues to water.")
    pdf.add_bullet("Charge Change", "Loss of +/- charge breaks holding salt bridges.")
    
    pdf.add_sub_header("Substitution & Evolution")
    pdf.add_bullet("BLOSUM62", "Empirical statistical likelihood of nature accepting this specific amino acid swap.")
    
    pdf.add_sub_header("3D Structural Geometry")
    pdf.add_bullet("RSA & is_buried", "Relative Solvent Accessibility. Is the amino acid buried in the core or floating on the surface?")
    pdf.add_bullet("B-factor", "The thermal flexibility and spatial confidence of the residue.")
    pdf.add_bullet("Coordinates", "Distance (A) to the DNA-binding surface and PALB2 interfaces.")
    pdf.add_bullet("Secondary Structure", "Boolean flags for alpha-helices vs beta-sheets.")
    
    pdf.add_sub_header("Cross-Interactions")
    pdf.add_bullet("Buried_x_BLOSUM", "Severe penalization if an unnatural swap occurs deep inside the core.")
    pdf.add_bullet("Conserv_x_BLOSUM", "Severe penalization if a conserved residue receives an abhorrent mutation.")
    
    pdf.add_sub_header("Principal Components (PCA)")
    pdf.add_bullet("ESM-2 PCAs (0-19)", "The top 20 compressed components of Meta's protein language neural network.")

    # SEC 6: The AI Models
    pdf.add_section_header("6. The Heterogeneous Ensemble Model")
    pdf.multi_cell(0, 6, "To maximize stability and prevent single-algorithm failure, SteppeDNA uses a Stacked Ensemble:")
    pdf.add_bullet("XGBoost (60% weight)", "Gradient-boosted decision trees. Dominates tabular data and handles missing values natively.")
    pdf.add_bullet("MLP (40% weight)", "Multi-Layer Perceptron neural network. Captures non-linear relationships within ESM-2 embeddings and cross-feature interactions.")
    pdf.add_bullet("Isotonic Calibrator", "The weighted ensemble outputs are passed through an Isotonic Calibrator trained on the calibration split. Forces scores into an interpretable Clinical Risk Probability curve (0 to 1).")

    # SEC 7: ACMG Rules & Backend
    pdf.add_section_header("7. ACMG Evidence Guidelines & Backend Logic")
    pdf.add_bullet("PM1 (Moderate Pathogenic)", "System checks 3D distances. If variant is < 5 Angstroms from DNA/PALB2 interfaces, or inside a BRC repeat, it triggers PM1.")
    pdf.add_bullet("BS1 (Strong Benign)", "If the gnomAD frequency exceeds 5% (>0.05).")
    pdf.add_bullet("PP3 / BP4 (Supporting)", "Triggered mathematically if the ensemble probability > 0.90 or < 0.10.")
    
    pdf.add_sub_header("VCF Parsing (Reverse Strand Biology)")
    pdf.multi_cell(0, 6, "Genes like BRCA2 are transcribed backwards on the DNA Minus Strand. SteppeDNA handles active reverse-complement arithmetic in <400ms. It maps genomic coordinates to cDNA, flips nucleotides (A > G becomes T > C), and recalculates the amino acid translation.")

    # UI Placeholder
    pdf.draw_placeholder_box()

    output_path = r"c:\Users\User\OneDrive\Desktop\Project explanation\v2_SteppeDNA_Deep_Dive.pdf"
    pdf.output(output_path)
    print(f"Success! Elegant B&W PDF generated at {output_path}")

if __name__ == "__main__":
    generate_pdf()
