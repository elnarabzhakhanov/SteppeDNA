from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# --- STYLE CONFIG ---
COLOR_PURPLE = (99, 102, 241)  # Indigo/Purple accent
COLOR_TEXT = (31, 41, 55)      # Dark Gray text
COLOR_SUBTITLE = (107, 114, 128) # Muted Gray

class SteppeDNAPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font("helvetica", "B", 24)
            self.set_text_color(*COLOR_PURPLE)
            self.cell(0, 20, "SteppeDNA v3 & v4 - Comprehensive Deep Dive", 
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            self.set_draw_color(*COLOR_PURPLE)
            self.set_line_width(0.8)
            self.line(self.get_x(), self.get_y()-5, self.get_x()+190, self.get_y()-5)
            
            self.set_font("helvetica", "I", 11)
            self.set_text_color(*COLOR_SUBTITLE)
            self.multi_cell(0, 10, "A complete, granular breakdown of the true architecture, performance metrics (0.73 ROC-AUC), exact 128 feature set, external datasets, and data leakage prevention mechanisms across the 154-file project codebase.")
            self.ln(5)
            self.set_draw_color(229, 231, 235) # Light gray line
            self.set_line_width(0.2)
            self.line(self.get_x(), self.get_y(), self.get_x()+190, self.get_y())
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"SteppeDNA Comprehensive Deep Dive | Validation Metrics | Page {self.page_no()}", align="C")

    def add_section_header(self, text):
        self.ln(8)
        self.set_font("helvetica", "B", 18)
        self.set_text_color(*COLOR_PURPLE)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*COLOR_PURPLE)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y() - 1, self.get_x() + 190, self.get_y() - 1)
        self.ln(4)

    def add_sub_header(self, text):
        self.ln(4)
        self.set_font("helvetica", "B", 13)
        self.set_text_color(*COLOR_PURPLE)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*COLOR_TEXT)
        self.ln(1)

    def add_bullet(self, label, text):
        self.ln(1)
        processed_text = f"- {label}: {text}" if label else f"- {text}"
        self.set_left_margin(15)
        self.set_font("helvetica", "", 10)
        self.set_text_color(*COLOR_TEXT)
        self.multi_cell(0, 6, processed_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_left_margin(10) # Reset to default margin
        self.ln(1)
        
    def draw_placeholder_box(self):
        self.ln(10)
        start_x = self.get_x() + 10
        start_y = self.get_y()
        self.set_draw_color(*COLOR_PURPLE)
        self.set_line_width(0.5)
        
        # Draw dashed rectangle manually
        width = 170
        height = 90
        
        # Simple solid box (fpdf doesn't support dashed rects easily)
        self.rect(start_x, start_y, width, height)
        
        self.set_xy(start_x, start_y + (height/2) - 10)
        self.set_font("helvetica", "B", 14)
        self.set_text_color(*COLOR_PURPLE)
        self.cell(width, 10, "[ UI SCREENSHOT PLACEHOLDER ]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_font("helvetica", "I", 10)
        self.set_text_color(150, 150, 150)
        self.cell(width, 10, "Insert latest Glassmorphism UI screenshot showing Pathogenicity prediction and SHAP charts here.", align="C")
        
        self.set_xy(10, start_y + height + 10)

def generate_pdf():
    pdf = SteppeDNAPDF()
    pdf.set_margins(10, 15, 10)
    pdf.add_page()
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(*COLOR_TEXT)

    # SEC 1: Project Scope
    pdf.add_section_header("1. Core Project Scope & Scale")
    pdf.multi_cell(0, 6, "SteppeDNA predicts whether missense mutations in 5 core HR-pathway genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D) are pathogenic or benign. Doctors use this to interpret 'Variants of Uncertain Significance' (VUS) instantly to prevent cancer, rather than waiting months for lab assays.")
    pdf.add_sub_header("Project Dimensions")
    pdf.add_bullet("Total Files", "154 system files spanning pipelines, backend, frontend, and tests.")
    pdf.add_bullet("Codebase Size", "Approximately 13,097 lines of code (majority Python & FastAPI, alongside JS/CSS).")
    pdf.add_bullet("Genes Analyzed", "5 total (BRCA1, BRCA2, PALB2, RAD51C, RAD51D).")
    pdf.add_bullet("Total Features", "128 dynamic biochemical, structural, and evolutionary features per variant.")

    # SEC 2: The True Performance Metrics
    pdf.add_section_header("2. Model Evaluation Evolution & True Validation Metrics")
    pdf.multi_cell(0, 6, "Machine Learning in biology is prone to extreme data leakage if evaluated incorrectly. SteppeDNA's evaluation explicitly prevents this.")
    
    pdf.add_sub_header("The 0.99 AUC Era (The 'Cheat' Model)")
    pdf.multi_cell(0, 6, "Early versions focused solely on BRCA2 and included MAVE validation scores as input features. The ROC-AUC was ~0.99. However, because MAVE is a near-perfect proxy for pathogenicity, the model was simply learning to read the lab result rather than learning the biology. This was an overfit scenario.")
    
    pdf.add_sub_header("The 0.91 -> 0.77 AUC Era (The 'MAVE-Blind' Model)")
    pdf.multi_cell(0, 6, "We explicitly dropped MAVE features to test the model's true predictive power on an independent MAVE holdout set. While standard splits showed 0.91 AUC, the strict independent holdout score dropped to 0.77. The model struggled with Recall (identifying novel pathogenic variants) without the MAVE 'cheat code'.")
    
    pdf.add_sub_header("The Present: v3 Universal Pan-Gene Model (0.736 AUC)")
    pdf.multi_cell(0, 6, "To build a truly generalizable system, SteppeDNA v3 combined all 5 genes into a 759-variant dataset without relying on single-gene lab shortcuts. Evaluated through a rigorous, strictly-isolated 5-Fold Cross Validation:")
    pdf.add_bullet("True ROC-AUC", "0.7363 (± 0.019)")
    pdf.add_bullet("True Accuracy", "0.7194 (± 0.021)")
    pdf.add_bullet("True F1 Score", "0.8164 (± 0.015)")
    pdf.multi_cell(0, 6, "This 0.73 AUC is the real, un-leaked capacity of the machine to predict complex disease biology across entirely different protein domains. It is a highly capable tool built on generalized biology, not overfit shortcuts.")

    pdf.add_page()
    
    # SEC 3: Data Leakage & Overfitting Prevention
    pdf.add_section_header("3. Overfitting & Data Leakage Prevention Mechanisms")
    pdf.multi_cell(0, 6, "To achieve the 0.736 true score, several strict software architectures were implemented:")
    pdf.add_bullet("SMOTE Strict Isolation", "Because benign variants vastly outnumber pathogenic ones, we use SMOTE to generate synthetic pathogenic data. Crucially, SMOTE is ONLY applied to the training split. Applying it before splitting causes massive data leakage where the test set contains interpolations of the training data.")
    pdf.add_bullet("Out-of-Fold (OOF) CV", "Features and early stopping are evaluated using a 5-Fold Stratified K-Fold setup. The 'test' score is strictly on variants the model has never seen.")
    pdf.add_bullet("Sparse Feature Pruning", "All one-hot encoded amino acid strings (e.g. 'Mutation_A>C') are actively dropped during true blind tests because they act as unique identifiers that tree models use to memorize specific variants.")

    # SEC 4: External Models & Datasets
    pdf.add_section_header("4. External Datasets & Tool Integrations")
    pdf.add_bullet("ClinVar", "The NIH's ground-truth database. Used strictly to provide the 1/0 (Pathogenic/Benign) labels for the 759 training vectors.")
    pdf.add_bullet("AlphaMissense", "Google DeepMind's specialized deep learning predictions for mutations. Incorporated as a highly-weighted consensus feature.")
    pdf.add_bullet("ESM-2 (Meta AI)", "An 8-Million parameter Protein LLM. The model converts raw amino acid sequences into a 64-dimensional latent embedding space, capturing hidden evolutionary 'grammar'.")
    pdf.add_bullet("PhyloP (100-way)", "Measures evolutionary conservation across 100 vertebrate species over 400M years. Immutable locations imply critical biological functions.")
    pdf.add_bullet("gnomAD", "The Genome Aggregation Database (800k+ individuals). High allele frequency indicates natural population survival, heavily ruling out high-penetrance pathogenicity.")
    pdf.add_bullet("SpliceAI (Illumina)", "Detects if a mutation accidentally destroys an RNA splicing site (preventing the protein from even forming) rather than just altering the 3D shape.")
    pdf.add_bullet("MAVE", "Multiplexed Assays of Variant Effect. Ground-truth wet-lab data. Used carefully in v3 to guide baseline predictions without overfitting.")

    pdf.add_page()

    # SEC 5: Feature Engineering Breakdown
    pdf.add_section_header("5. Feature Engineering Pipeline (All 128 Features)")
    pdf.multi_cell(0, 6, "Every variant is converted into a 128-dimensional numerical tensor before model inference:")
    pdf.add_sub_header("Physicochemical Physics (Size, Charge, Water affinity)")
    pdf.add_bullet("Volume Diff", "Volumetric shift. Replacing tiny Glycine with massive Tryptophan creates localized 3D stress.")
    pdf.add_bullet("Hydrophobicity (Ref, Alt, Diff)", "Tracks the oily/water-loving nature. Flipping these incorrectly exposes core residues to water, unfolding the protein.")
    pdf.add_bullet("Charge Change", "Loss of +/- charge breaks holding salt bridges.")
    
    pdf.add_sub_header("Substitution & Evolution")
    pdf.add_bullet("BLOSUM62", "Empirical statistical likelihood of nature accepting this specific amino acid swap.")
    
    pdf.add_sub_header("3D Structural Geometry")
    pdf.add_bullet("RSA & is_buried", "Relative Solvent Accessibility. Is the amino acid buried in the core or floating on the surface?")
    pdf.add_bullet("B-factor", "The thermal flexibility and spatial confidence of the residue.")
    pdf.add_bullet("Coordinates", "Distance (Angstroms) to the DNA-binding surface and PALB2-interaction interfaces.")
    pdf.add_bullet("Secondary Structure", "Boolean flags for alpha-helices vs beta-sheets.")
    
    pdf.add_sub_header("Cross-Interactions")
    pdf.add_bullet("Buried_x_BLOSUM", "Severe penalization if an unnatural swap occurs deep inside the core.")
    pdf.add_bullet("Conserv_x_BLOSUM", "Severe penalization if a 400-million-year conserved residue receives an abhorrent mutation.")
    
    pdf.add_sub_header("Principal Components (PCA)")
    pdf.add_bullet("ESM-2 PCAs (0 through 19)", "The top 20 compressed components of Meta's protein language neural network.")

    # SEC 6: The AI Models
    pdf.add_section_header("6. The Heterogeneous Ensemble Model")
    pdf.multi_cell(0, 6, "To maximize stability and prevent single-algorithm failure, SteppeDNA uses a Stacked Ensemble:")
    pdf.add_bullet("XGBoost (Booster)", "A forest of 400 deep decision trees. XGBoost dominates tabular data and easily handles missing values (e.g. when SpliceAI or gnomAD data doesn't exist for a variant).")
    pdf.add_bullet("MLP (Neural Network)", "A TensorFlow Multi-Layer Perceptron (128>64>32). Neural Nets are mathematically superior at resolving non-linear relationships within the continuous ESM-2 embeddings.")
    pdf.add_bullet("Logistic Stacker & Isotonic Calibrator", "The raw outputs of XGBoost and the NN are blended (60/40), and passed through an Isotonic Calibrator. The calibrator forces the raw ML scores into a true, interpretable Clinical Risk Probability curve (0 to 1).")

    pdf.add_page()

    # SEC 7: ACMG Rules & Backend
    pdf.add_section_header("7. ACMG Evidence Guidelines & Backend Logic")
    pdf.multi_cell(0, 6, "SteppeDNA automatically evaluates American College of Medical Genetics clinical rules:")
    pdf.add_bullet("PM1 (Moderate Pathogenic)", "System checks 3D distances. If variant is < 5 Angstroms from DNA or PALB2 interfaces, or inside a BRC repeat, it triggers PM1.")
    pdf.add_bullet("BS1 (Strong Benign)", "If the gnomAD frequency exceeds 5% (>0.05).")
    pdf.add_bullet("PP3 / BP4 (Supporting)", "Triggered mathematically if the ensemble probability > 0.90 or < 0.10.")
    
    pdf.add_sub_header("VCF Parsing (Reverse Strand Biology)")
    pdf.multi_cell(0, 6, "Genes like BRCA2 are transcribed backwards on the DNA Minus Strand. SteppeDNA handles active reverse-complement arithmetic in <400ms. It maps genomic coordinates to cDNA, flips nucleotides (A > G becomes T > C), and recalculates the amino acid translation.")

    # UI Placeholder
    pdf.draw_placeholder_box()

    output_path = r"c:\Users\User\OneDrive\Desktop\Project explanation\v2_SteppeDNA_Deep_Dive_Updated.pdf"
    pdf.output(output_path)
    print(f"Success! Highly detailed PDF generated at {output_path}")

if __name__ == "__main__":
    generate_pdf()
