from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# --- MINIMALIST UNIFORM STYLE CONFIG ---
COLOR_PRIMARY = (30, 30, 30)      
COLOR_TEXT = (50, 50, 50)      
FONT_FAMILY = "times"          
BASE_FONT_SIZE = 11.0

class SteppeDNAPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            # Main title - only slightly larger
            self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE + 3)
            self.set_text_color(*COLOR_PRIMARY)
            self.cell(0, 10, "SteppeDNA: Complete Project Explanation", 
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            self.set_draw_color(30, 30, 30)
            self.set_line_width(0.3)
            self.line(self.get_x(), self.get_y(), self.get_x()+190, self.get_y())
            self.ln(4)
            
            # Subtitle
            self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
            self.set_text_color(*COLOR_TEXT)
            self.multi_cell(0, 6, "A beginner-friendly guide to how SteppeDNA uses Artificial Intelligence to predict cancer risks from genetic mutations. This document explains the exact dataset sizes, project scale, how the AI makes decisions, and how we ensure it doesn't 'cheat' on its tests.")
            self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE - 2)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"SteppeDNA Project Explanation | Page {self.page_no()}", align="C")

    def add_section_header(self, text):
        self.ln(4)
        self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE + 1)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def add_sub_header(self, text):
        self.ln(2)
        self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*COLOR_TEXT)

    def add_bullet(self, label, text):
        self.set_left_margin(15)
        self.set_text_color(*COLOR_PRIMARY)
        
        if label:
            self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE)
            self.write(6, f"- {label}: ")
            self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
            self.set_text_color(*COLOR_TEXT)
            self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
            self.set_text_color(*COLOR_TEXT)
            self.multi_cell(0, 6, f"- {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
        self.set_left_margin(10)
        
    def draw_placeholder_box(self):
        self.ln(6)
        start_x = self.get_x() + 10
        start_y = self.get_y()
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.2)
        
        width = 170
        height = 60
        
        self.rect(start_x, start_y, width, height)
        
        self.set_xy(start_x, start_y + (height/2) - 6)
        self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
        self.set_text_color(120, 120, 120)
        self.cell(width, 12, "[ Insert latest SteppeDNA app interface screenshot here ]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_xy(10, start_y + height + 6)

def generate_pdf():
    pdf = SteppeDNAPDF()
    pdf.set_margins(12, 15, 12)
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
    pdf.set_text_color(*COLOR_TEXT)

    # SEC 1
    pdf.add_section_header("1. What Does SteppeDNA Do?")
    pdf.multi_cell(0, 6, "Think of human DNA as a massive instruction manual. Sometimes, there is a 'typo' (a mutation) in the manual. SteppeDNA looks at 5 specific 'chapters' (genes) of this manual that deal with repairing DNA: BRCA1, BRCA2, PALB2, RAD51C, and RAD51D. \n\nWhen a doctor finds a typo in a patient, they often don't know if it's 'Pathogenic' (dangerous, increasing cancer risk) or 'Benign' (a harmless typo). Laboratory tests to find out take months. SteppeDNA uses Artificial Intelligence to instantly read the typo and predict if it is dangerous.")
    
    pdf.add_sub_header("Project Size & Scale")
    pdf.add_bullet("Total Files", "154 files containing the data, website code, and AI logic.")
    pdf.add_bullet("Lines of Code", "13,097 lines of programming (mostly Python logic).")
    pdf.add_bullet("Total Genes", "5 core breast and ovarian cancer genes.")
    pdf.add_bullet("The Clues (Features)", "128 different mathematical and biological clues are gathered for every single mutation to help the AI make its decision.")

    # SEC 2
    pdf.add_section_header("2. Honest Grading: How Good is the AI?")
    pdf.multi_cell(0, 6, "Just like a student taking a test, an AI can sometimes 'cheat' if you aren't careful. Over time, we made the tests harder to see how 'smart' SteppeDNA truly was.")
    
    pdf.add_sub_header("The 99% Era (The Cheat Code)")
    pdf.multi_cell(0, 6, "Initially, the AI scored a near-perfect 99% accuracy on a single gene (BRCA2). However, we realized we were giving the AI a physical lab-test result (called a MAVE score) as one of its clues. The AI wasn't learning biology; it was just reading the doctor's answer key!")
    
    pdf.add_sub_header("The 73% Era (The Real-World Universal Test)")
    pdf.multi_cell(0, 6, "We removed the cheat code, and forced the AI to learn all 5 genes at once (759 different mutations). In the real world, the AI will face mutations no human has ever seen before, so we tested it strictly on 'unseen' data. \n\nThe true, honest grade is 73.6% accuracy (specifically, a metric called 'ROC-AUC'). In the medical AI world, predicting complex cancer biology across 5 completely different genes with 73.6% unassisted accuracy is a very strong, realistic achievement that shows the AI has genuinely learned the natural rules of life.")

    # SEC 3
    pdf.add_section_header("3. How We Prevent 'Memorization' (Overfitting)")
    pdf.multi_cell(0, 6, "AI often tries to memorize the training textbook instead of learning the underlying concepts. We built software locks to stop this (preventing 'Data Leakage'):")
    pdf.add_bullet("Creating Practice Questions (SMOTE)", "Dangerous mutations are rare. To help the AI practice, we generate 'synthetic' fake dangerous mutations. But crucially, we ONLY do this during 'study time' (training). If we left fake examples in the final test, the AI would get an artificially high grade.")
    pdf.add_bullet("Hiding the Test", "We test the AI 5 separate times using a method called 'Cross-Validation'. In every test, the AI is completely blocked from seeing the test questions while it studies.")
    pdf.add_bullet("Removing Names", "We hide the explicit names of the mutations from the AI, so it can't memorize 'Mutation X is bad'. It has to look at the chemistry.")

    # SEC 4
    pdf.add_section_header("4. The 128 Clues: How the AI Decides")
    pdf.multi_cell(0, 6, "To make a prediction, SteppeDNA gathers 128 unique clues about the typo. Here are some of the most important ones, explained simply:")
    
    pdf.add_sub_header("Evolutionary Clues (Has Nature Changed This?)")
    pdf.add_bullet("PhyloP (The 400-Million-Year Test)", "We check the DNA of 100 animals (from fish to monkeys). If a piece of DNA hasn't changed in 400 million years, it means Nature thinks it's perfectly tuned. If the human mutation changes that exact spot, it is almost certainly dangerous.")
    
    pdf.add_sub_header("Physical Clues (Does It Break the Machine?)")
    pdf.add_bullet("Size and Charge", "Proteins are 3D puzzle pieces. If you replace a tiny, negatively-charged puzzle piece with a massive, positively-charged one, the whole structure might break apart.")
    pdf.add_bullet("Is it hidden?", "If the typo happens deep inside the core of the protein where water shouldn't get in, it causes worse damage than on the surface.")
    pdf.add_bullet("Distance to the Engine", "Proteins bind to DNA to fix it. We calculate how physically close the typo is to the DNA grabbing 'hands'.")

    pdf.add_page()
    
    # SEC 5
    pdf.add_section_header("5. Where We Get Our Data")
    pdf.multi_cell(0, 6, "SteppeDNA automatically connects to massive global databases to collect its clues:")
    pdf.add_bullet("ClinVar", "The global database of known cancer mutations. This acts as our 'Answer Key' for training.")
    pdf.add_bullet("ESM-2 (Meta AI)", "An Artificial Intelligence built by Meta (Facebook) that acts like a dictionary translator for proteins. It gives us 64 clues about the 'grammar' of the mutation.")
    pdf.add_bullet("gnomAD", "A database of 800,000 healthy people. If a mutation is very common in healthy people, SteppeDNA uses this to confidently safely assume it cannot be causing severe cancer.")
    pdf.add_bullet("AlphaMissense", "Google DeepMind's own AI predictions. We ask Google for a second opinion and feed it into our system.")

    # SEC 6
    pdf.add_section_header("6. The 'Committee' of AI Models")
    pdf.multi_cell(0, 6, "SteppeDNA doesn't just ask one AI. It asks a committee of two different types of AI, and averages their votes to be safe:")
    pdf.add_bullet("The Decision Tree (XGBoost)", "This AI is really good at looking at solid, hard facts (like 'Is this common in healthy people?'). It acts as the logical thinker.")
    pdf.add_bullet("The Neural Network (Deep Learning)", "This AI acts more like a human brain. It looks at the 64 'grammar' clues from Meta's AI to find hidden, invisible patterns that simple math can't see.")
    pdf.add_bullet("The Calibrator", "A mathematical filter that takes the raw 'yes/no' feelings from the AIs and converts it into a realistic, trustworthy percentage from 0 to 100%.")

    # SEC 7
    pdf.add_section_header("7. Medical Rules and Reverse Translation")
    pdf.multi_cell(0, 6, "A cool feature of SteppeDNA is that it respects both doctors and biology:")
    pdf.add_bullet("Standard Doctor Rules (ACMG)", "Doctors have standard checklists for grading mutations. SteppeDNA's backend automatically fills out these checklists (like noting if the mutation is too close to the DNA-binding region).")
    pdf.add_bullet("Reverse Biology (The Minus Strand)", "Some genes (like BRCA2) are written 'backwards' on the DNA double-helix. The SteppeDNA software automatically calculates this backwards math in milliseconds, flipping the DNA letters appropriately so the AI gets the right sequence.")

    # UI Placeholder
    pdf.draw_placeholder_box()

    output_path = r"c:\Users\User\OneDrive\Desktop\Project explanation\v2_SteppeDNA_Deep_Dive.pdf"
    pdf.output(output_path)
    print(f"Success! Minimalist Uniform B&W Layman PDF generated at {output_path}")

if __name__ == "__main__":
    generate_pdf()
