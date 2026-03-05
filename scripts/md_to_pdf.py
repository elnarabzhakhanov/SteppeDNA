from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

class DeepDivePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font('helvetica', 'B', 15)
            self.cell(0, 10, 'SteppeDNA v3 - Universal AI Deep Dive', 
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def clean_text(text):
    # Replace common non-latin1 characters with ASCII approximations
    replacements = {
        '\u2014': '-', # em dash
        '\u2013': '-', # en dash
        '\u2022': '*', # bullet
        '\u2018': "'", # left single quote
        '\u2019': "'", # right single quote
        '\u201c': '"', # left double quote
        '\u201d': '"', # right double quote
        '\xb5': 'u',    # micro sign
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def convert_md_to_pdf(md_path, pdf_path):
    pdf = DeepDivePDF()
    pdf.add_page()
    pdf.set_font('helvetica', size=11)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = clean_text(line.strip())
        
        if not line:
            pdf.ln(5)
            continue
            
        if line.startswith('# '):
            pdf.set_font('helvetica', 'B', 18)
            pdf.cell(0, 12, line[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 11)
            pdf.ln(5)
        elif line.startswith('## '):
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, line[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 11)
            pdf.ln(2)
        elif line.startswith('### '):
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 8, line[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 11)
        elif line.startswith('---'):
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(5)
        elif line.startswith('* ') or line.startswith('- '):
            pdf.cell(5)
            pdf.cell(5, 6, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
            text = line[2:].replace('**', '')
            pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            text = line.replace('**', '')
            pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(pdf_path)

if __name__ == "__main__":
    md_file = r"c:\Users\User\OneDrive\Desktop\Project explanation\v3_SteppeDNA_Deep_Dive.md"
    pdf_file = r"c:\Users\User\OneDrive\Desktop\Project explanation\v3_SteppeDNA_Deep_Dive.pdf"
    
    print(f"Converting {md_file} to PDF...")
    try:
        convert_md_to_pdf(md_file, pdf_file)
        print(f"Success! PDF saved to {pdf_file}")
    except Exception as e:
        print(f"Error: {e}")
