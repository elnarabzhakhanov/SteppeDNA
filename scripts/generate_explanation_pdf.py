import markdown
import pdfkit
import os

# Define Paths
MD_PATH = "explanation_of_SteppeDNA.md"
PDF_PATH = "explanation_of_SteppeDNA.pdf"

# Read markdown content
with open(MD_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to basic HTML with extensions
html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# Wrap in CSS styling to make it look professional
html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        margin: 40px auto;
        max-width: 900px;
        padding: 0 40px;
    }}
    h1 {{
        color: #6a1b9a;
        border-bottom: 2px solid #6a1b9a;
        padding-bottom: 10px;
        font-size: 2.2em;
    }}
    h2 {{
        color: #8e24aa;
        margin-top: 30px;
        font-size: 1.6em;
    }}
    h3 {{
        color: #ab47bc;
        font-size: 1.3em;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }}
    th {{
        background-color: #f3e5f5;
        color: #4a148c;
    }}
    code {{
        background-color: #f4f4f4;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: Consolas, monospace;
        font-size: 0.9em;
    }}
    .placeholder-box {{
        width: 100%;
        height: 350px;
        border: 3px dashed #ce93d8;
        background-color: #fdfaf6;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #8e24aa;
        font-weight: bold;
        font-size: 1.5em;
        margin-top: 20px;
        border-radius: 10px;
        text-align: center;
    }}
    blockquote {{
        border-left: 5px solid #ce93d8;
        padding-left: 15px;
        color: #666;
        font-style: italic;
        background: #fdfaf6;
        margin: 20px 0;
        padding: 10px 20px;
    }}
</style>
</head>
<body>
{html_body.replace('<img alt="SteppeDNA System Interface Placeholder" src="placeholder.jpg" />', '<div class="placeholder-box"> [UI IMAGE PLACEHOLDER] <br> Insert your current real SteppeDNA App Screenshot Here </div>')}
</body>
</html>
"""

# Configure pdfkit
options = {
    'page-size': 'A4',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': "UTF-8",
    'enable-local-file-access': None
}

try:
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    pdfkit.from_string(html, PDF_PATH, options=options, configuration=config)
    print(f"Success! Scaled PDF exported to {PDF_PATH}")
except Exception as e:
    print(f"Error during PDF generation: {e}")
    # Fallback to standard installation if the specific path fails
    try:
        print("Attempting generation without explicit path...")
        pdfkit.from_string(html, PDF_PATH, options=options)
        print(f"Success! Scaled PDF exported to {PDF_PATH}")
    except Exception as e2:
        print(f"Strict fallback failed: {e2}")
