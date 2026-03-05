import markdown
import os
import webbrowser

MD_PATH = "explanation_of_SteppeDNA.md"
HTML_PATH = "explanation_of_SteppeDNA.html"

# Read markdown content
with open(MD_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to basic HTML with extensions
html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# Wrap in CSS styling to make it look professional both on screen and in print
html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Explanation of SteppeDNA</title>
<style>
    @page {{ size: A4; margin: 2cm; }}
    body {{
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1c1c1c;
        margin: 40px auto;
        max-width: 850px;
        padding: 0 40px;
    }}
    h1 {{
        color: #7B1FA2;
        border-bottom: 2px solid #E1BEE7;
        padding-bottom: 10px;
        font-size: 2.2em;
        margin-top: 0;
    }}
    h2 {{
        color: #9C27B0;
        margin-top: 30px;
        font-size: 1.6em;
        border-bottom: 1px solid #f3e5f5;
        padding-bottom: 5px;
    }}
    h3 {{
        color: #BA68C8;
        font-size: 1.3em;
        margin-bottom: 10px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 0.95em;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }}
    th {{
        background-color: #F8BBD0;
        color: #4A148C;
    }}
    tr:nth-child(even) {{
        background-color: #fafafa;
    }}
    code {{
        background-color: #f1f3f5;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: Consolas, 'Courier New', monospace;
        font-size: 0.9em;
        color: #C2185B;
    }}
    .placeholder-box {{
        width: 100%;
        height: 350px;
        border: 3px dashed #ce93d8;
        background-color: #fdfaf6;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #8e24aa;
        font-weight: bold;
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        text-align: center;
    }}
    .placeholder-box p {{
        font-size: 0.6em;
        font-weight: normal;
        color: #777;
    }}
    blockquote {{
        border-left: 5px solid #ce93d8;
        padding-left: 15px;
        color: #555;
        font-style: italic;
        background: #fdfaf6;
        margin: 20px 0;
        padding: 10px 20px;
        border-radius: 0 8px 8px 0;
    }}
    @media print {{
        .placeholder-box {{ border: 3px dashed #999; color: #555; background-color: #fff; }}
        body {{ max-width: 100%; padding: 0; margin: 0; }}
    }}
</style>
</head>
<body>
{html_body.replace('<img alt="SteppeDNA System Interface Placeholder" src="placeholder.jpg" />', '<div class="placeholder-box"><span>[UI GRAPHIC PLACEHOLDER]</span><p>Insert your current real SteppeDNA App Screenshot Here</p></div>')}
<div style="text-align: center; margin-top: 50px; padding: 20px; color: white; background: #9C27B0; border-radius: 8px; cursor: pointer; display: inline-block; font-weight: bold;" onclick="window.print()" id="printbtn">
    Click Here to Save as PDF
</div>
<script>
    window.onbeforeprint = function() {{ document.getElementById('printbtn').style.display = 'none'; }}
    window.onafterprint = function() {{ document.getElementById('printbtn').style.display = 'inline-block'; }}
</script>
</body>
</html>
"""

with open(HTML_PATH, "w", encoding='utf-8') as f:
    f.write(html)

print(f"Generated clean HTML representation at: {os.path.abspath(HTML_PATH)}")
print("Attempting to open in default browser...")
try:
    webbrowser.open(f"file://{os.path.abspath(HTML_PATH)}")
except:
    pass
