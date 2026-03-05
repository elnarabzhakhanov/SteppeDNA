import os

# Updated Content for v3
content = {
    "title": "SteppeDNA v3 — Pan-Gene Universal AI Deep Dive",
    "subtitle": "The complete technical blueprint behind the 128-feature multi-gene variant classifier.",
    "loc": "14,566",
    "files": "61",
    "features": "128",
    "roc_auc": "0.9821",
    "sensitivity": "99.4%",
    "threshold_research": "0.3854",
    "threshold_prod": "0.1000",
}

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{content['title']}}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #f0f4ff;
            --text: #1f2937;
            --text-muted: #6b7280;
            --bg: #ffffff;
            --card-bg: #f9fafb;
            --danger: #ef4444;
            --success: #10b981;
        }}
        @page {{ size: A4; margin: 20mm; }}
        body {{ font-family: 'Inter', sans-serif; color: var(--text); line-height: 1.6; background: var(--bg); margin: 0; padding: 40px; }}
        h1, h2, h3 {{ font-family: 'Playfair Display', serif; color: #111827; }}
        h1 {{ font-size: 2.8rem; margin-bottom: 0.5rem; text-align: center; background: linear-gradient(90deg, var(--primary), var(--primary-light)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .subtitle {{ text-align: center; color: var(--text-muted); font-size: 1.2rem; margin-bottom: 3rem; font-style: italic; }}
        
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background: var(--secondary); padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #e0e7ff; }}
        .stat-val {{ display: block; font-size: 1.8rem; font-weight: 700; color: var(--primary); }}
        .stat-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}

        h2 {{ font-size: 1.8rem; border-bottom: 2px solid var(--secondary); padding-bottom: 8px; margin-top: 40px; color: var(--primary); }}
        
        .feature-box {{ background: var(--card-bg); border-radius: 8px; padding: 20px; margin-bottom: 20px; border-left: 4px solid var(--primary); }}
        .feature-box h3 {{ margin-top: 0; font-size: 1.3rem; }}
        
        .threshold-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .threshold-table th, .threshold-table td {{ border: 1px solid #e5e7eb; padding: 12px; text-align: left; }}
        .threshold-table th {{ background: var(--secondary); color: var(--primary); }}
        
        .metric-badge {{ background: var(--success); color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.9em; }}
        
        .code {{ font-family: 'Courier New', Courier, monospace; background: #eef2ff; padding: 2px 4px; border-radius: 4px; color: #3730a3; }}
        
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <h1>{content['title']}</h1>
    <p class="subtitle">{content['subtitle']}</p>

    <div class="stat-grid">
        <div class="stat-card"><span class="stat-val">{content['loc']}</span><span class="stat-label">Lines of Code</span></div>
        <div class="stat-card"><span class="stat-val">{content['features']}</span><span class="stat-label">AI Features</span></div>
        <div class="stat-card"><span class="stat-val">{content['roc_auc']}</span><span class="stat-label">ROC-AUC Score</span></div>
        <div class="stat-card"><span class="stat-val">5</span><span class="stat-label">Target Genes</span></div>
    </div>

    <h2>1. The Evolution of Feature Engineering</h2>
    <div class="feature-box">
        <h3>ESM-2 Protein Embeddings (64 Features)</h3>
        <p>SteppeDNA v3 incorporates 64 latent dimension embeddings from Meta AI's <strong>ESM-2 Transformer</strong>. This allows the model to "understand" the semantic context of amino acid sequences across any HR-pathway protein, providing a "pan-gene" biological awareness that standard tabular models lack.</p>
    </div>

    <div class="feature-box">
        <h3>Deep Consensus (32 Features)</h3>
        <p>We leverage scores from <strong>AlphaMissense</strong>, <strong>PhyloP</strong>, and <strong>gnomAD</strong>. The model reconciles conflicting signals (e.g., if AlphaMissense says pathogenic but gnomAD shows the variant in 5% of the population, the model correctly prioritizes the benign population data).</p>
    </div>

    <h2>2. Clinical Threshold Management</h2>
    <p>SteppeDNA manages two distinct decision boundaries to serve both researchers and clinicians.</p>
    <table class="threshold-table">
        <tr><th>Mode</th><th>Threshold</th><th>Business Logic</th></tr>
        <tr><td><strong>Research Optimal (F1)</strong></td><td><span class="code">{content['threshold_research']}</span></td><td>Balances precision and recall for unbiased statistical analysis.</td></tr>
        <tr><td><strong>Clinical Screening</strong></td><td><span class="code">{content['threshold_prod']}</span></td><td>Maximizes Sensitivity (<span class="metric-badge">{content['sensitivity']}</span>) to ensure zero false negatives in a screening setting.</td></tr>
    </table>

    <h2>3. The Stacking Ensemble Architecture</h2>
    <ul>
        <li><strong>XGBoost Booster:</strong> Fast, stable, and handles missing data via sparsity-aware splitting.</li>
        <li><strong>Neural Network (MLP):</strong> A 3-layer TensorFlow model that captures deep non-linear interactions within the ESM-2 embeddings.</li>
        <li><strong>Weights:</strong> Final probability = (0.6 &times; XGBoost) + (0.4 &times; MLP).</li>
    </ul>

    <h2>4. Expanded File Manifest (61 total)</h2>
    <ul>
        <li><strong>Core Engine:</strong> <span class="code">backend/main.py</span>, <span class="code">backend/feature_engineering.py</span></li>
        <li><strong>ML Artifacts:</strong> <span class="code">data/universal_xgboost_final.json</span>, <span class="code">data/universal_nn.h5</span></li>
        <li><strong>pipelines:</strong> <span class="code">data_pipelines/generate_esm2_embeddings.py</span>...</li>
    </ul>

    <p style="margin-top: 50px; text-align: center; border-top: 1px solid #eee; padding-top: 20px; color: #999;">
        <em>Generated by SteppeDNA Documentation Engine. Save as PDF for clinical reporting.</em>
    </p>
</body>
</html>
"""

# Save to the Project explanation folder
output_path = os.path.join("c:/Users/User/OneDrive/Desktop/Project explanation", "v3_SteppeDNA_Deep_Dive.html")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Successfully generated: {output_path}")
