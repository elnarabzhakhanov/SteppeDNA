/* SteppeDNA — Application Logic */
/* Depends on: api.js (loaded first) which provides API_URL and VCF_API */
"use strict";

// ─── HTML Escaping ──────────────────────────────────────────────────────────
// Prevents XSS when interpolating backend data into innerHTML templates.
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ─── Fetch with Timeout ────────────────────────────────────────────────────
// Wraps fetch() with an AbortController timeout to prevent indefinite hangs.
function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timeout));
}

// ─── Debounce helper ─────────────────────────────────────────────────────
// Prevents rapid repeated clicks on fetch buttons (ClinVar, gnomAD)
const _debounceTimers = {};
function isDebounced(key, delayMs = 2000) {
    if (_debounceTimers[key]) return true;
    _debounceTimers[key] = setTimeout(() => { delete _debounceTimers[key]; }, delayMs);
    return false;
}

// ─── Variant Analysis History (localStorage) ────────────────────────────────
const HISTORY_KEY = 'steppedna_history';
const MAX_HISTORY = 50;

function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
    catch { return []; }
}

function saveToHistory(entry) {
    const history = getHistory();
    // Deduplicate by variant key (gene + AA change)
    const key = `${entry.gene}_${entry.aa_ref}${entry.aa_pos}${entry.aa_alt}`;
    const idx = history.findIndex(h => `${h.gene}_${h.aa_ref}${h.aa_pos}${h.aa_alt}` === key);
    if (idx !== -1) history.splice(idx, 1);
    history.unshift(entry);
    if (history.length > MAX_HISTORY) history.length = MAX_HISTORY;
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        if (e.name === 'QuotaExceededError' || e.code === 22) {
            // Auto-prune oldest 10 entries and retry
            history.splice(Math.max(0, history.length - 10), 10);
            try {
                localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            } catch (_) { /* give up silently */ }
            showToast('History storage full — oldest entries removed.', 'error');
        }
    }
    renderHistory();
}

function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
}

function renderHistory() {
    const panel = document.getElementById('historyPanel');
    const list = document.getElementById('historyList');
    if (!panel || !list) return;
    const history = getHistory();
    if (history.length === 0) {
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    list.innerHTML = history.map((h, i) => {
        const tierClass = h.probability > 0.7 ? 'pathogenic' : (h.probability < 0.3 ? 'benign' : 'uncertain');
        const pct = (h.probability * 100).toFixed(1);
        // Custom date formatting — browser kk-KZ locale falls back to Russian
        const kkMonths = ['Qan', 'Aqp', 'Nau', 'Sau', 'Mam', 'Mau', 'Shil', 'Tam', 'Qyr', 'Qaz', 'Qar', 'Zhel'];
        const dt = new Date(h.timestamp);
        let timeStr;
        if (currentLang === 'kk') {
            const hh = String(dt.getHours()).padStart(2, '0');
            const mm = String(dt.getMinutes()).padStart(2, '0');
            timeStr = `${dt.getDate()} ${kkMonths[dt.getMonth()]}, ${hh}:${mm}`;
        } else {
            const langLocale = { en: 'en-US', ru: 'ru-RU' };
            timeStr = dt.toLocaleDateString(langLocale[currentLang] || 'en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }
        return `<div class="history-item ${tierClass}" data-index="${i}" role="button" tabindex="0" title="Click to re-analyze">
            <span class="history-variant">${escapeHtml(h.gene)} ${escapeHtml(h.aa_ref)}${h.aa_pos}${escapeHtml(h.aa_alt)}</span>
            <span class="history-prob">${pct}%</span>
            <span class="history-time">${escapeHtml(timeStr)}</span>
        </div>`;
    }).join('');

    // Click handler to re-fill the form
    list.querySelectorAll('.history-item').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.index);
            const h = history[idx];
            if (!h) return;
            const geneSelect = document.getElementById('geneSelect');
            const cdnaInput = document.getElementById('cDNA_pos');
            const refInput = document.getElementById('AA_ref');
            const altInput = document.getElementById('AA_alt');
            if (geneSelect) geneSelect.value = h.gene;
            if (cdnaInput) cdnaInput.value = h.cdna_pos;
            if (refInput) refInput.value = h.aa_ref;
            if (altInput) altInput.value = h.aa_alt;
            // Trigger gene change to update UI
            if (geneSelect) geneSelect.dispatchEvent(new Event('change'));
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

// ─── Global Error Boundary ──────────────────────────────────────────────────
// Catches unhandled rejections and runtime errors so the UI never silently breaks.
window.addEventListener('unhandledrejection', e => {
    console.error('[SteppeDNA] Unhandled promise rejection:', e.reason);
    if (typeof showToast === 'function') {
        showToast('Something went wrong. Please try again.', 'error');
    }
    e.preventDefault();
});
window.addEventListener('error', e => {
    console.error('[SteppeDNA] Runtime error:', e.message, e.filename, e.lineno);
});

// the dark mode toggle is hacky but it survives page reloads.
const themeToggle = document.getElementById('themeToggle');
const htmlEl = document.documentElement;
const moonIcon = document.getElementById('moonIcon');
const sunIcon = document.getElementById('sunIcon');

const savedTheme = localStorage.getItem('theme');
const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
    htmlEl.setAttribute('data-theme', 'dark');
    moonIcon.style.display = 'block';
    sunIcon.style.display = 'none';
}

themeToggle.addEventListener('click', () => {
    if (htmlEl.getAttribute('data-theme') === 'dark') {
        htmlEl.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
        moonIcon.style.display = 'none';
        sunIcon.style.display = 'block';
    } else {
        htmlEl.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        moonIcon.style.display = 'block';
        sunIcon.style.display = 'none';
    }
});

// API_URL, VCF_API, METRICS_API are defined in api.js (loaded before this file)

// Fetch and display model metrics in hero section
(async function loadHeroMetrics() {
    try {
        const resp = await fetch(METRICS_API, { signal: AbortSignal.timeout(5000) });
        if (!resp.ok) return;
        const m = await resp.json();
        const el = (id) => document.getElementById(id);
        if (el('hero-roc-auc') && m.roc_auc) el('hero-roc-auc').textContent = m.roc_auc.toFixed(3);
        if (el('hero-mcc') && m.mcc) el('hero-mcc').textContent = m.mcc.toFixed(3);
        if (el('hero-bal-acc') && m.balanced_accuracy) el('hero-bal-acc').textContent = (m.balanced_accuracy * 100).toFixed(1) + '%';
    } catch (_) { /* backend may not be running */ }
})();

// custom toast because importing a whole library for 3 alerts is ridiculous
function showToast(message, type = 'error', duration = 4000) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'error'
        ? '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>'
        : type === 'warning'
            ? '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            : '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="#6260FF" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>';
    const msgSpan = document.createElement('span');
    msgSpan.textContent = message;
    toast.innerHTML = icon;
    toast.appendChild(msgSpan);
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

const VALID_AAS = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
    'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val',
    'Fs', 'Del', 'Ins', 'Dup', '*'];

// Maximum cDNA lengths per supported gene (single source of truth for frontend validation)
const GENE_MAX_CDNA = { 'BRCA1': 5592, 'BRCA2': 10257, 'PALB2': 3561, 'RAD51C': 1131, 'RAD51D': 987 };

// Maximum AA positions per gene (protein length)
const GENE_MAX_AA = { 'BRCA1': 1863, 'BRCA2': 3418, 'PALB2': 1186, 'RAD51C': 376, 'RAD51D': 328 };

// ─── Translation helpers for dynamic result content ─────────────────────────
// These translate backend response strings into the active UI language.
const RESULT_TR = {
    "High Confidence": { ru: "Высокая уверенность", kk: "Жоғары сенімділік", en: "High Confidence" },
    "Medium Confidence": { ru: "Средняя уверенность", kk: "Орташа сенімділік", en: "Medium Confidence" },
    "Moderate Confidence": { ru: "Средняя уверенность", kk: "Орташа сенімділік", en: "Moderate Confidence" },
    "Low Confidence": { ru: "Низкая уверенность", kk: "Төмен сенімділік", en: "Low Confidence" },
    "Pathogenic": { ru: "Патогенный", kk: "Патогендік", en: "Pathogenic" },
    "Benign": { ru: "Доброкачественный", kk: "Қатерсіз", en: "Benign" },
    "Neutral": { ru: "Нейтральный", kk: "Бейтарап", en: "Neutral" },
    "Abnormal": { ru: "Аномальный", kk: "Аномальді", en: "Abnormal" },
    "Normal": { ru: "Нормальный", kk: "Қалыпты", en: "Normal" },
    "Conserved": { ru: "Консервативный", kk: "Консервативті", en: "Conserved" },
    "Ultra-conserved": { ru: "Ультраконсервативный", kk: "Ультраконсервативті", en: "Ultra-conserved" },
    "Buried": { ru: "Погруженный", kk: "Көмкерілген", en: "Buried" },
    "Surface": { ru: "Поверхностный", kk: "Беткі", en: "Surface" },
    "DNA Contact": { ru: "Контакт с ДНК", kk: "ДНҚ контакты", en: "DNA Contact" },
    "Unknown": { ru: "Неизвестно", kk: "Белгісіз", en: "Unknown" },
    "No Data": { ru: "Нет данных", kk: "Деректер жоқ", en: "No Data" },
    "AlphaMissense": { ru: "AlphaMissense", kk: "AlphaMissense", en: "AlphaMissense" },
    "MAVE Score": { ru: "Оценка MAVE", kk: "MAVE ұпайы", en: "MAVE Score" },
    "PhyloP (Cons)": { ru: "PhyloP (Конс)", kk: "PhyloP (Конс)", en: "PhyloP (Cons)" },
    "3D Structure": { ru: "3D Структура", kk: "3D құрылымы", en: "3D Structure" },
    "BLOSUM62": { ru: "BLOSUM62", kk: "BLOSUM62", en: "BLOSUM62" },
    "Volume Diff": { ru: "Разница объемов", kk: "Көлем айырмасы", en: "Volume Diff" },
    "Hydro Diff": { ru: "Разница гидрофоб.", kk: "Гидрофоб. айырмасы", en: "Hydro Diff" },
    "Charge Change": { ru: "Смена заряда", kk: "Зарядтың өзгеруі", en: "Charge Change" },
    "Nonsense": { ru: "Нонсенс", kk: "Нонсенс", en: "Nonsense" },
    "Critical Domain": { ru: "Критический домен", kk: "Критикалық домен", en: "Critical Domain" },
    "Yes": { ru: "Да", kk: "Иә", en: "Yes" },
    "No": { ru: "Нет", kk: "Жоқ", en: "No" }
};
function tr(k) {
    if (RESULT_TR[k] && RESULT_TR[k][currentLang]) return RESULT_TR[k][currentLang];
    if (typeof i18n !== 'undefined' && i18n[currentLang] && i18n[currentLang][k]) return i18n[currentLang][k];
    return k;
}
function fmtScore(v) { return (v !== undefined && v !== null) ? escapeHtml(String(v)) : '<span style="color:#ccc">-</span>'; }
const tip = (fallback, key) => {
    const text = (key && typeof i18n !== 'undefined' && i18n[currentLang] && i18n[currentLang][key]) ? i18n[currentLang][key] : fallback;
    const attr = key ? ` data-i18n-tooltip="${escapeHtml(key)}"` : '';
    return `<button class="help-tip" aria-label="Help">?<span class="help-text"${attr}>${escapeHtml(text)}</span></button>`;
};

// ─── Shared state for re-rendering on language change ───────────────────────
let lastPredictionData = null;
let lastPredictionInput = null;

// ─── Reusable rendering helpers (called from submit + langchange) ───────────
function renderConfidenceHtml(ci) {
    if (!ci || !ci.label) return '';
    const ciColor = ci.label === 'High Confidence' ? 'var(--success)' : (ci.label === 'Low Confidence' ? 'var(--danger)' : 'var(--warning)');

    // CI label text: show method if bootstrap
    const isBootstrap = ci.method === 'bootstrap';
    const ciPctLabel = isBootstrap ? '90% CI' : '95% CI';
    const methodTag = isBootstrap
        ? ' <span style="font-size:0.7rem;color:var(--text-mid);font-style:italic">(bootstrap, ' + escapeHtml(String(ci.n_models || 50)) + ' models)</span>'
        : '';

    // CI text line
    const ciLine = (ci.ci_lower != null && ci.ci_upper != null)
        ? `<div style="font-size:0.8rem;color:var(--text-body);margin-top:4px">${escapeHtml(ciPctLabel)}: ${(ci.ci_lower * 100).toFixed(1)}% – ${(ci.ci_upper * 100).toFixed(1)}%${methodTag}</div>`
        : '';

    // Visual error bar showing CI range on a 0-100% scale
    let errorBarHtml = '';
    if (ci.ci_lower != null && ci.ci_upper != null && ci.probability != null) {
        const pPct = (ci.probability * 100).toFixed(1);
        const lPct = (ci.ci_lower * 100);
        const uPct = (ci.ci_upper * 100);
        const widthPct = ci.ci_width != null ? (ci.ci_width * 100).toFixed(1) : (uPct - lPct).toFixed(1);
        // Bar colors based on width
        const barColor = parseFloat(widthPct) < 10 ? 'var(--success)' : (parseFloat(widthPct) < 25 ? 'var(--warning)' : 'var(--danger)');
        errorBarHtml = `
            <div style="position:relative;height:18px;background:var(--bg-input);border-radius:9px;margin-top:6px;overflow:hidden" title="CI width: ${escapeHtml(widthPct)}%">
                <div style="position:absolute;left:${lPct}%;width:${Math.max(uPct - lPct, 1)}%;height:100%;background:${barColor};opacity:0.25;border-radius:9px"></div>
                <div style="position:absolute;left:${Math.max(ci.probability * 100 - 0.5, 0)}%;width:3px;height:100%;background:${barColor};border-radius:2px"></div>
                <div style="position:absolute;right:4px;top:1px;font-size:0.65rem;color:var(--text-mid)">CI width: ${escapeHtml(widthPct)}%</div>
            </div>`;
    }

    return `<span style="display:inline-block;padding:3px 12px;border-radius:12px;font-size:0.82rem;font-weight:600;background:${ciColor}15;color:${ciColor};border:1px solid ${ciColor}40;margin-bottom:6px">${escapeHtml(tr(ci.label))}</span>${ciLine}${errorBarHtml}`;
}

function renderDsGridHtml(ds) {
    const am = ds.alphamissense || {};
    const mave = ds.mave || {};
    const phy = ds.phylop || {};
    const struct = ds.structure || {};
    return `
        <div class="ds-card">
            <div class="ds-title">${tr('AlphaMissense')} ${tip('Google DeepMind predictor trained on primate evolution + unsupervised learning. Score 0-1, >0.564 = likely pathogenic.', 'tip_alphamissense')}</div>
            <div class="ds-val">${fmtScore(am.score)}</div>
            <div class="ds-badge ${am.label === 'Pathogenic' ? 'pathogenic' : (am.label === 'Benign' ? 'benign' : 'neutral')}">${escapeHtml(tr(am.label || 'No Data'))}</div>
        </div>
        <div class="ds-card">
            <div class="ds-title">${tr('MAVE Score')} ${tip('Multiplexed Assay of Variant Effect. Wet-lab measurement of protein function. Lower scores = loss of function = likely pathogenic.', 'tip_mave')}</div>
            <div class="ds-val">${fmtScore(mave.score)}</div>
            <div class="ds-badge ${mave.label === 'Abnormal' ? 'pathogenic' : (mave.label === 'Normal' ? 'benign' : 'neutral')}">${escapeHtml(tr(mave.label || 'No Data'))}</div>
        </div>
        <div class="ds-card">
            <div class="ds-title">${tr('PhyloP (Cons)')} ${tip('Phylogenetic P-value measuring evolutionary conservation across 100 vertebrate species. Higher = more conserved = mutations more likely damaging.', 'tip_phylop')}</div>
            <div class="ds-val">${fmtScore(phy.score)}</div>
            <div class="ds-badge ${phy.label === 'Conserved' || phy.label === 'Ultra-conserved' ? 'pathogenic' : 'neutral'}">${escapeHtml(tr(phy.label || 'No Data'))}</div>
        </div>
        <div class="ds-card">
            <div class="ds-title">${tr('3D Structure')} ${tip('AlphaFold-predicted protein structure. Shows if the mutation site is buried inside the protein, on the surface, or in contact with DNA.', 'tip_3d_structure')}</div>
            <div class="ds-val">${struct.domain ? escapeHtml(struct.domain.replace(/_/g, ' ')) : escapeHtml(tr('Unknown'))}</div>
            <div class="ds-badge ${struct.is_buried ? 'pathogenic' : (struct.is_dna_contact ? 'pathogenic' : 'neutral')}">${escapeHtml(tr(struct.is_dna_contact ? 'DNA Contact' : (struct.is_buried ? 'Buried' : (struct.secondary_structure || 'Surface'))))}</div>
        </div>`;
}

function renderFeaturesGridHtml(f) {
    return `
        <div class="feat"><span class="lbl">${escapeHtml(tr('BLOSUM62'))} ${tip('Amino acid substitution score from the BLOSUM62 matrix. Negative = rare/damaging substitution. Positive = common/tolerated.', 'tip_blosum62')}</span><span class="val">${escapeHtml(String(f.blosum62_score))}</span></div>
        <div class="feat"><span class="lbl">${escapeHtml(tr('Volume Diff'))} ${tip('Difference in physical volume between original and mutant amino acid (in cubic angstroms). Large changes disrupt protein packing.', 'tip_volume_diff')}</span><span class="val">${escapeHtml(String(f.volume_diff))}</span></div>
        <div class="feat"><span class="lbl">${escapeHtml(tr('Hydro Diff'))} ${tip('Difference in hydrophobicity (water-repelling tendency). Large changes can flip a residue from buried to exposed or vice versa.', 'tip_hydro_diff')}</span><span class="val">${escapeHtml(String(f.hydro_diff))}</span></div>
        <div class="feat"><span class="lbl">${escapeHtml(tr('Charge Change'))} ${tip('Whether the mutation changes the electric charge of the amino acid (e.g. positive to negative). Charge changes often disrupt protein interactions.', 'tip_charge_change')}</span><span class="val">${escapeHtml(tr(f.charge_changed ? 'Yes' : 'No'))}</span></div>
        <div class="feat"><span class="lbl">${escapeHtml(tr('Nonsense'))} ${tip('Whether this mutation creates a premature stop codon (Ter), truncating the protein. Almost always pathogenic.', 'tip_nonsense')}</span><span class="val">${escapeHtml(tr(f.is_nonsense ? 'Yes' : 'No'))}</span></div>
        <div class="feat"><span class="lbl">${escapeHtml(tr('Critical Domain'))} ${tip('Whether the mutation falls in a known functional domain (e.g. DNA binding, PALB2 binding, BRC repeats). Domain mutations are more likely damaging.', 'tip_critical_domain')}</span><span class="val">${escapeHtml(tr(f.in_critical_domain ? 'Yes' : 'No'))}</span></div>`;
}

function updateResultTranslations() {
    if (!lastPredictionData) return;
    const data = lastPredictionData;
    const body = lastPredictionInput;
    const resultCard = document.getElementById('resultCard');
    if (!resultCard || resultCard.style.display === 'none') return;

    const aaPos = data.aa_pos || data.features_used?.aa_position || Math.ceil(body.cDNA_pos / 3);
    // Note: for re-render path, aa_pos should already be set from the original prediction

    // Badge
    document.getElementById('badge').textContent = tr(data.prediction);

    // Confidence
    const ci = data.confidence || {};
    const ciDiv = document.getElementById('uncertaintyInfo');
    if (ci.label) {
        ciDiv.innerHTML = renderConfidenceHtml(ci);
        ciDiv.style.display = 'block';
    }

    // Data sources + Features
    document.getElementById('dsGrid').innerHTML = renderDsGridHtml(data.data_sources || {});
    document.getElementById('featuresGrid').innerHTML = renderFeaturesGridHtml(data.features_used);

    // Lollipop plot (domain labels need translation-ready font)
    renderLollipopPlot(body.gene_name, aaPos, data.prediction);

    // SHAP toggle button text
    const shapToggleBtn = document.getElementById('shapToggleBtn');
    if (shapToggleBtn && shapToggleBtn.style.display !== 'none') {
        const txt = shapToggleBtn.textContent;
        const isExpanded = txt.includes('Top 8') || txt.includes('8');
        shapToggleBtn.textContent = isExpanded
            ? (i18n[currentLang]['shap_show_top'] || 'Show Top 8 Only')
            : (i18n[currentLang]['shap_show_all'] || 'Show All Features');
    }
}

// users kept typing 'ala' instead of 'Ala' and breaking the python dictionary lookups. brute forcing it here.
// real-time auto-capitalization as the user types
function forceAACase(el) {
    el.addEventListener('input', function () {
        const pos = this.selectionStart;
        let v = this.value;
        if (v.length >= 1) v = v.charAt(0).toUpperCase() + v.slice(1).toLowerCase();
        this.value = v;
        this.setSelectionRange(pos, pos);
    });
}
forceAACase(document.getElementById('AA_ref'));
forceAACase(document.getElementById('AA_alt'));

// mutation field (G>C) must be fully uppercase
document.getElementById('mutation').addEventListener('input', function () {
    const pos = this.selectionStart;
    this.value = this.value.toUpperCase();
    this.setSelectionRange(pos, pos);
});
document.getElementById('cDNA_pos').addEventListener('blur', function () {
    const v = parseInt(this.value);
    const gene = document.getElementById('geneSelect')?.value;
    if (!gene) return;
    const maxLen = GENE_MAX_CDNA[gene] || 15000;
    if (this.value !== '' && (isNaN(v) || v < 1 || v > maxLen)) {
        showToast(`cDNA position must be between 1 and ${maxLen} for ${gene}`, 'warning');
        this.style.borderColor = 'var(--danger)';
    } else {
        this.style.borderColor = '';
    }
});

document.getElementById('AA_ref').addEventListener('blur', function () {
    let v = this.value.trim();
    if (v.length === 3) {
        v = v.charAt(0).toUpperCase() + v.slice(1).toLowerCase();
        this.value = v;
    }
    if (v !== '' && !VALID_AAS.includes(v)) {
        showToast('Invalid Reference AA: "' + v + '" — try Ala, Val, Gly, Ter, Del, Fs, etc.', 'warning');
        this.style.borderColor = 'var(--danger)';
    } else {
        this.style.borderColor = '';
    }
});

document.getElementById('AA_alt').addEventListener('blur', function () {
    let v = this.value.trim();
    if (v.length === 3) {
        v = v.charAt(0).toUpperCase() + v.slice(1).toLowerCase();
        this.value = v;
    }
    if (v !== '' && !VALID_AAS.includes(v)) {
        showToast('Invalid Alternate AA: "' + v + '" — try Ala, Val, Gly, Ter, Del, Fs, etc.', 'warning');
        this.style.borderColor = 'var(--danger)';
    } else {
        this.style.borderColor = '';
    }
});

document.getElementById('geneSelect').addEventListener('change', function () {
    const gene = this.value;
    const maxLen = GENE_MAX_CDNA[gene] || 15000;

    const cdnaInput = document.getElementById('cDNA_pos');
    cdnaInput.max = maxLen;
    cdnaInput.placeholder = `1 - ${maxLen}`;

    const cdnaHint = document.getElementById('cdnaHint');
    if (cdnaHint) {
        cdnaHint.textContent = `(1 - ${maxLen})`;
    }
});

// ---- PREMIUM DROPDOWN LOGIC (with keyboard accessibility) ----
document.querySelectorAll('.custom-dropdown').forEach(dropdown => {
    dropdown.setAttribute('role', 'listbox');
    dropdown.setAttribute('aria-expanded', 'false');

    dropdown.addEventListener('click', function (e) {
        document.querySelectorAll('.custom-dropdown').forEach(d => {
            if (d !== this) { d.classList.remove('open'); d.setAttribute('aria-expanded', 'false'); }
        });
        const isOpen = this.classList.toggle('open');
        this.setAttribute('aria-expanded', String(isOpen));
    });

    // Keyboard navigation: ArrowDown, ArrowUp, Enter, Escape
    dropdown.addEventListener('keydown', function (e) {
        const options = Array.from(this.querySelectorAll('.dropdown-option'));
        const activeIdx = options.findIndex(o => o.classList.contains('active'));

        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            if (!this.classList.contains('open')) {
                this.classList.add('open');
                this.setAttribute('aria-expanded', 'true');
                return;
            }
            const next = e.key === 'ArrowDown'
                ? Math.min(activeIdx + 1, options.length - 1)
                : Math.max(activeIdx - 1, 0);
            options.forEach(o => o.classList.remove('active'));
            options[next].classList.add('active');
            options[next].scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (this.classList.contains('open') && activeIdx >= 0) {
                options[activeIdx].click();
            } else {
                this.classList.add('open');
                this.setAttribute('aria-expanded', 'true');
            }
        } else if (e.key === 'Escape') {
            this.classList.remove('open');
            this.setAttribute('aria-expanded', 'false');
        }
    });

    const options = dropdown.querySelectorAll('.dropdown-option');
    const selectedText = dropdown.querySelector('.dropdown-selected');
    const container = dropdown.closest('.custom-dropdown-container');
    const hiddenInput = container ? container.querySelector('input[type="hidden"]') : null;

    options.forEach((opt, i) => {
        opt.setAttribute('role', 'option');
        opt.addEventListener('click', function (e) {
            e.stopPropagation();
            options.forEach(o => { o.classList.remove('active'); o.setAttribute('aria-selected', 'false'); });
            this.classList.add('active');
            this.setAttribute('aria-selected', 'true');

            selectedText.textContent = this.textContent;
            dropdown.classList.remove('open');
            dropdown.setAttribute('aria-expanded', 'false');

            const val = this.getAttribute('data-value');

            if (hiddenInput) {
                hiddenInput.value = val;
                hiddenInput.dispatchEvent(new Event('change'));
            }

            if (dropdown.id === 'mutTypeDropdown') {
                const aaAltGroup = document.getElementById('aaAltGroup');
                const aaAltInput = document.getElementById('AA_alt');
                const mutationInput = document.getElementById('mutation');

                if (val === 'missense') {
                    aaAltGroup.style.display = 'block';
                    aaAltInput.value = '';
                    mutationInput.value = '';
                } else if (val === 'nonsense') {
                    aaAltGroup.style.display = 'none';
                    aaAltInput.value = 'Ter';
                    mutationInput.value = '';
                } else if (val === 'frameshift') {
                    aaAltGroup.style.display = 'none';
                    aaAltInput.value = 'Del';
                    mutationInput.value = 'fs';
                }
            }
        });
    });
});

document.addEventListener('click', function (e) {
    if (!e.target.closest('.custom-dropdown')) {
        document.querySelectorAll('.custom-dropdown').forEach(d => {
            d.classList.remove('open');
            d.setAttribute('aria-expanded', 'false');
        });
    }
});

// vcf drag and drop logic. the backend takes forever so we show a spinner.
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
// VCF_API is defined in api.js

function handleVCFFile(file) {
    if (!file.name.toLowerCase().endsWith('.vcf')) {
        showToast(i18n[currentLang]['err_vcf_only'] || 'Only .vcf files are accepted', 'error');
        return;
    }
    uploadVCF(file);
}

async function uploadVCF(file) {
    const vcfResults = document.getElementById('vcfResults');
    const vcfLoading = document.getElementById('vcfLoading');
    const vcfBody = document.getElementById('vcfBody');
    const vcfSummary = document.getElementById('vcfSummary');

    vcfResults.style.display = 'block';
    vcfLoading.style.display = 'block';
    vcfBody.innerHTML = '';
    vcfSummary.innerHTML = '';

    // Update drop zone to show file name
    dropZone.querySelector('h3').textContent = file.name;
    dropZone.querySelector('p').textContent = 'Uploading & analyzing...';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const geneCtx = document.getElementById('vcfGeneSelect').value;
        formData.append('gene', geneCtx);

        const resp = await fetchWithTimeout(VCF_API, {
            method: 'POST',
            body: formData,
        }, 120000); // 2 minute timeout for large VCF files

        if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
        const data = await resp.json();

        if (data.error) {
            showToast(data.error, 'error');
            vcfResults.style.display = 'none';
            return;
        }

        vcfLoading.style.display = 'none';


        // Summary stats
        const nPath = data.predictions.filter(p => p.prediction === 'Pathogenic' || p.prediction === 'Likely Pathogenic').length;
        const nBen = data.predictions.filter(p => (p.prediction || '').includes('Benign')).length;
        const nVUS = data.predictions.filter(p => (p.prediction || '').includes('VUS')).length;
        const nTotal = Number(data.total_variants_in_file) || 0;
        const nAnalyzedStat = data.variants_classified !== undefined ? Number(data.variants_classified) : data.predictions.length;
        vcfSummary.innerHTML = `
            <div class="vcf-stat"><div class="num">${escapeHtml(String(nTotal))}</div> <span>Total in file</span></div>
            <div class="vcf-stat"><div class="num">${escapeHtml(String(nAnalyzedStat))}</div> <span>Classified</span></div>
            <div class="vcf-stat" style="background:#FEE2E2; color:#1f2937;"><div class="num" style="color:var(--danger)">${escapeHtml(String(nPath))}</div> <span>Pathogenic</span></div>
            <div class="vcf-stat" style="background:#FEF3C7; color:#92400e;"><div class="num" style="color:#d97706">${escapeHtml(String(nVUS))}</div> <span>VUS</span></div>
            <div class="vcf-stat" style="background:#D1FAE5; color:#065f46;"><div class="num" style="color:var(--success)">${escapeHtml(String(nBen))}</div> <span>Benign</span></div>
        `;

        // ─── Compound Heterozygosity Warning ──────────────────────────────
        const chwDiv = document.getElementById('compoundHetWarning');
        if (chwDiv) {
            if (data.compound_het_warning) {
                const chw = data.compound_het_warning;
                const varList = chw.variants.map(v => `<li>${escapeHtml(v)}</li>`).join('');
                const gtNote = !chw.has_gt_data ? '<p style="font-size:0.85rem;margin-top:0.5rem;opacity:0.8">Note: VCF file did not contain genotype (GT) data. Warning is based on variant count only.</p>' : '';
                chwDiv.innerHTML = `
                    <div class="compound-het-warning" role="alert">
                        <strong>&#9888; Compound Heterozygosity Warning</strong>
                        <p>${escapeHtml(chw.message)}</p>
                        <ul style="margin:0.5rem 0 0 1.2rem;font-size:0.9rem">${varList}</ul>
                        ${gtNote}
                    </div>`;
                chwDiv.style.display = 'block';
            } else {
                chwDiv.innerHTML = '';
                chwDiv.style.display = 'none';
            }
        }

        // ─── Variant type badge helper ────────────────────────────────────
        const typeBadge = (vt) => {
            const map = {
                'missense': ['Missense', 'badge-info'],
                'nonsense': ['Nonsense', 'badge-pathogenic'],
                'frameshift': ['Frameshift', 'badge-pathogenic'],
                'splice_canonical': ['Splice', 'badge-pathogenic'],
                'splice_near': ['Near Splice', 'badge-warning'],
                'inframe_indel': ['In-Frame Indel', 'badge-warning'],
                'synonymous': ['Synonymous', 'badge-benign'],
                'unknown': ['Unknown Type', 'badge-info'],
            };
            const [label, cls] = map[vt] || [String(vt || 'Unknown Type'), 'badge-info'];
            return `<span class="badge ${cls}" style="font-size:0.75rem;padding:2px 8px">${escapeHtml(label)}</span>`;
        };

        // Separate non-synonymous (main table) from synonymous (collapsible)
        const mainPreds = data.predictions.filter(p => p.variant_type !== 'synonymous');
        const synPreds = data.predictions.filter(p => p.variant_type === 'synonymous');

        if (mainPreds.length === 0 && synPreds.length === 0) {
            vcfBody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--text-body)">No classifiable variants found in this VCF file for ${escapeHtml(document.getElementById('vcfGeneSelect').value)}</td></tr>`;
        } else {
            // Sort: pathogenic first, then by probability desc
            mainPreds.sort((a, b) => b.probability - a.probability);
            const renderRow = (p, i) => {
                const tier = p.probability > 0.7 ? 'high' : (p.probability < 0.3 ? 'low' : 'mid');
                const pred = p.prediction || '';
                const rowClass = pred.includes('Pathogenic') ? 'row-pathogenic' : (pred.includes('VUS') ? 'row-uncertain' : (pred.includes('Benign') ? 'row-benign' : ''));
                const pct = Math.round(p.probability * 100);
                const predBadge = pred.includes('Pathogenic') ? 'badge-pathogenic' : (pred.includes('Benign') ? 'badge-benign' : 'badge-warning');
                return `<tr class="${rowClass}">
                    <td>${i + 1}</td>
                    <td>${typeBadge(p.variant_type)}</td>
                    <td><strong>${escapeHtml(p.hgvs_p)}</strong></td>
                    <td>${p.cdna_pos ? 'c.' + escapeHtml(String(p.cdna_pos)) : '-'}</td>
                    <td>${escapeHtml(p.aa_ref)} ${p.aa_ref !== p.aa_alt ? '→ ' + escapeHtml(p.aa_alt) : ''}</td>
                    <td>${escapeHtml(p.mutation)}</td>
                    <td><span class="badge ${predBadge}" style="font-size:0.8rem;padding:2px 10px">${escapeHtml(pred)}</span></td>
                    <td>${pct}% <div class="prob-bar"><div class="prob-fill ${tier}" style="width:${pct}%"></div></div></td>
                </tr>`;
            };
            vcfBody.innerHTML = mainPreds.map((p, i) => renderRow(p, i)).join('');
        }

        // ─── Synonymous Variants Collapsible ──────────────────────────────
        const skippedDiv = document.getElementById('skippedVariants');
        if (skippedDiv) {
            let extraHtml = '';
            if (synPreds.length > 0) {
                synPreds.sort((a, b) => b.probability - a.probability);
                const synRows = synPreds.map((p, i) => {
                    const pct = Math.round(p.probability * 100);
                    const tier = p.probability > 0.7 ? 'high' : (p.probability < 0.3 ? 'low' : 'mid');
                    const pred = p.prediction || '';
                    const predBadge = pred.includes('VUS') ? 'badge-warning' : 'badge-benign';
                    return `<tr>
                        <td>${i + 1}</td>
                        <td>${typeBadge('synonymous')}</td>
                        <td><strong>${escapeHtml(p.hgvs_p)}</strong></td>
                        <td>c.${escapeHtml(String(p.cdna_pos))}</td>
                        <td>${escapeHtml(p.aa_ref)}</td>
                        <td>${escapeHtml(p.mutation)}</td>
                        <td><span class="badge ${predBadge}" style="font-size:0.8rem;padding:2px 10px">${escapeHtml(pred)}</span></td>
                        <td>${pct}% <div class="prob-bar"><div class="prob-fill ${tier}" style="width:${pct}%"></div></div></td>
                    </tr>`;
                }).join('');
                extraHtml += `<details class="skipped-section">
                    <summary>Synonymous Variants (${synPreds.length})</summary>
                    <table class="vcf-table" style="margin-top:0.5rem"><thead><tr>
                        <th>#</th><th>Type</th><th>HGVS</th><th>cDNA</th><th>AA</th><th>Mut</th><th>Prediction</th><th>Prob</th>
                    </tr></thead><tbody>${synRows}</tbody></table>
                </details>`;
            }
            // Skipped variants section
            if (data.skipped_reasons && data.skipped_reasons.length > 0) {
                const skipRows = data.skipped_reasons.map((s, i) =>
                    `<tr><td>${escapeHtml(String(s.line || i + 1))}</td><td>${escapeHtml(String(s.pos || '-'))}</td><td>${escapeHtml(s.reason)}</td></tr>`
                ).join('');
                extraHtml += `<details class="skipped-section">
                    <summary>Skipped Variants (${data.skipped_count || data.skipped_reasons.length})</summary>
                    <table class="vcf-table" style="margin-top:0.5rem"><thead><tr>
                        <th>Line</th><th>Position</th><th>Reason</th>
                    </tr></thead><tbody>${skipRows}</tbody></table>
                </details>`;
            }
            skippedDiv.innerHTML = extraHtml;
        }

        // Reset drop zone text
        dropZone.querySelector('h3').textContent = file.name + ' \u2713';
        const nAnalyzed = data.variants_classified !== undefined ? data.variants_classified : data.predictions.length;
        dropZone.querySelector('p').textContent = `${nAnalyzed} variants classified \u00B7 drop another file`;

        // CSV download button
        const csvBtn = document.getElementById('csvDownloadBtn');
        if (data.predictions.length > 0) {
            csvBtn.style.display = 'inline-flex';
            csvBtn.onclick = () => {
                const headers = ['#', 'Gene', 'Type', 'HGVS', 'cDNA_pos', 'AA_ref', 'AA_alt', 'Mutation', 'Prediction', 'Probability'];
                const rows = data.predictions.map((p, i) => {
                    // CSV values use raw data (not HTML-escaped) — quote fields that may contain commas or quotes
                    const csvEscape = (v) => {
                        const s = String(v);
                        return (s.includes(',') || s.includes('"') || s.includes('\n'))
                            ? '"' + s.replace(/"/g, '""') + '"' : s;
                    };
                    return [
                        i + 1,
                        csvEscape(document.getElementById('vcfGeneSelect').value),
                        csvEscape(p.variant_type || 'unknown'),
                        csvEscape(p.hgvs_p),
                        p.cdna_pos,
                        csvEscape(p.aa_ref),
                        csvEscape(p.aa_alt),
                        csvEscape(p.mutation),
                        csvEscape(p.prediction),
                        (p.probability * 100).toFixed(1) + '%'
                    ].join(',');
                });
                const csv = [headers.join(','), ...rows].join('\n');
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'steppedna_predictions.csv';
                a.click();
                URL.revokeObjectURL(url);
            };
        } else {
            csvBtn.style.display = 'none';
        }

        // Scroll to results
        vcfResults.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        vcfLoading.style.display = 'none';
        vcfResults.style.display = 'none';
        const vcfMsg = err.name === 'AbortError'
            ? (i18n[currentLang]['err_timeout'] || 'Request timed out — the file may be too large or the backend is overloaded.')
            : (i18n[currentLang]['err_vcf_failed'] || 'VCF upload failed') + ': ' + err.message;
        showToast(vcfMsg, 'error');
        dropZone.querySelector('h3').textContent = 'Drop your VCF file here';
        dropZone.querySelector('p').textContent = 'or click to browse · .vcf format';
    }
}

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleVCFFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleVCFFile(fileInput.files[0]);
        fileInput.value = '';
    }
});

// main form submit handler
document.getElementById('mutationForm').addEventListener('submit', async e => {
    e.preventDefault();

    const btn = document.getElementById('submitBtn');
    const spinner = document.getElementById('spinner');
    const btnText = document.getElementById('btnText');
    const resultCard = document.getElementById('resultCard');

    btn.disabled = true;
    spinner.style.display = 'block';
    btnText.textContent = i18n[currentLang]['analyzing'] || 'Analyzing...';
    resultCard.style.display = 'none';
    const skeletonCard = document.getElementById('skeletonCard');
    if (skeletonCard) skeletonCard.style.display = 'block';

    let aaRef = document.getElementById('AA_ref').value.trim();
    let aaAlt = document.getElementById('AA_alt').value.trim();

    if (aaRef.length === 3) aaRef = aaRef.charAt(0).toUpperCase() + aaRef.slice(1).toLowerCase();
    if (aaAlt.length === 3) aaAlt = aaAlt.charAt(0).toUpperCase() + aaAlt.slice(1).toLowerCase();

    document.getElementById('AA_ref').value = aaRef;
    document.getElementById('AA_alt').value = aaAlt;
    const cdnaValStr = document.getElementById('cDNA_pos').value.trim();
    const cdnaVal = parseInt(cdnaValStr);

    // show active errors first so we don't spam the user with 5 toast notifications at once
    if (aaRef !== '' && !VALID_AAS.includes(aaRef)) {
        showToast('Invalid Reference AA: "' + aaRef + '" — try Ala, Val, Gly, Ter, Del, Fs, etc.', 'error');
        btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = i18n[currentLang]['analyze_btn'];
        return;
    }
    if (aaAlt !== '' && !VALID_AAS.includes(aaAlt)) {
        showToast('Invalid Alternate AA: "' + aaAlt + '" — try Ala, Val, Gly, Ter, Del, Fs, etc.', 'error');
        btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = i18n[currentLang]['analyze_btn'];
        return;
    }
    const geneCtx = document.getElementById('geneSelect').value;
    const maxCdna = GENE_MAX_CDNA[geneCtx] || 15000;

    if (cdnaValStr !== '' && (isNaN(cdnaVal) || cdnaVal < 1 || cdnaVal > maxCdna)) {
        showToast(currentLang === 'kk' ? `cDNA позициясы 1 мен ${maxCdna} аралығында болуы керек` : currentLang === 'ru' ? `Позиция cDNA должна быть от 1 до ${maxCdna}` : `cDNA position must be between 1 and ${maxCdna} (${geneCtx} gene length)`, 'error');
        btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = i18n[currentLang]['analyze_btn'];
        return;
    }

    // fallback check for missing required fields
    if (cdnaValStr === '') {
        showToast(currentLang === 'kk' ? `cDNA тізбегінің позициясын енгізіңіз (1 - ${maxCdna})` : currentLang === 'ru' ? `Пожалуйста, введите позицию последовательности cDNA (1 - ${maxCdna})` : `Please enter a cDNA sequence position (1 - ${maxCdna})`, 'warning');
        btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = i18n[currentLang]['analyze_btn'];
        return;
    }
    if (aaRef === '') {
        showToast(i18n[currentLang]['err_enter_ref'] || 'Please enter a Reference Amino Acid (e.g. Arg)', 'warning');
        btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = i18n[currentLang]['analyze_btn'];
        return;
    }
    if (aaAlt === '') {
        showToast(i18n[currentLang]['err_enter_alt'] || 'Please enter an Alternate Amino Acid (e.g. Cys)', 'warning');
        btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = i18n[currentLang]['analyze_btn'];
        return;
    }

    const body = {
        gene_name: document.getElementById('geneSelect').value,
        cDNA_pos: parseInt(document.getElementById('cDNA_pos').value),
        AA_ref: aaRef,
        AA_alt: aaAlt,
        Mutation: document.getElementById('mutation').value.trim() || 'Unknown',
    };

    try {
        const resp = await fetchWithTimeout(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }, 30000);

        if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
        const data = await resp.json();

        if (data.error) {
            showToast(data.error, 'error');
            btn.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = i18n[currentLang]['analyze_btn'];
            return;
        }

        const prob = data.probability;
        let tier;
        if (prob > 0.7) tier = 'pathogenic';
        else if (prob < 0.3) tier = 'benign';
        else tier = 'uncertain';

        resultCard.className = `result-card card-full ${tier}`;

        const badge = document.getElementById('badge');
        badge.textContent = tr(data.prediction);
        badge.className = `badge badge-${tier}`;

        const conf = document.getElementById('confidence');
        const ciData = data.confidence || {};
        if (ciData.ci_lower != null && ciData.ci_upper != null) {
            conf.innerHTML = escapeHtml((prob * 100).toFixed(1) + '%') +
                ' <span style="font-size:0.55em;color:var(--text-mid);font-weight:400">' +
                (ciData.method === 'bootstrap' ? '90' : '95') + '% CI: ' +
                escapeHtml((ciData.ci_lower * 100).toFixed(1)) + '-' +
                escapeHtml((ciData.ci_upper * 100).toFixed(1)) + '%</span>';
        } else {
            conf.textContent = (prob * 100).toFixed(1) + '%';
        }
        conf.className = `confidence conf-${tier}`;

        // Save variant identifiers for live lookups
        // Prefer backend-computed aa_pos (accounts for start codon offset correctly).
        // Fallback Math.ceil(cdna/3) is inaccurate (ignores start codon offset) — mark as approx.
        let aaPos = data.aa_pos || data.features_used?.aa_position || null;
        let aaPosApprox = false;
        if (aaPos == null) {
            aaPos = Math.ceil(parseInt(cdnaValStr) / 3);
            aaPosApprox = true;
        }
        const aaPosDisplay = aaPosApprox ? aaPos + ' (approx)' : aaPos;
        const theHgvs = "p." + aaRef + aaPos + aaAlt;
        document.getElementById('btnClinvar').setAttribute('data-variant', theHgvs);
        if (data.genomic_pos) {
            document.getElementById('btnGnomad').setAttribute('data-variant', data.genomic_pos);
        } else {
            // fallback if genomic pos isn't returned
            document.getElementById('btnGnomad').setAttribute('data-variant', theHgvs);
        }

        // Reset live result boxes whenever a new prediction is made
        document.getElementById('resClinvar').classList.remove('show');
        document.getElementById('resClinvar').innerHTML = '';
        document.getElementById('resGnomad').classList.remove('show');
        document.getElementById('resGnomad').innerHTML = '';

        // Store for re-rendering on language change
        lastPredictionData = data;
        lastPredictionInput = body;

        // Confidence interval display
        const ci = data.confidence || {};
        const ciDiv = document.getElementById('uncertaintyInfo');
        if (ci.label) {
            ciDiv.innerHTML = renderConfidenceHtml(ci);
            ciDiv.style.display = 'block';
        } else {
            ciDiv.style.display = 'none';
        }

        // Conformal Prediction Set (Item 5.1)
        const conformalDiv = document.getElementById('conformalPrediction');
        if (conformalDiv && data.conformal_prediction) {
            const cp = data.conformal_prediction;
            const setLabels = cp.conformal_set.map(c => escapeHtml(tr(c.toLowerCase()) || c)).join(', ');
            const coveragePct = Math.round(cp.conformal_coverage * 100);
            const isSingleton = cp.set_size === 1;
            const badgeColor = isSingleton ? '#10b981' : '#f59e0b';
            const badgeIcon = isSingleton
                ? '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="#fff" stroke-width="2.5" style="vertical-align:-2px;margin-right:3px"><path d="M20 6L9 17l-5-5"/></svg>'
                : '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="#fff" stroke-width="2" style="vertical-align:-2px;margin-right:3px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
            conformalDiv.innerHTML =
                '<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;color:#fff;background:' + badgeColor + '">' +
                badgeIcon +
                escapeHtml(tr('conformal_coverage_label') || (coveragePct + '% Coverage Guarantee')) +
                ': {' + setLabels + '}' +
                '</span>' +
                '<span class="help-tip" style="margin-left:4px" aria-label="What is this?">?<span class="help-text">' +
                escapeHtml(tr('conformal_tip') || 'Conformal prediction produces a set of classes guaranteed to contain the true class with ' + coveragePct + '% probability. Smaller sets = more informative predictions.') +
                '</span></span>';
            conformalDiv.style.display = 'block';
        } else if (conformalDiv) {
            conformalDiv.style.display = 'none';
        }

        // Gene reliability warning for low/moderate-performing genes
        const relDiv = document.getElementById('geneReliability');
        const rel = data.gene_reliability;
        if (rel && rel.tier && rel.tier !== 'high') {
            const tierLabel = rel.tier === 'low' ? 'Low' : 'Moderate';
            relDiv.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--warning)" stroke-width="2" style="vertical-align:-3px;margin-right:4px"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' +
                '<strong>' + escapeHtml(body.gene_name) + '</strong> ' +
                escapeHtml(tr('gene_reliability_warn')) + ': ' +
                escapeHtml(tierLabel) + ' (AUC: ' + escapeHtml(String(rel.auc)) + '). ' +
                escapeHtml(rel.note || '');
            relDiv.style.display = 'block';
        } else if (relDiv) {
            relDiv.style.display = 'none';
        }

        // Ensemble weights display (Item 38: gene-adaptive)
        const ewDiv = document.getElementById('ensembleWeights');
        if (ewDiv && data.ensemble_weights) {
            const ew = data.ensemble_weights;
            ewDiv.textContent = 'XGB ' + Math.round(ew.xgb * 100) + '% + MLP ' + Math.round(ew.mlp * 100) + '%';
            ewDiv.style.display = '';
        } else if (ewDiv) {
            ewDiv.style.display = 'none';
        }

        // Model disagreement warning (XGBoost vs MLP)
        const disagreeDiv = document.getElementById('modelDisagreement');
        if (disagreeDiv && data.model_disagreement != null && data.model_disagreement > 0.3) {
            disagreeDiv.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--warning)" stroke-width="2" style="vertical-align:-3px;margin-right:4px"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' +
                '<strong>Model Disagreement:</strong> Models show significant disagreement on this variant (delta: ' + escapeHtml(data.model_disagreement.toFixed(3)) + '). Interpret with caution.';
            disagreeDiv.style.display = 'block';
        } else if (disagreeDiv) {
            disagreeDiv.style.display = 'none';
        }

        // VUS guidance for uncertain predictions
        const vusDiv = document.getElementById('vusGuidance');
        if (data.risk_tier === 'uncertain' && vusDiv) {
            vusDiv.innerHTML = '<strong>VUS Guidance:</strong> ' + escapeHtml(tr('vus_guidance'));
            vusDiv.style.display = 'block';
        } else if (vusDiv) {
            vusDiv.style.display = 'none';
        }

        // Feature coverage indicator
        const fcDiv = document.getElementById('featureCoverage');
        if (fcDiv && data.feature_coverage) {
            const fc = data.feature_coverage;
            fcDiv.textContent = tr('feature_coverage_label') + ': ' + fc.nonzero + '/' + fc.total + ' (' + fc.percentage + '%)';
            fcDiv.style.display = '';
        } else if (fcDiv) {
            fcDiv.style.display = 'none';
        }

        // Data Scarcity Quantification badge (Item 41)
        const dsDiv = document.getElementById('dataSupport');
        if (dsDiv && data.data_support) {
            const ds = data.data_support;
            const levelColors = { HIGH: '#10b981', MODERATE: '#f59e0b', LOW: '#ef4444' };
            const levelColor = levelColors[ds.level] || '#6b7280';
            dsDiv.innerHTML = tr('data_support_label') + ': <span style="padding:1px 6px;border-radius:8px;font-weight:600;font-size:0.7rem;color:#fff;background:' + levelColor + '">' + escapeHtml(ds.level) + '</span>';
            dsDiv.style.display = '';
        } else if (dsDiv) {
            dsDiv.style.display = 'none';
        }

        // Show/hide metadata row
        const metaRow = document.getElementById('predictionMeta');
        if (metaRow) {
            const anyMeta = (ewDiv && ewDiv.style.display !== 'none') || (fcDiv && fcDiv.style.display !== 'none') || (dsDiv && dsDiv.style.display !== 'none');
            metaRow.style.display = anyMeta ? 'flex' : 'none';
        }

        // Data source cards (uses module-level helper)
        document.getElementById('dsGrid').innerHTML = renderDsGridHtml(data.data_sources || {});

        // Features grid (uses module-level helper)
        const f = data.features_used;
        document.getElementById('featuresGrid').innerHTML = renderFeaturesGridHtml(f);

        // ---- ACMG Evidence Board ----
        const acmgSection = document.getElementById('acmgSection');
        const acmgGrid = document.getElementById('acmgGrid');
        const acmgData = data.acmg_evidence || {};

        if (Object.keys(acmgData).length > 0) {
            acmgGrid.innerHTML = Object.entries(acmgData).map(([code, desc]) => {
                // PP3, PM1, PM5 etc are Pathogenic. BP4, BS1 etc are Benign.
                const isPathogenic = code.startsWith('P');
                const codeClass = isPathogenic ? 'pathogenic' : 'benign';
                return `
                    <div class="acmg-badge-row">
                        <div class="acmg-code ${codeClass}">${escapeHtml(code)}</div>
                        <div class="acmg-desc">${escapeHtml(desc)}</div>
                    </div>
                `;
            }).join('');
            acmgSection.style.display = 'block';
        } else {
            acmgSection.style.display = 'none';
        }

        // ---- SHAP Chart ----
        const shapSection = document.getElementById('shapSection');
        const shapChart = document.getElementById('shapChart');
        const shapToggleBtn = document.getElementById('shapToggleBtn');

        function renderShapBars(features, maxAbsOverride) {
            if (!features || features.length === 0) return '<p class="text-muted">No SHAP features available.</p>';
            const maxAbs = maxAbsOverride || Math.max(...features.map(s => Math.abs(s.value)), 0.01);
            return features.map(s => {
                const pct = Math.min((Math.abs(s.value) / maxAbs) * 50, 50);
                const dir = s.value > 0 ? 'pathogenic' : 'benign';
                const valClass = s.value > 0 ? 'positive' : 'negative';
                const sign = s.value > 0 ? '+' : '';
                return `
                    <div class="shap-bar-row">
                        <div class="shap-label">${escapeHtml(s.feature)}</div>
                        <div class="shap-bar-container">
                            <div class="shap-center-line"></div>
                            <div class="shap-bar ${dir}" style="width:${pct}%"></div>
                        </div>
                        <div class="shap-val ${valClass}">${sign}${s.value.toFixed(3)}</div>
                    </div>`;
            }).join('');
        }

        if (data.shap_explanation && data.shap_explanation.length > 0) {
            const allFeats = data.shap_all || data.shap_explanation;
            const topFeats = data.shap_explanation;
            const maxAbsAll = Math.max(...allFeats.map(s => Math.abs(s.value)), 0.01);

            shapChart.innerHTML = renderShapBars(topFeats, maxAbsAll);
            shapSection.style.display = 'block';

            // Toggle button for expanded view
            if (allFeats.length > topFeats.length) {
                let expanded = false;
                shapToggleBtn.style.display = 'inline-block';
                shapToggleBtn.textContent = i18n[currentLang]['shap_show_all'] || 'Show All Features (' + allFeats.length + ')';
                shapToggleBtn.onclick = () => {
                    expanded = !expanded;
                    shapChart.innerHTML = renderShapBars(expanded ? allFeats : topFeats, maxAbsAll);
                    shapToggleBtn.textContent = expanded
                        ? (i18n[currentLang]['shap_show_top'] || 'Show Top 8 Only')
                        : (i18n[currentLang]['shap_show_all'] || 'Show All Features (' + allFeats.length + ')');
                };
            } else {
                shapToggleBtn.style.display = 'none';
            }
        } else {
            shapSection.style.display = 'none';
            shapToggleBtn.style.display = 'none';
        }

        // ---- Contrastive Explanation (Item 43) ----
        const contrastiveSection = document.getElementById('contrastiveSection');
        const contrastiveContent = document.getElementById('contrastiveContent');
        if (contrastiveSection && contrastiveContent && data.contrastive_explanation) {
            const ce = data.contrastive_explanation;
            const ceVariant = escapeHtml(ce.contrast_variant);
            const ceClassLower = ce.contrast_class.toLowerCase();
            const ceDist = escapeHtml(String(ce.contrast_distance));

            // SVG icon: arrows pointing apart (divergence)
            const ceIconSvg = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>';

            // Header card: shows the contrast variant like an ACMG badge
            var ceHtml = '<div class="contrastive-header">' +
                '<div class="ce-icon ' + escapeHtml(ceClassLower) + '">' + ceIconSvg + '</div>' +
                '<div class="ce-meta">' +
                '<div class="ce-variant-name">' + ceVariant + '</div>' +
                '<div class="ce-variant-detail">' +
                (tr('contrastive_compared_to') || 'Nearest opposite-class training variant') +
                ' &middot; ' + (tr('contrastive_distance') || 'distance') + ': ' + ceDist +
                '</div></div>' +
                '<span class="ce-class-tag ' + escapeHtml(ceClassLower) + '">' + escapeHtml(ce.contrast_class) + '</span>' +
                '</div>';

            // Difference bars — same visual pattern as SHAP bars
            var maxDiff = 0.01;
            for (var di = 0; di < ce.key_differences.length; di++) {
                if (ce.key_differences[di].difference > maxDiff) maxDiff = ce.key_differences[di].difference;
            }

            for (var di = 0; di < ce.key_differences.length; di++) {
                var diff = ce.key_differences[di];
                var barPct = Math.min((diff.difference / maxDiff) * 85, 85);
                ceHtml += '<div class="ce-diff-row" style="--i:' + di + ';animation-delay:' + (0.04 * di) + 's">' +
                    '<div class="ce-diff-label">' + escapeHtml(diff.feature) + '</div>' +
                    '<div class="ce-diff-bar-wrap">' +
                    '<div class="ce-diff-bar ' + escapeHtml(diff.importance) + '" style="width:' + barPct + '%"></div>' +
                    '</div>' +
                    '<div class="ce-diff-val">' + escapeHtml(String(diff.difference)) + '</div>' +
                    '</div>';
            }

            contrastiveContent.innerHTML = ceHtml;
            contrastiveSection.style.display = 'block';
        } else if (contrastiveSection) {
            contrastiveSection.style.display = 'none';
        }

        // ---- PDF Export Button ----
        const pdfBtn = document.getElementById('pdfExportBtn');
        pdfBtn.style.display = 'inline-flex';
        pdfBtn.onclick = () => {
            if (pdfBtn.classList.contains('disabled')) return;
            pdfBtn.classList.add('disabled');
            pdfBtn.style.opacity = '0.5';
            pdfBtn.style.pointerEvents = 'none';
            try {
                generatePDFReport(data, body);
            } finally {
                setTimeout(() => {
                    pdfBtn.classList.remove('disabled');
                    pdfBtn.style.opacity = '';
                    pdfBtn.style.pointerEvents = '';
                }, 2000);
            }
        };

        // ---- Compare Button ----
        const cmpBtn = document.getElementById('compareBtn');
        cmpBtn.style.display = 'inline-flex';
        cmpBtn.onclick = () => saveForComparison(data, body, aaPos);

        // ---- JSON Export Button ----
        const jsonBtn = document.getElementById('jsonExportBtn');
        jsonBtn.style.display = 'inline-flex';
        jsonBtn.onclick = () => {
            const jsonStr = JSON.stringify(data, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `steppedna_${body.gene_name}_${body.AA_ref}${aaPos}${body.AA_alt}.json`;
            a.click();
            URL.revokeObjectURL(url);
        };

        // Render protein domain lollipop plot
        renderLollipopPlot(body.gene_name, aaPos, data.prediction);

        // Render UMAP variant landscape (async, non-blocking)
        loadAndRenderUMAP(prob);

        // Render 3D protein structure viewer (async, non-blocking)
        render3DViewer(body.gene_name, aaPos, data.prediction);

        if (skeletonCard) skeletonCard.style.display = 'none';
        resultCard.style.display = 'block';
        // Accessibility: move focus to result for keyboard/screen-reader users
        resultCard.setAttribute('tabindex', '-1');
        resultCard.focus({ preventScroll: true });
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Save to variant analysis history
        saveToHistory({
            gene: body.gene_name,
            cdna_pos: body.cDNA_pos,
            aa_ref: body.AA_ref,
            aa_alt: body.AA_alt,
            aa_pos: aaPos,
            prediction: data.prediction,
            probability: prob,
            timestamp: Date.now()
        });

    } catch (err) {
        if (skeletonCard) skeletonCard.style.display = 'none';
        const msg = err.name === 'AbortError'
            ? (i18n[currentLang]['err_timeout'] || 'Request timed out — the backend may be overloaded.')
            : (i18n[currentLang]['err_backend'] || 'Could not reach the backend — make sure the server is running.') + ' ' + err.message;
        showToast(msg, 'error', 6000);
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = i18n[currentLang]['analyze_btn'];
    }
});

// ---- LIVE DATABASE LOOKUPS ----
document.getElementById('btnClinvar').addEventListener('click', async function () {
    const variant = this.getAttribute('data-variant');
    if (!variant) return;
    if (isDebounced('clinvar')) return;

    const btn = this;
    const spinner = document.getElementById('spinClinvar');
    const resBox = document.getElementById('resClinvar');
    const gene = document.getElementById('geneSelect')?.value || 'BRCA2';

    btn.disabled = true;
    spinner.style.display = 'block';
    resBox.classList.add('show');
    resBox.innerHTML = '<span style="color:var(--text-muted)">Looking up ClinVar...</span>';

    try {
        const resp = await fetchWithTimeout(CLINVAR_API + encodeURIComponent(variant) + '?gene=' + encodeURIComponent(gene), {}, 15000);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (data.error || data.clinvar === "Not found") {
            resBox.innerHTML = `<strong>ClinVar:</strong> Not found (<a href="https://www.ncbi.nlm.nih.gov/clinvar/?term=${encodeURIComponent(variant)}" target="_blank" style="color:var(--primary)">Search Manually</a>)`;
        } else {
            // Highlight 'Conflicting interpretations'
            let sigStr = escapeHtml(data.clinical_significance);
            if (data.clinical_significance.toLowerCase().includes('conflicting')) {
                sigStr = `<span style="color:var(--warning);font-weight:700;">${sigStr}</span>`;
            }
            resBox.innerHTML = `
                <div style="margin-bottom:4px"><strong>Significance:</strong> <span style="color:var(--text-dark)">${sigStr}</span></div>
                <div style="margin-bottom:4px"><strong>Review Status:</strong> ${escapeHtml(data.review_status)}</div>
                <div><a href="https://www.ncbi.nlm.nih.gov/clinvar/RCV${escapeHtml(String(data.clinvar_id).padStart(9, '0'))}/" target="_blank" style="color:var(--primary);text-decoration:none;font-weight:600">View RCV Record ↗</a></div>
            `;
        }
    } catch (e) {
        showToast('ClinVar lookup failed: ' + e.message, 'error');
        resBox.innerHTML = `<span style="color:var(--danger)">Error connecting to ClinVar API.</span> <button onclick="document.getElementById('btnClinvar').click()" style="margin-left:8px;padding:2px 10px;border-radius:4px;border:1px solid var(--danger);background:transparent;color:var(--danger);cursor:pointer;font-size:0.8rem">Retry</button>`;
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
    }
});

document.getElementById('btnGnomad').addEventListener('click', async function () {
    const variant = this.getAttribute('data-variant');
    if (!variant) return;
    if (isDebounced('gnomad')) return;

    const btn = this;
    const spinner = document.getElementById('spinGnomad');
    const resBox = document.getElementById('resGnomad');

    // If variant is protein notation (not genomic coords), show helpful message instead of silent failure
    if (variant.startsWith('p.') || !/^\d/.test(variant)) {
        resBox.classList.add('show');
        resBox.innerHTML = `<span style="color:var(--warning)">Genomic coordinates not available for this variant. gnomAD requires chr-pos-ref-alt format. Provide a nucleotide mutation (e.g. A&gt;G) for genomic position mapping.</span>`;
        return;
    }

    btn.disabled = true;
    spinner.style.display = 'block';
    resBox.classList.add('show');
    resBox.innerHTML = '<span style="color:var(--text-muted)">Looking up gnomAD...</span>';

    try {
        const resp = await fetchWithTimeout(GNOMAD_API + encodeURIComponent(variant), {}, 15000);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (data.error || data.gnomad) {
            resBox.innerHTML = `<strong>gnomAD v4:</strong> ${escapeHtml(data.error || data.gnomad)}`;
        } else if (data.note) {
            resBox.innerHTML = `<span style="color:var(--warning)">${escapeHtml(data.note)}</span>`;
        } else {
            const af = data.genome_af !== null ? data.genome_af : data.exome_af;
            const ac = data.genome_ac !== null ? data.genome_ac : data.exome_ac;
            const hom = data.genome_hom !== null ? data.genome_hom : data.exome_hom;
            const freqFmt = af ? af.toExponential(3) : "0.00";
            resBox.innerHTML = `
                <div style="margin-bottom:4px"><strong>Allele Frequency:</strong> <span style="color:var(--text-dark)">${escapeHtml(freqFmt)}</span></div>
                <div style="margin-bottom:4px"><strong>Allele Count:</strong> ${escapeHtml(String(ac || 0))} (Hom: ${escapeHtml(String(hom || 0))})</div>
                <div><a href="https://gnomad.broadinstitute.org/variant/${encodeURIComponent(variant)}?dataset=gnomad_r4" target="_blank" style="color:var(--primary);text-decoration:none;font-weight:600">View gnomAD Record ↗</a></div>
            `;
        }
    } catch (e) {
        showToast('gnomAD lookup failed: ' + e.message, 'error');
        resBox.innerHTML = `<span style="color:var(--danger)">Error connecting to Broad Institute API.</span> <button onclick="document.getElementById('btnGnomad').click()" style="margin-left:8px;padding:2px 10px;border-radius:4px;border:1px solid var(--danger);background:transparent;color:var(--danger);cursor:pointer;font-size:0.8rem">Retry</button>`;
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
    }
});

// ---- PDF REPORT GENERATOR ----
function generatePDFReport(data, input) {
    const t = tr; // Use module-level translator
    const prob = data.probability;
    const pred = data.prediction;
    const ci = data.confidence || {};
    const ds = data.data_sources || {};
    const am = ds.alphamissense || {};
    const mave = ds.mave || {};
    const phy = ds.phylop || {};
    const struct = ds.structure || {};
    const f = data.features_used || {};
    const shap = data.shap_explanation || [];
    const acmg = data.acmg_evidence || {};

    const acmgRows = Object.entries(acmg).map(([code, desc]) =>
        `<tr><td style="font-weight:700;color:${code.startsWith('P') ? '#dc2626' : '#059669'};padding:4px 12px 4px 0">${code}</td><td style="padding:4px 0">${desc}</td></tr>`
    ).join('');

    const shapRows = shap.map(s =>
        `<tr><td style="padding:3px 12px 3px 0">${s.feature}</td><td style="padding:3px 0;color:${s.value > 0 ? '#dc2626' : '#059669'};font-weight:600">${s.value > 0 ? '+' : ''}${s.value.toFixed(4)}</td><td style="padding:3px 0">${s.direction}</td></tr>`
    ).join('');

    const ciText = (ci.ci_lower != null && ci.ci_upper != null)
        ? `<p style="margin-top:8px"><strong>95% Confidence Interval:</strong> ${(ci.ci_lower * 100).toFixed(1)}% - ${(ci.ci_upper * 100).toFixed(1)}%</p>`
        : '';

    const now = new Date().toISOString().split('T')[0];
    // Prefer backend-computed aa_pos; fallback formula is inaccurate (ignores start codon offset)
    let aaPos = data.aa_pos || f.aa_position || null;
    const aaPosApprox = (aaPos == null);
    if (aaPos == null) aaPos = Math.ceil(input.cDNA_pos / 3);
    const hgvs = 'p.' + input.AA_ref + aaPos + input.AA_alt + (aaPosApprox ? ' (approx)' : '');

    const html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>SteppeDNA Variant Report - ${hgvs}</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a2e; padding: 40px; max-width: 800px; margin: 0 auto; font-size: 13px; line-height: 1.6; }
    .header { border-bottom: 3px solid #6260FF; padding-bottom: 16px; margin-bottom: 24px; }
    .header h1 { font-size: 22px; color: #6260FF; margin-bottom: 4px; }
    .header p { color: #666; font-size: 11px; }
    .disclaimer { background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 6px; padding: 10px 14px; font-size: 11px; color: #92400E; margin-bottom: 20px; }
    .result-box { background: ${prob > 0.7 ? '#FEE2E2' : (prob < 0.3 ? '#D1FAE5' : '#FEF3C7')}; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 24px; border: 1px solid ${prob > 0.7 ? '#FECACA' : (prob < 0.3 ? '#A7F3D0' : '#FDE68A')}; }
    .result-box .pred { font-size: 24px; font-weight: 800; color: ${prob > 0.7 ? '#dc2626' : (prob < 0.3 ? '#059669' : '#d97706')}; }
    .result-box .prob { font-size: 16px; color: #374151; margin-top: 4px; }
    h2 { font-size: 15px; color: #6260FF; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; margin: 20px 0 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .info-table td { padding: 5px 0; border-bottom: 1px solid #f3f4f6; }
    .info-table td:first-child { font-weight: 600; width: 180px; color: #4b5563; }
    .footer { margin-top: 30px; padding-top: 12px; border-top: 1px solid #e5e7eb; font-size: 10px; color: #9ca3af; text-align: center; }
    @media print { body { padding: 20px; } .disclaimer, .result-box { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
</style></head><body>

<div class="header">
    <h1>SteppeDNA Variant Report</h1>
    <p>Pan-Gene HR Pathway Variant Classifier &bull; Generated: ${now} &bull; Research Use Only</p>
</div>

<div class="disclaimer" style="background:#FEE2E2;border-color:#EF4444;color:#991B1B;font-weight:600;text-align:center">
    RESEARCH USE ONLY (RUO) &mdash; NOT FOR CLINICAL DIAGNOSTIC USE
</div>
<div class="disclaimer">
    This report is generated by SteppeDNA, a computational research tool. It has NOT been validated for clinical diagnostic use and must NOT be used as the sole basis for medical decisions. All classifications are computational approximations that require independent expert review. ACMG evidence codes are approximate and have not been validated by clinical geneticists.
</div>

<div class="result-box">
    <div class="pred">${pred}</div>
    <div class="prob">Pathogenicity Score: ${(prob * 100).toFixed(1)}%${ci.ci_lower != null ? ' (' + (ci.method === 'bootstrap' ? '90' : '95') + '% CI: ' + (ci.ci_lower * 100).toFixed(1) + '%-' + (ci.ci_upper * 100).toFixed(1) + '%)' : ''}</div>
    ${ci.label ? '<div style="margin-top:6px;font-size:12px;color:#6b7280">' + ci.label + (ci.method === 'bootstrap' ? ' (Bootstrap, ' + (ci.n_models || 50) + ' models)' : '') + '</div>' : ''}
</div>

<h2>Variant Information</h2>
<table class="info-table">
    <tr><td>Gene</td><td>${input.gene_name}</td></tr>
    <tr><td>HGVS Protein</td><td>${hgvs}</td></tr>
    <tr><td>cDNA Position</td><td>c.${input.cDNA_pos}</td></tr>
    <tr><td>Amino Acid Change</td><td>${input.AA_ref} &rarr; ${input.AA_alt}</td></tr>
    <tr><td>Nucleotide Change</td><td>${input.Mutation || 'Not specified'}</td></tr>
    <tr><td>Classification</td><td>${pred} (${(prob * 100).toFixed(1)}%)</td></tr>
</table>
${ciText}

<h2>Data Source Scores</h2>
<table class="info-table">
    <tr><td>AlphaMissense</td><td>${am.score != null ? am.score + ' (' + (am.label || '-') + ')' : 'No data'}</td></tr>
    <tr><td>MAVE Functional Assay</td><td>${mave.score != null ? mave.score + ' (' + (mave.label || '-') + ')' : 'No data'}</td></tr>
    <tr><td>PhyloP Conservation</td><td>${phy.score != null ? phy.score + ' (' + (phy.label || '-') + ')' : 'No data'}</td></tr>
    <tr><td>3D Structure Domain</td><td>${struct.domain || 'Unknown'}</td></tr>
    <tr><td>Secondary Structure</td><td>${struct.secondary_structure || '-'}</td></tr>
    <tr><td>Solvent Accessibility</td><td>${struct.is_buried ? 'Buried' : 'Surface'} (RSA: ${struct.rsa || '-'})</td></tr>
    <tr><td>DNA Contact</td><td>${struct.is_dna_contact ? 'Yes' : 'No'}</td></tr>
</table>

<h2>Biochemical Features</h2>
<table class="info-table">
    <tr><td>BLOSUM62 Score</td><td>${f.blosum62_score}</td></tr>
    <tr><td>Volume Difference</td><td>${f.volume_diff}</td></tr>
    <tr><td>Hydrophobicity Diff</td><td>${f.hydro_diff}</td></tr>
    <tr><td>Charge Changed</td><td>${f.charge_changed ? 'Yes' : 'No'}</td></tr>
    <tr><td>Critical Domain</td><td>${f.in_critical_domain ? 'Yes' : 'No'}</td></tr>
    <tr><td>Nonsense Mutation</td><td>${f.is_nonsense ? 'Yes' : 'No'}</td></tr>
</table>

${acmgRows ? '<h2>ACMG Evidence Codes</h2><table class="info-table">' + acmgRows + '</table>' : ''}

${shapRows ? '<h2>SHAP Feature Attribution (Top 8)</h2><table class="info-table"><tr style="font-weight:700;border-bottom:2px solid #e5e7eb"><td style="padding:4px 12px 4px 0">Feature</td><td style="padding:4px 0">SHAP Value</td><td style="padding:4px 0">Direction</td></tr>' + shapRows + '</table>' : ''}

<h2>Model Information</h2>
<table class="info-table">
    <tr><td>Model</td><td>XGBoost (60%) + MLP (40%) Ensemble</td></tr>
    <tr><td>Calibration</td><td>Isotonic regression on held-out data</td></tr>
    <tr><td>Features</td><td>103 engineered features from 5 databases</td></tr>
    <tr><td>Training Data</td><td>19,223 variants (ClinVar + gnomAD)</td></tr>
    <tr><td>Validation ROC-AUC</td><td>0.978</td></tr>
    <tr><td>Validation MCC</td><td>0.881</td></tr>
    <tr><td>Genes Covered</td><td>BRCA1, BRCA2, PALB2, RAD51C, RAD51D</td></tr>
</table>

<h2>Classification Summary</h2>
<div class="result-box" style="margin-top:12px">
    <div style="font-size:13px;font-weight:700;color:#374151;margin-bottom:6px">Final Classification</div>
    <div class="pred">${pred}</div>
    <div class="prob">${(prob * 100).toFixed(1)}% Pathogenicity Score</div>
    ${ci.label ? '<div style="margin-top:6px;font-size:12px;color:#6b7280">' + ci.label + ' (CI: ' + (ci.ci_lower != null ? (ci.ci_lower * 100).toFixed(1) + '%-' + (ci.ci_upper * 100).toFixed(1) + '%' : 'N/A') + ')</div>' : ''}
</div>

<div class="disclaimer" style="margin-top:16px;background:#FEE2E2;border-color:#EF4444;color:#991B1B">
    <strong>RESEARCH USE ONLY</strong> &mdash; This report must NOT be used as the sole basis for clinical decisions about patient care. All classifications require independent expert review and should be corroborated by functional assays and family segregation data.
</div>

<div class="footer">
    SteppeDNA v5.2 &mdash; Pan-Gene Variant Classifier &bull; Research Use Only &mdash; Not a diagnostic tool<br>
    ACMG evidence codes are computational approximations and do not replace expert clinical curation.<br>
    Training data predominantly European ancestry. Performance on other populations unknown.
</div>

</body></html>`;

    const printWindow = window.open('', '_blank', 'width=800,height=1000');
    if (!printWindow || printWindow.closed || typeof printWindow.closed === 'undefined') {
        showToast('Popup blocked — please allow popups for this site to generate PDF reports.', 'warning', 6000);
        return;
    }
    printWindow.document.write(html);
    printWindow.document.close();
    setTimeout(() => printWindow.print(), 500);
}

// ─── UMAP Variant Landscape ──────────────────────────────────────────────────
let umapData = null;

async function loadAndRenderUMAP(queryProb) {
    const section = document.getElementById('umapSection');
    const canvas = document.getElementById('umapCanvas');
    if (!section || !canvas) return;

    try {
        if (!umapData) {
            const resp = await fetchWithTimeout(UMAP_API, {}, 10000);
            if (!resp.ok) { section.style.display = 'none'; return; }
            umapData = await resp.json();
        }
        if (!umapData.points || umapData.points.length === 0) { section.style.display = 'none'; return; }

        section.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const pad = 20;
        ctx.clearRect(0, 0, W, H);

        // Draw background points
        const pts = umapData.points;
        for (const p of pts) {
            const x = pad + p.x * (W - 2 * pad);
            const y = pad + p.y * (H - 2 * pad);
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fillStyle = p.l === 1 ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)';
            ctx.fill();
        }

        // Draw query variant as a larger golden dot at approximate position
        // Simple heuristic: place based on probability (pathogenic = closer to pathogenic cluster)
        // For real projection, the backend would need to project the new variant through the same UMAP transform
        // This is an approximation for visual effect
        const qx = pad + queryProb * (W - 2 * pad) * 0.6 + (W - 2 * pad) * 0.2;
        const qy = pad + (1 - queryProb) * (H - 2 * pad) * 0.5 + (H - 2 * pad) * 0.25;
        ctx.beginPath();
        ctx.arc(qx, qy, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        // Glow effect
        ctx.beginPath();
        ctx.arc(qx, qy, 12, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(245,158,11,0.3)';
        ctx.lineWidth = 3;
        ctx.stroke();
    } catch (e) {
        section.style.display = 'none';
    }
}

// ─── Variant Comparison Mode ─────────────────────────────────────────────────
let comparisonSlots = [null, null];

function saveForComparison(data, body, aaPos) {
    // Fill first empty slot, or overwrite slot 0 if both full
    const slotIdx = comparisonSlots[0] === null ? 0 : (comparisonSlots[1] === null ? 1 : 0);
    comparisonSlots[slotIdx] = {
        gene: body.gene_name,
        variant: `${body.AA_ref}${aaPos}${body.AA_alt}`,
        prediction: data.prediction,
        probability: data.probability,
        shap: (data.shap_explanation || []).slice(0, 5),
        features: data.features_used || {},
        ds: data.data_sources || {},
    };
    renderComparison();
}

function renderComparison() {
    const panel = document.getElementById('comparisonPanel');
    if (!panel) return;
    const [a, b] = comparisonSlots;
    if (!a && !b) { panel.style.display = 'none'; return; }
    panel.style.display = 'block';

    const renderSlot = (s, idx) => {
        if (!s) return `<div class="cmp-slot cmp-empty">Slot ${idx + 1}: Analyze a variant, then click "Compare"</div>`;
        const tierClass = s.probability > 0.7 ? 'pathogenic' : (s.probability < 0.3 ? 'benign' : 'uncertain');
        const pct = (s.probability * 100).toFixed(1);
        const shapHtml = s.shap.map(sh =>
            `<div class="cmp-shap-row"><span>${escapeHtml(sh.feature)}</span><span style="color:${sh.direction === 'pathogenic' ? 'var(--danger)' : 'var(--success)'}">${sh.value > 0 ? '+' : ''}${sh.value.toFixed(3)}</span></div>`
        ).join('');
        return `<div class="cmp-slot">
            <div class="cmp-header ${tierClass}">
                <strong>${escapeHtml(s.gene)} ${escapeHtml(s.variant)}</strong>
                <span class="cmp-pct">${pct}% ${escapeHtml(s.prediction)}</span>
            </div>
            <div class="cmp-shap">${shapHtml}</div>
            <button class="cmp-clear-btn" onclick="comparisonSlots[${idx}]=null;renderComparison();">Remove</button>
        </div>`;
    };

    const cmpTitle = (typeof i18n !== 'undefined' && i18n[currentLang] && i18n[currentLang]['cmp_title']) ? i18n[currentLang]['cmp_title'] : 'Variant Comparison';
    panel.innerHTML = `
        <div class="cmp-title">${escapeHtml(cmpTitle)}</div>
        <div class="cmp-grid">${renderSlot(a, 0)}${renderSlot(b, 1)}</div>
    `;
}

// ─── Protein Domain Lollipop Plot ────────────────────────────────────────────
const GENE_DOMAINS = {
    BRCA2: { len: 3418, domains: [
        { name: 'PALB2 Interaction', start: 10, end: 40, color: '#6366f1' },
        { name: 'BRC Repeats', start: 1009, end: 2083, color: '#f59e0b' },
        { name: 'DNA Binding', start: 2402, end: 3190, color: '#ef4444' },
        { name: 'OB Folds', start: 2670, end: 3102, color: '#10b981' },
        { name: 'NLS Nuclear Localization', start: 3263, end: 3330, color: '#8b5cf6' },
    ]},
    BRCA1: { len: 1863, domains: [
        { name: 'RING', start: 8, end: 96, color: '#ef4444' },
        { name: 'BARD1 Interaction', start: 8, end: 96, color: '#a855f7' },
        { name: 'DNA Binding', start: 452, end: 1079, color: '#3b82f6' },
        { name: 'SCD', start: 1280, end: 1524, color: '#6366f1' },
        { name: 'PALB2 Interaction', start: 1364, end: 1437, color: '#14b8a6' },
        { name: 'BRCT1', start: 1646, end: 1736, color: '#f59e0b' },
        { name: 'BRCT2', start: 1760, end: 1855, color: '#10b981' },
    ]},
    PALB2: { len: 1186, domains: [
        { name: 'BRCA1 Interaction', start: 9, end: 42, color: '#ef4444' },
        { name: 'ChAM DNA Binding', start: 395, end: 442, color: '#f59e0b' },
        { name: 'WD40 Repeats', start: 853, end: 1186, color: '#10b981' },
        { name: 'BRCA2 Interaction', start: 1085, end: 1186, color: '#6366f1' },
    ]},
    RAD51C: { len: 376, domains: [
        { name: 'Holliday Junction Resolution', start: 1, end: 126, color: '#6366f1' },
        { name: 'RAD51B RAD51D XRCC3 Interaction', start: 79, end: 136, color: '#a855f7' },
        { name: 'Walker A', start: 135, end: 142, color: '#ef4444' },
        { name: 'Walker B', start: 238, end: 242, color: '#f59e0b' },
        { name: 'NLS Nuclear Localization', start: 366, end: 370, color: '#8b5cf6' },
    ]},
    RAD51D: { len: 328, domains: [
        { name: 'ssDNA Binding', start: 1, end: 83, color: '#6366f1' },
        { name: 'N Terminal Domain', start: 2, end: 28, color: '#a855f7' },
        { name: 'Walker A', start: 108, end: 115, color: '#ef4444' },
        { name: 'Walker B', start: 206, end: 210, color: '#f59e0b' },
    ]},
};

function renderLollipopPlot(gene, aaPos, prediction) {
    const container = document.getElementById('lollipopPlot');
    if (!container) return;
    const gd = GENE_DOMAINS[gene];
    if (!gd) { container.innerHTML = ''; return; }

    // Responsive: scale font based on container width
    const containerPx = container.clientWidth || 800;
    const fontScale = Math.max(0.7, Math.min(1.0, containerPx / 800));

    const W = 800, pad = 40, scaleMarkerY = 56, trackY = 70, trackH = 14;
    const scale = (pos) => pad + (pos / gd.len) * (W - 2 * pad);
    const pinX = scale(aaPos);
    const pinColor = prediction === 'Pathogenic' ? '#ef4444' : (prediction === 'Benign' ? '#10b981' : '#f59e0b');

    // Collect label bounding boxes for collision detection before building SVG
    const labelInfos = [];
    for (let di = 0; di < gd.domains.length; di++) {
        const d = gd.domains[di];
        const x1 = scale(d.start), x2 = scale(d.end);
        const midX = (x1 + x2) / 2;
        const domainPxWidth = x2 - x1;
        const isNarrow = domainPxWidth < 15;
        const fontSize = Math.round((domainPxWidth > 50 ? 9 : 7) * fontScale);
        const charWidth = fontSize * 0.6;
        const labelW = d.name.length * charWidth;
        labelInfos.push({ di, midX, x1, x2, domainPxWidth, isNarrow, fontSize, labelW, name: d.name, color: d.color, level: 0 });
    }

    // Collision detection: assign stagger levels so no labels overlap
    const maxLevels = Math.max(6, gd.domains.length); // scale with domain count
    const staggerSpacing = 13;
    const baseLabelY = trackY + trackH + 14;
    for (let i = 0; i < labelInfos.length; i++) {
        const li = labelInfos[i];
        const liLeft = li.midX - li.labelW / 2;
        const liRight = li.midX + li.labelW / 2;
        for (let lvl = 0; lvl < maxLevels; lvl++) {
            let collides = false;
            for (let j = 0; j < i; j++) {
                const lj = labelInfos[j];
                if (lj.level !== lvl) continue;
                const ljLeft = lj.midX - lj.labelW / 2;
                const ljRight = lj.midX + lj.labelW / 2;
                if (liLeft < ljRight + 6 && liRight > ljLeft - 6) {
                    collides = true;
                    break;
                }
            }
            if (!collides) { li.level = lvl; break; }
        }
    }

    // Compute dynamic SVG height based on max stagger level used
    const maxLevel = labelInfos.reduce((mx, li) => Math.max(mx, li.level), 0);
    const H = baseLabelY + (maxLevel + 1) * staggerSpacing + 8;

    // Now build SVG with computed height
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:${W}px">`;
    // Track backbone
    svg += `<rect x="${pad}" y="${trackY}" width="${W - 2 * pad}" height="${trackH}" rx="4" fill="var(--border)" opacity="0.5"/>`;

    // Domain rects
    for (const li of labelInfos) {
        svg += `<rect x="${li.x1}" y="${trackY - 2}" width="${li.x2 - li.x1}" height="${trackH + 4}" rx="4" fill="${li.color}" opacity="0.7"/>`;
    }

    // Domain labels with collision-aware stagger + edge clamping
    for (const li of labelInfos) {
        const labelY = baseLabelY + li.level * staggerSpacing;
        if (li.isNarrow) {
            // Narrow domain: tick mark + leader line to offset label
            const tickTop = trackY + trackH + 2;
            const tickBot = labelY - li.fontSize;
            svg += `<line x1="${li.midX}" y1="${tickTop}" x2="${li.midX}" y2="${tickBot}" stroke="${li.color}" stroke-width="1" opacity="0.6"/>`;
            svg += `<circle cx="${li.midX}" cy="${tickTop}" r="1.5" fill="${li.color}" opacity="0.6"/>`;
        }
        // Clamp labels that would overflow SVG edges
        let anchor = 'middle', labelX = li.midX;
        const halfW = li.labelW / 2;
        if (li.midX + halfW > W - 4) {
            anchor = 'end'; labelX = W - 4;
        } else if (li.midX - halfW < 4) {
            anchor = 'start'; labelX = 4;
        }
        svg += `<text x="${labelX}" y="${labelY}" text-anchor="${anchor}" font-size="${li.fontSize}" fill="var(--text-body)" font-family="system-ui">${li.name}</text>`;
    }

    // Lollipop pin
    svg += `<line x1="${pinX}" y1="${trackY - 4}" x2="${pinX}" y2="18" stroke="${pinColor}" stroke-width="2"/>`;
    svg += `<circle cx="${pinX}" cy="14" r="7" fill="${pinColor}" stroke="#fff" stroke-width="1.5"/>`;
    svg += `<text x="${pinX}" y="6" text-anchor="middle" font-size="${Math.round(9 * fontScale)}" fill="var(--text-dark)" font-weight="600" font-family="system-ui">${gene} p.${aaPos}</text>`;
    // Scale markers — placed just above the track bar
    svg += `<text x="${pad}" y="${scaleMarkerY}" text-anchor="start" font-size="${Math.round(8 * fontScale)}" fill="var(--text-body)" opacity="0.5" font-family="system-ui">1</text>`;
    svg += `<text x="${W - pad}" y="${scaleMarkerY}" text-anchor="end" font-size="${Math.round(8 * fontScale)}" fill="var(--text-body)" opacity="0.5" font-family="system-ui">${gd.len}</text>`;
    svg += '</svg>';
    container.innerHTML = svg;
    container.style.display = 'block';
}

// ─── Init History on Load ───────────────────────────────────────────────────
(function initHistory() {
    // Render history on first load; also wire up clear button
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            renderHistory();
            const clearBtn = document.getElementById('clearHistoryBtn');
            if (clearBtn) clearBtn.addEventListener('click', clearHistory);
        });
    } else {
        renderHistory();
        const clearBtn = document.getElementById('clearHistoryBtn');
        if (clearBtn) clearBtn.addEventListener('click', clearHistory);
    }
})();

// ─── Language Change Listener ───────────────────────────────────────────────
// Re-renders dynamic content (result card, history) when user switches language
window.addEventListener('langchange', () => {
    updateResultTranslations();
    renderHistory();
});

// ─── 3D Protein Viewer (NGL.js) ─────────────────────────────────────────────
// Dynamically loads NGL.js and renders AlphaFold structures with mutation highlighting.

const UNIPROT_IDS = {
    BRCA1: 'P38398',
    BRCA2: 'P51587',
    PALB2: 'Q86YC2',
    RAD51C: 'O43502',
    RAD51D: 'O75771'
};

let nglLoaded = false;
let nglLoadPromise = null;
let nglStage = null;

function loadNGL() {
    if (nglLoaded && window.NGL) return Promise.resolve();
    if (nglLoadPromise) return nglLoadPromise;
    nglLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/ngl@2.3.1/dist/ngl.js';
        script.onload = () => { nglLoaded = true; resolve(); };
        script.onerror = () => reject(new Error('Failed to load NGL.js'));
        document.head.appendChild(script);
    });
    return nglLoadPromise;
}

async function render3DViewer(geneName, aaPos, prediction) {
    const section = document.getElementById('viewer3dSection');
    const loading = document.getElementById('viewer3dLoading');
    const errorDiv = document.getElementById('viewer3dError');
    const viewport = document.getElementById('viewer3dViewport');
    if (!section || !viewport) return;

    // BRCA2: Use local AlphaFold fragments (12 overlapping fragments, stride 200, 1400 AA each)
    // AlphaFold doesn't host BRCA2 individually — we extracted fragments from the proteome archive
    const BRCA2_FRAGMENTS = [
        { frag: 1,  start: 1,    end: 1400 },
        { frag: 2,  start: 201,  end: 1600 },
        { frag: 3,  start: 401,  end: 1800 },
        { frag: 4,  start: 601,  end: 2000 },
        { frag: 5,  start: 801,  end: 2200 },
        { frag: 6,  start: 1001, end: 2400 },
        { frag: 7,  start: 1201, end: 2600 },
        { frag: 8,  start: 1401, end: 2800 },
        { frag: 9,  start: 1601, end: 3000 },
        { frag: 10, start: 1801, end: 3200 },
        { frag: 11, start: 2001, end: 3400 },
        { frag: 12, start: 2201, end: 3418 },
    ];
    let brca2Frag = null;
    if (geneName === 'BRCA2') {
        // Find fragment that covers the mutation position (prefer smallest fragment centered on position)
        brca2Frag = BRCA2_FRAGMENTS.find(f => aaPos >= f.start && aaPos <= f.end);
        if (!brca2Frag) brca2Frag = BRCA2_FRAGMENTS[0];
    }

    const uniprotId = UNIPROT_IDS[geneName];
    if (!uniprotId) {
        section.style.display = 'none';
        return;
    }

    // Show section with loading state
    section.style.display = 'block';
    loading.style.display = 'flex';
    errorDiv.style.display = 'none';
    viewport.innerHTML = '';

    // Dispose previous stage if any
    if (nglStage) {
        nglStage.dispose();
        nglStage = null;
    }

    try {
        // Load NGL.js dynamically
        await loadNGL();

        // Create NGL Stage
        nglStage = new NGL.Stage(viewport, {
            backgroundColor: 'black',
            quality: 'medium',
            impostor: true,
            clipDist: 0,
        });

        // Handle window resize
        const handleResize = () => { if (nglStage) nglStage.handleResize(); };
        window.addEventListener('resize', handleResize);

        let component = null;
        let structureNote = '';

        if (brca2Frag) {
            // BRCA2: Load from local AlphaFold fragment served by backend
            const fragUrl = `${API_BASE}/structure/brca2/${brca2Frag.frag}`;
            component = await nglStage.loadFile(fragUrl, { ext: 'pdb' });
            if (!component) throw new Error('BRCA2 fragment unavailable');
            structureNote = `AlphaFold fragment F${brca2Frag.frag} (residues ${brca2Frag.start}–${brca2Frag.end})`;
        } else {
            // Non-BRCA2: Load from AlphaFold (v6 → v4 → v3)
            const baseUrl = `https://alphafold.ebi.ac.uk/files/AF-${uniprotId}-F1`;
            const versionFallback = ['v6', 'v4', 'v3'];
            for (const ver of versionFallback) {
                const pdbUrl = `${baseUrl}-model_${ver}.pdb`;
                try {
                    component = await nglStage.loadFile(pdbUrl, { ext: 'pdb' });
                    break;
                } catch (e) {
                    console.warn(`[SteppeDNA] AlphaFold model_${ver} failed for ${uniprotId}, trying next...`);
                }
            }
            if (!component) throw new Error(`AlphaFold structure unavailable for ${uniprotId}`);
        }

        // Hide loading indicator
        loading.style.display = 'none';

        // Cartoon representation colored by secondary structure
        component.addRepresentation('cartoon', {
            colorScheme: 'sstruc',
            opacity: 0.85,
            quality: 'medium',
        });

        // Highlight the mutation position as red ball+stick
        // BRCA2 fragments use internal PDB numbering (1-based), so offset from fragment start
        const pdbResNum = brca2Frag ? (aaPos - brca2Frag.start + 1) : aaPos;
        const selStr = String(pdbResNum);
        component.addRepresentation('ball+stick', {
            sele: selStr,
            color: '#ef4444',
            aspectRatio: 2.0,
            radiusScale: 1.5,
        });

        // Add a transparent red surface around the mutation site for emphasis
        component.addRepresentation('spacefill', {
            sele: selStr,
            color: '#ef4444',
            opacity: 0.4,
            radiusScale: 1.2,
        });

        // Auto-center on the whole structure
        component.autoView();

        // Show structure source note
        if (structureNote) {
            const fragNote = document.createElement('div');
            fragNote.style.cssText = 'text-align:center;font-size:0.75rem;color:var(--text-mid);margin-top:4px;';
            fragNote.textContent = structureNote;
            viewport.parentNode.insertBefore(fragNote, viewport.nextSibling);
        }

        // Slow auto-rotation
        let isSpinning = true;
        nglStage.setSpin(true);
        nglStage.spinAnimation.axis.set(0, 1, 0);
        nglStage.spinAnimation.angle = 0.005; // radians per frame, slow

        // Pause/Resume spin button
        const spinBtn = document.createElement('button');
        spinBtn.className = 'shap-toggle-btn';
        spinBtn.style.cssText = 'margin:6px auto 0;display:block;font-size:0.75rem;padding:4px 14px;';
        spinBtn.textContent = 'Pause Rotation';
        spinBtn.addEventListener('click', () => {
            isSpinning = !isSpinning;
            nglStage.setSpin(isSpinning);
            spinBtn.textContent = isSpinning ? 'Pause Rotation' : 'Resume Rotation';
        });
        viewport.parentNode.insertBefore(spinBtn, viewport.nextSibling);

    } catch (err) {
        console.error('[SteppeDNA] 3D viewer error:', err);
        loading.style.display = 'none';
        errorDiv.style.display = 'block';
        errorDiv.textContent = (i18n[currentLang] && i18n[currentLang]['viewer3d_error_load']) || 'Could not load the 3D structure for this gene. The protein may be too large for browser rendering.';
        viewport.innerHTML = '';
    }
}


// ─── Research Priorities (Active Learning, Item 42) ──────────────────────────
let _researchPrioritiesData = null;
let _researchActiveGene = null;

async function loadResearchPriorities() {
    const section = document.getElementById('researchPriorities');
    if (!section) return;
    try {
        const resp = await fetchWithTimeout(API_URL.replace('/predict', '/research/priorities') + '?limit=10', {}, 10000);
        if (!resp.ok) { section.style.display = 'none'; return; }
        const data = await resp.json();
        if (data.error || !data.priorities) { section.style.display = 'none'; return; }
        _researchPrioritiesData = data;
        renderResearchPriorities(data);
        section.style.display = 'block';
    } catch (e) {
        console.warn('[SteppeDNA] Research priorities not available:', e.message);
        section.style.display = 'none';
    }
}

function renderResearchPriorities(data) {
    const filterDiv = document.getElementById('researchGeneFilter');
    const tbody = document.getElementById('researchBody');
    if (!filterDiv || !tbody) return;

    const priorities = data.priorities || {};
    const genes = Object.keys(priorities).sort();

    // Gene filter buttons
    filterDiv.innerHTML = genes.map(g => {
        const count = priorities[g] ? priorities[g].length : 0;
        const active = _researchActiveGene === g ? ' active' : '';
        return '<button class="rp-gene-btn' + active + '" onclick="filterResearchGene(\'' + escapeHtml(g) + '\')">' +
            escapeHtml(g) + ' (' + count + ')</button>';
    }).join('');

    // Determine which gene to show
    const activeGene = _researchActiveGene || genes[0];
    const entries = priorities[activeGene] || [];

    tbody.innerHTML = entries.map(function(e, i) {
        const probPct = (e.current_prediction * 100).toFixed(1);
        const probColor = e.current_prediction > 0.7 ? '#ef4444' : (e.current_prediction < 0.3 ? '#10b981' : '#f59e0b');
        const prioColor = e.priority_score > 0.01 ? '#ef4444' : (e.priority_score > 0.001 ? '#f59e0b' : '#6b7280');
        return '<tr>' +
            '<td>' + (i + 1) + '</td>' +
            '<td style="font-family:monospace;font-size:0.82rem">' + escapeHtml(e.variant) + '</td>' +
            '<td style="color:' + prioColor + ';font-weight:600">' + escapeHtml(e.priority_score.toFixed(6)) + '</td>' +
            '<td>' + escapeHtml(e.qbc_score.toFixed(4)) + '</td>' +
            '<td style="color:' + probColor + '">' + escapeHtml(probPct) + '%</td>' +
            '<td>' + escapeHtml(String(e.nearby_training)) + '</td>' +
            '<td style="font-size:0.78rem;color:var(--text-mid)">' + escapeHtml(e.reason) + '</td>' +
            '</tr>';
    }).join('');
}

function filterResearchGene(gene) {
    _researchActiveGene = gene;
    if (_researchPrioritiesData) {
        renderResearchPriorities(_researchPrioritiesData);
    }
}

// Load research priorities on page load (non-blocking)
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(loadResearchPriorities, 1500);
});
