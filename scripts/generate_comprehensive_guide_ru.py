"""
SteppeDNA вЂ” Comprehensive Project Guide (PDF Generator)
========================================================
Generates a richly-formatted PDF document that explains every aspect
of the SteppeDNA project: architecture, codebase metrics, ML pipeline,
features, external data, APIs, frontend, tests, and deployment.

Usage:
    python scripts/generate_comprehensive_guide.py
"""

import os, sys, textwrap, datetime

# в”Ђв”Ђ PDF library в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("reportlab not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable,
    )

# в”Ђв”Ђ Colours в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
BRAND       = HexColor("#6260FF")
BRAND_DARK  = HexColor("#4e4cdf")
BRAND_LIGHT = HexColor("#d9d9fc")
DARK_BG     = HexColor("#1a1a2e")
TEXT_DARK    = HexColor("#222244")
TEXT_BODY    = HexColor("#555577")
LIGHT_BG    = HexColor("#f5f5ff")
WHITE       = white
DANGER      = HexColor("#EF4444")
SUCCESS     = HexColor("#22C55E")

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "SteppeDNA_Comprehensive_Guide_RU.pdf")

# в”Ђв”Ђ Styles в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
styles = getSampleStyleSheet()

def _ps(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=styles[parent], **kw)

sTitle = _ps("CoverTitle", fontSize=32, leading=40, textColor=BRAND,
             fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=8)
sSubtitle = _ps("CoverSub", fontSize=14, leading=20, textColor=TEXT_BODY,
                fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)
sH1 = _ps("H1", fontSize=22, leading=28, textColor=BRAND,
          fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=10,
          borderWidth=0, borderPadding=0)
sH2 = _ps("H2", fontSize=16, leading=22, textColor=BRAND_DARK,
          fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
sH3 = _ps("H3", fontSize=13, leading=18, textColor=TEXT_DARK,
          fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
sBody = _ps("Body2", fontSize=10.5, leading=15, textColor=TEXT_DARK,
            fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6)
sBullet = _ps("Bullet", fontSize=10.5, leading=15, textColor=TEXT_DARK,
              fontName="Helvetica", leftIndent=18, bulletIndent=6,
              spaceAfter=3)
sCode = _ps("Code2", fontSize=9, leading=13, textColor=HexColor("#333355"),
            fontName="Courier", backColor=LIGHT_BG, leftIndent=12,
            rightIndent=12, spaceBefore=4, spaceAfter=6,
            borderWidth=0.5, borderColor=BRAND_LIGHT, borderPadding=6)
sCaption = _ps("Caption", fontSize=9, leading=12, textColor=TEXT_BODY,
               fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=8)
sSmall = _ps("Small", fontSize=9, leading=12, textColor=TEXT_BODY,
             fontName="Helvetica", spaceAfter=4)

# в”Ђв”Ђ Helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def h1(t):  return Paragraph(t, sH1)
def h2(t):  return Paragraph(t, sH2)
def h3(t):  return Paragraph(t, sH3)
def p(t):   return Paragraph(t, sBody)
def b(t):   return Paragraph(f"вЂў {t}", sBullet)
def sp(n=6):return Spacer(1, n)
def hr():   return HRFlowable(width="100%", thickness=1, color=BRAND_LIGHT,
                              spaceBefore=6, spaceAfter=6)

def tbl(data, col_widths=None, header=True):
    """Create a branded table."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("BACKGROUND",  (0,0), (-1,0), BRAND),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 10),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9.5),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("GRID",        (0,0), (-1,-1), 0.4, BRAND_LIGHT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_BG]),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(cmds))
    return t

# в”Ђв”Ђ Page callbacks в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_BODY)
    canvas.drawString(2*cm, 1.2*cm, "SteppeDNA вЂ” Comprehensive Project Guide")
    canvas.drawRightString(w - 2*cm, 1.2*cm, f"Page {doc.page}")
    # Top accent line
    canvas.setStrokeColor(BRAND)
    canvas.setLineWidth(2)
    canvas.line(2*cm, h - 1.5*cm, w - 2*cm, h - 1.5*cm)
    canvas.restoreState()

# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
#  CONTENT
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
def build_content():
    S = []   # story list

    # в”Ђв”Ђ COVER PAGE в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    S.append(Spacer(1, 80))
    S.append(Paragraph("SteppeDNA", sTitle))
    S.append(Paragraph("РљР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ РїР°С‚РѕРіРµРЅРЅРѕСЃС‚Рё РІР°СЂРёР°РЅС‚РѕРІ РіРµРЅРѕРІ", sSubtitle))
    S.append(sp(12))
    S.append(Paragraph("РџРѕР»РЅРѕРµ Р СѓРєРѕРІРѕРґСЃС‚РІРѕ РїРѕ РџСЂРѕРµРєС‚Сѓ", _ps("x", fontSize=18,
             leading=24, textColor=TEXT_DARK, fontName="Helvetica-Bold",
             alignment=TA_CENTER)))
    S.append(sp(20))
    S.append(hr())
    S.append(Paragraph(
        f"РЎРіРµРЅРµСЂРёСЂРѕРІР°РЅРѕ {datetime.datetime.now().strftime('%d.%m.%Y')}",
        _ps("dt", fontSize=11, alignment=TA_CENTER, textColor=TEXT_BODY)))
    S.append(sp(8))
    S.append(Paragraph(
        "128 РџСЂРёР·РЅР°РєРѕРІ В· 5 Р“РµРЅРѕРІ HR-РїСѓС‚Рё В· РђРЅСЃР°РјР±Р»РµРІРѕРµ ML В· РћР±СЉСЏСЃРЅРµРЅРёСЏ SHAP В· РљСЂРёС‚РµСЂРёРё ACMG/AMP",
        _ps("tag", fontSize=10, alignment=TA_CENTER, textColor=BRAND)))
    S.append(sp(30))
    S.append(Paragraph(
        "Р­С‚РѕС‚ РґРѕРєСѓРјРµРЅС‚ РїСЂРµРґРѕСЃС‚Р°РІР»СЏРµС‚ РіР»СѓР±РѕРєРёР№ РїРѕСЃС‚Р°С‚РµР№РЅС‹Р№ СЂР°Р·Р±РѕСЂ РєР°Р¶РґРѕРіРѕ РєРѕРјРїРѕРЅРµРЅС‚Р°, "
        "РїСЂРёР·РЅР°РєР°, РёСЃС‚РѕС‡РЅРёРєР° РґР°РЅРЅС‹С…, РјРµС‚РѕРґР° Рё С„Р°Р№Р»Р° РІ РїСЂРѕРµРєС‚Рµ SteppeDNA. РћРЅРѕ РїСЂРµРґРЅР°Р·РЅР°С‡РµРЅРѕ "
        "РєР°Рє СЂСѓРєРѕРІРѕРґСЃС‚РІРѕ РґР»СЏ РїРѕРЅРёРјР°РЅРёСЏ РїРѕР»РЅРѕРіРѕ РѕС…РІР°С‚Р° Рё С‚РµС…РЅРёС‡РµСЃРєРѕР№ РіР»СѓР±РёРЅС‹ СЂР°Р±РѕС‚С‹.",
        _ps("intro", fontSize=10.5, leading=16, alignment=TA_CENTER,
            textColor=TEXT_BODY, leftIndent=40, rightIndent=40)))
    S.append(PageBreak())

    # в”Ђв”Ђ Оглавление (manual) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    S.append(h1("Оглавление"))
    toc_items = [
        "1. Обзор проекта и Цели",
        "2. Метрики кодовой базы",
        "3. Языки, фреймворки и зависимости",
        "4. Полный инвентарь файлов",
        "5. Архитектура системы",
        "6. Пайплайн машинного обучения",
        "7. Описание всех 128 признаков",
        "8. External Datasets & APIs",
        "9. Эндпоинты Backend API",
        "10. Frontend & User Interface",
        "11. Пайплайны данных (Data Pipelines)",
        "12. Testing & Validation",
        "13. Меры безопасности",
        "14. Развертывание (Docker)",
        "15. Полный список функций (пользовательских)",
        "16. Глоссарий ключевых терминов",
    ]
    for item in toc_items:
        S.append(Paragraph(item, _ps("toc", fontSize=11, leading=18,
                 textColor=TEXT_DARK, leftIndent=20, spaceAfter=2)))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 1. PROJECT OVERVIEW
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("1. Обзор проекта и Цели"))
    S.append(p(
        "SteppeDNA — это веб-инструмент биоинформатики, который предсказывает, является ли миссенс-мутация ДНК "
        "в одном из пяти генов репарации ДНК путем гомологичной рекомбинации (HR) "
        "<b>патогенной</b> (вызывающей заболевание) или <b>доброкачественной</b> (безвредной). "
        "The five supported genes вЂ” <b>BRCA1, BRCA2, PALB2, RAD51C, RAD51D</b> вЂ” are all "
        "являются частью HR-пути, который критически важен для репарации двунитевых разрывов ДНК. "
        "Мутации в этих генах тесно связаны с наследственным раком молочной железы и яичников. "
        ""
    ))
    S.append(p(
        "Система использует <b>ансамбль моделей машинного обучения</b> (XGBoost + "
        "Глубокая нейронная сеть), обученных на 750+ клинически аннотированных миссенс-вариантах из "
        "ClinVar, обогащенных 128 признаками из 8 внешних биологических баз данных "
        "и вычислительных инструментов. Каждое предсказание сопровождается:"
    ))
    S.append(b("A calibrated pathogenicity probability (0.0 вЂ“ 1.0)"))
    S.append(b("Доверительным интервалом на основе аппроксимации бета-распределением"))
    S.append(b("Объяснениями SHAP, показывающими <i>почему</i> модель приняла такое решение"))
    S.append(b("Клиническими критериями ACMG/AMP (PM1, PP3, BP4, BS1, PVS1)"))
    S.append(b("Живой кросс-проверкой по базам данных ClinVar и gnomAD"))
    S.append(sp())
    S.append(p(
        "<b>Ключевое отличие:</b> SteppeDNA - это не просто черный ящик. Включает "
        "перехватчик 1-го уровня (Tier 1) для очевидных усекающих мутаций (nonsense/frameshift), "
        "и ML-движок 2-го уровня (Tier 2) для сложных миссенс-вариантов. Каждый результат объясним."
    ))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 2. CODEBASE METRICS
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("2. Метрики кодовой базы"))
    S.append(p("Следующая таблица показывает разбивку исходного кода по языкам:"))
    S.append(tbl([
        ["Язык / Формат", "Файлы", "Строки кода", "Размер (КБ)"],
        ["Python (.py)",       "69",  "10,955",          "567.5"],
        ["JavaScript (.js)",   "3",   "975",             "50.8"],
        ["CSS (.css)",         "1",   "1,436",           "28.4"],
        ["HTML (.html)",       "3",   "552",             "41.8"],
        ["JSON (.json)",       "10",  "188",             "1,585.1"],
        ["Markdown (.md)",     "5",   "187",             "23.6"],
        ["VCF (.vcf)",         "2",   "1,086",           "12.6"],
        ["YAML (.yml)",        "1",   "10",              "0.2"],
        ["Text (.txt)",        "7",   "51",              "22.9"],
        ["ВСЕГО",              "101", "15,440",          "~2,332"],
    ], col_widths=[140, 50, 90, 70]))
    S.append(sp())
    S.append(p(
        "<b>Основной язык:</b> Python (75.6% всего кода). "
        "Проект также включает сериализованные артефакты моделей (.pkl, .h5, .json) объемом "
        "~60 МБ в директории <font face='Courier'>data/</font>."
    ))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 3. Языки, фреймворки и зависимости
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("3. Языки, фреймворки и зависимости"))

    S.append(h2("3.1 Backend (Python)"))
    S.append(tbl([
        ["Пакет", "Версия", "Назначение"],
        ["FastAPI", "0.129.0", "Высокопроизводительный асинхронный веб-фреймворк для REST API"],
        ["Uvicorn", "0.41.0", "Сервер ASGI для запуска FastAPI"],
        ["XGBoost", "2.1.4", "Модель градиентного бустинга деревьев (Tier 2 ML)"],
        ["TensorFlow", "в‰Ґ2.15.0", "Глубокая нейронная сеть (часть ансамбля + резерв ESM-2)"],
        ["scikit-learn", "1.7.2", "StandardScaler, калибровка IsotonicRegression, метрики"],
        ["SHAP", "0.49.1", "Объяснимость модели через значения Шепли (SHAP)"],
        ["pandas", "2.3.2", "Манипуляции с табличными данными в обучающих скриптах"],
        ["NumPy", "1.26.4", "Вычислительная математика и векторы признаков"],
        ["SciPy", "в‰Ґ1.12.0", "Бета-распределение для доверительных интервалов"],
        ["BioPython", "1.85", "Трансляция CDS, таблица кодонов, работа с последовательностями"],
        ["httpx", "в‰Ґ0.27.0", "Асинхронный HTTP-клиент для живых запросов к ClinVar/gnomAD"],
        ["imbalanced-learn", "0.14.1", "Оверсемплинг SMOTE для дисбаланса классов"],
        ["python-multipart", "вЂ”", "Парсинг загрузки файлов для VCF эндпоинта"],
    ], col_widths=[100, 60, 300]))
    S.append(sp())

    S.append(h2("3.2 Frontend"))
    S.append(b("<b>HTML5</b> вЂ” semantic markup with accessibility attributes (ARIA roles)"))
    S.append(b("<b>Vanilla CSS</b> вЂ” 1,436 lines with CSS custom properties, dark/light theme, gradient animations"))
    S.append(b("<b>Vanilla JavaScript</b> вЂ” no frameworks, 875 lines of hand-written logic"))
    S.append(b("<b>Inter font</b> вЂ” loaded from Google Fonts for premium typography"))
    S.append(sp())

    S.append(h2("3.3 Infrastructure"))
    S.append(b("<b>Docker</b> вЂ” Dockerfile + docker-compose.yml for containerized deployment"))
    S.append(b("<b>Python 3.10-slim</b> вЂ” base Docker image"))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 4. Полный инвентарь файлов
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("4. Полный инвентарь файлов"))

    S.append(h2("4.1 Backend (backend/)"))
    S.append(tbl([
        ["Файл", "Lines", "Назначение"],
        ["main.py", "1,311", "FastAPI приложение: 8 эндпоинтов, ML инференс, парсинг VCF, прокси ClinVar/gnomAD, middleware"],
        ["feature_engineering.py", "472", "Общие биологические таблицы (BLOSUM62, гидрофобность, объемы), engineer_features() для обучения"],
        ["acmg_rules.py", "55", "Движок клинических правил ACMG/AMP: оценка доказательств PM1, PP3, BP4, BS1"],
        ["constants.py", "27", "Таблица трансляции кодонов (64 кодона), маппинг комплементарности ДНК"],
        ["gene_configs/*.json (Г—5)", "~165", "Конфигурация по генам: хромосома, цепь, длина CDS, длина AA, функциональные домены"],
    ], col_widths=[130, 40, 290]))
    S.append(sp())

    S.append(h2("4.2 Frontend (frontend/)"))
    S.append(tbl([
        ["Файл", "Lines", "Назначение"],
        ["index.html", "315", "Главное одностраничное приложение (SPA) с семантической структурой HTML5"],
        ["index-split.html", "237", "Альтернативный вариант макета"],
        ["app.js", "735", "Основная логика: отправка форм, загрузка VCF, рендеринг результатов"],
        ["api.js", "16", "Единый источник истины для всех URL-констант бэкенда"],
        ["lang.js", "224", "Трехъязычная система i18n: английский, казахский, русский"],
        ["styles.css", "1,436", "Полная дизайн-система: CSS-переменные, темы, анимации"],
    ], col_widths=[110, 40, 310]))
    S.append(sp())

    S.append(h2("4.3 Training &amp; Analysis Scripts (scripts/)"))
    S.append(tbl([
        ["Файл", "Lines", "Назначение"],
        ["cross_validate.py", "336", "10-кратная стратифицированная CV с бутстрап-доверительными интервалами"],
        ["train_ensemble_baseline.py", "393", "Ансамбль DNN с SMOTE + focal-loss (5 сидов) + изотоническая калибровка"],
        ["train_ensemble_blind.py", "~170", "Вариант ансамбля для слепого тестирования"],
        ["train_xgboost.py", "~110", "Автономный тренер XGBoost с гиперпараметрами Optuna"],
        ["tune_xgboost.py / tune_xgboost_blind.py", "~260", "Байесовский поиск гиперпараметров через Optuna"],
        ["train_multitask_blind.py", "~240", "Вариант многозадачной нейронной сети"],
        ["train_universal_model.py", "~155", "Тренер универсальной (пан-генной) модели"],
        ["vus_reclassification.py", "513", "Извлечение ClinVar VUS + реклассификация SteppeDNA + визуализация"],
        ["ablation_study_xgb.py", "~450", "Анализ абляции с исключением по одной группе признаков"],
        ["sota_comparison.py", "~480", "Бенчмарки против REVEL, CADD, AlphaMissense, PolyPhen-2"],
        ["external_validation*.py (Г—3)", "~700", "Валидация MAVE (отложенная выборка) для XGBoost, ансамбля, multitask"],
        ["run_benchmark.py", "~370", "Сквозной запуск бенчмарков"],
        ["generate_*_pdf.py (Г—8)", "~1,400", "Различные генераторы PDF-отчетов"],
        ["check_*.py (Г—5)", "~350", "Проверки целостности данных (утечки, мутации, VCF, известные варианты)"],
    ], col_widths=[155, 40, 265]))
    S.append(sp())

    S.append(h2("4.4 Data Pipelines (data_pipelines/)"))
    S.append(tbl([
        ["Файл", "Lines", "Назначение"],
        ["fetch_alphafold.py", "321", "3D structural features from PDB 1MIU + ESMFold (RSA, B-factor, secondary structure, distances)"],
        ["fetch_alphamissense.py", "~180", "DeepMind AlphaMissense pathogenicity scores (Cheng et al. 2023)"],
        ["fetch_gnomad.py", "~350", "gnomAD v4 allele frequencies (global + 4 sub-populations)"],
        ["fetch_phylop.py", "~240", "PhyloP 100-way vertebrate conservation scores from UCSC"],
        ["fetch_dbnsfp.py", "~520", "dbNSFP aggregated functional predictions"],
        ["fetch_mave.py", "~140", "MAVE HDR functional assay scores (Hu C et al. 2024)"],
        ["fetch_spliceai.py", "~170", "SpliceAI delta scores for splice-site disruption"],
        ["generate_esm2_embeddings.py", "246", "ESM-2 protein language model embeddings (Meta AI)"],
        ["prepare_brca1.py / prepare_all_genes.py", "~440", "Подготовка и унификация датасетов для конкретных генов"],
    ], col_widths=[170, 40, 250]))
    S.append(sp())

    S.append(h2("4.5 Tests (tests/)"))
    S.append(tbl([
        ["Файл", "Lines", "Назначение"],
        ["conftest.py", "~45", "Фикстуры Pytest и общая конфигурация тестов"],
        ["test_acmg_rules.py", "~180", "Тесты оценки кодов доказательств ACMG"],
        ["test_feature_engineering.py", "~180", "Тесты корректности построения вектора признаков"],
        ["test_negative_cases.py", "~165", "Пограничные случаи: неверный ввод, граничные условия"],
        ["test_shap_stability.py", "~185", "Тесты детерминированности и стабильности значений SHAP"],
        ["test_variants.vcf", "вЂ”", "Пример VCF-файла для интеграционных тестов"],
    ], col_widths=[160, 40, 260]))
    S.append(sp())

    S.append(h2("4.6 Other Files"))
    S.append(tbl([
        ["Файл", "Назначение"],
        ["Dockerfile", "Определение образа контейнера (на базе Python 3.10-slim)"],
        ["docker-compose.yml", "Конфигурация оркестрации единого сервиса"],
        ["requirements.txt", "13 закрепленных зависимостей Python"],
        ["comprehensive_test_suite.vcf", "224-строчный VCF с патогенными + доброкачественными + граничными вариантами"],
        ["brca2_missense_dataset_2.csv", "Обучающий датасет ClinVar (759 строк)"],
        ["gen_vcf.py", "Скрипт для генерации тестовых VCF-файлов"],
        ["SteppeDNA_Guide.md", "Краткое руководство пользователя"],
        ["visual_proofs/ (13 files)", "Графики: ROC-кривые, абляция, VUS, валидация MAVE"],
    ], col_widths=[180, 280]))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 5. Архитектура системы
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("5. Архитектура системы"))
    S.append(p("SteppeDNA follows a classic <b>client-server</b> architecture:"))
    S.append(sp())
    S.append(Paragraph(
        "<font face='Courier' size='9'>"
        "в”Њв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”ђ\n"
        "в”‚       FRONTEND  (Статичный HTML/CSS/JS)             в”‚\n"
        "в”‚ index.html + app.js + styles.css + lang.js       в”‚\n"
        "в”‚ в”Ђ Ручной ввод формы (cDNA, AA_ref, AA_alt)       в”‚\n"
        "в”‚ в”Ђ Загрузка VCF файла (drag-and-drop)                  в”‚\n"
        "в”‚ в”Ђ Переключение темной/светлой темы                        в”‚\n"
        "в”‚ в”Ђ Трехъязычный (EN / KK / RU)                      в”‚\n"
        "в””в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”¬в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”\n"
        "          в”‚  HTTP POST / GET (JSON)\n"
        "          в–ј\n"
        "в”Њв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”ђ\n"
        "в”‚     БЭКЭНД   (FastAPI + Uvicorn, port 8000)      в”‚\n"
        "в”‚ Middleware: CORS в†’ API Key в†’ Rate Limit в†’ Req ID в”‚\n"
        "в”‚ УРОВЕНЬ 1: Правила (Nonsense/Frameshift)        в”‚\n"
        "в”‚    в””в”Ђ PVS1 авто-патогенный                       в”‚\n"
        "в”‚ УРОВЕНЬ 2: ML Движок                                в”‚\n"
        "в”‚    в”њв”Ђ build_feature_vector() в†’ 128 features      в”‚\n"
        "в”‚    в”њв”Ђ Нормализация StandardScaler                в”‚\n"
        "в”‚    в”њв”Ђ Предсказание XGBoost (вес 60%)            в”‚\n"
        "в”‚    в”њв”Ђ Предсказание DNN (вес 40%)                в”‚\n"
        "в”‚    в”њв”Ђ Blended в†’ Isotonic calibration             в”‚\n"
        "в”‚    в”њв”Ђ Д.И. на основе бета-распределения       в”‚\n"
        "в”‚    в”њв”Ђ Топ-8 атрибуций признаков SHAP            в”‚\n"
        "в”‚    в””в”Ђ Оценка доказательств ACMG/AMP               в”‚\n"
        "в”‚ СЛОЙ ДАННЫХ: Артефакты моделей (Pickle/H5)       в”‚\n"
        "в”‚    Ленивая загрузка по генам с кэшем LRU         в”‚\n"
        "в””в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”\n"
        "</font>", sBody))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 6. ML PIPELINE
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("6. Пайплайн машинного обучения"))

    S.append(h2("6.1 Training Data"))
    S.append(p(
        "The master training dataset (<font face='Courier'>brca2_missense_dataset_2.csv</font>) "
        "contains <b>759 missense variants</b> from ClinVar with binary labels: "
        "Pathogenic (1) vs Benign (0). The class distribution is imbalanced (~11% pathogenic), "
        "which is addressed through SMOTE oversampling and focal loss."
    ))

    S.append(h2("6.2 Feature Engineering (128 Features)"))
    S.append(p(
        "Each variant is converted into a 128-dimensional feature vector. Features are grouped into "
        "8 categories (see Section 7 for the full list). The same <font face='Courier'>build_feature_vector()</font> "
        "function is used in both training and inference for consistency."
    ))

    S.append(h2("6.3 Model Architecture вЂ” Stacked Ensemble"))
    S.append(p("The Tier 2 ML engine uses a <b>stacked ensemble</b> of two models:"))
    S.append(b("<b>XGBoost</b> (gradient-boosted trees) вЂ” 60% blend weight. Hyperparameters tuned via Optuna Bayesian search."))
    S.append(b("<b>Deep Neural Network</b> (TensorFlow/Keras) вЂ” 40% blend weight. Multi-layer perceptron with dropout, trained with focal loss and class weighting. Ensemble of 5 models with different random seeds."))
    S.append(p(
        "The blended raw probability is passed through <b>Isotonic Regression</b> calibration "
        "(trained on the test split) to produce well-Откалиброванные вероятности."
    ))

    S.append(h2("6.4 Evaluation & Performance"))
    S.append(tbl([
        ["Метрика", "Значение", "Метод"],
        ["ROC-AUC (cross-val)", "~0.73", "10-fold stratified CV with 1000-bootstrap CIs"],
        ["Training set size", "759 variants", "ClinVar BRCA2 missense, deduplicated"],
        ["Feature count", "128", "8 biological data sources"],
        ["Validation", "External MAVE holdout", "Hu C et al. 2024 functional assay data"],
    ], col_widths=[140, 90, 230]))
    S.append(sp())

    S.append(h2("6.5 Overfitting Prevention"))
    S.append(b("Stratified K-fold cross-validation (10 folds) вЂ” no data leakage between folds"))
    S.append(b("Separate scaler fitted on each fold's training data only"))
    S.append(b("Early stopping with patience (DNN training)"))
    S.append(b("Data leakage checks (<font face='Courier'>check_leakage.py, check_leakage_brca2.py</font>)"))
    S.append(b("VUS deduplication against training set before reclassification"))
    S.append(b("External validation on MAVE holdout set (never seen during training)"))

    S.append(h2("6.6 Two-Tier Prediction System"))
    S.append(p("<b>Tier 1 вЂ” Rule Interceptor:</b> Truncating mutations (nonsense, frameshift, indels) "
               "are automatically classified as Pathogenic with p=0.9999 and PVS1 ACMG evidence. "
               "These don't need ML вЂ” the biology is clear."))
    S.append(p("<b>Tier 2 вЂ” ML Engine:</b> Missense variants go through the full 128-feature "
               "pipeline with stacked ensemble prediction, confidence estimation, and SHAP explanation."))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 7. ALL 128 FEATURES
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("7. Описание всех 128 признаков"))
    S.append(p("Every variant is encoded into exactly 128 features from 8 groups:"))
    S.append(sp())

    feature_groups = [
        ("Group 1: Amino Acid Properties (11 features)", [
            ("blosum62_score", "BLOSUM62 substitution matrix score вЂ” measures evolutionary similarity of the amino acid change"),
            ("volume_diff", "Absolute difference in molecular volume between reference and alternate AA"),
            ("hydro_diff / hydro_delta / ref_hydro / alt_hydro", "Hydrophobicity metrics (Kyte-Doolittle scale)"),
            ("charge_change", "Binary: did the electric charge class change? (positive/negative/nonpolar)"),
            ("nonpolar_to_charged", "Binary: specifically nonpolar в†’ charged transition (drastic)"),
            ("is_nonsense / is_transition / is_transversion", "Mutation type flags"),
        ]),
        ("Group 2: Positional & Domain (9 features)", [
            ("cDNA_pos / AA_pos", "Raw nucleotide and amino acid positions"),
            ("relative_cdna_pos / relative_aa_pos", "Position normalized to gene length (0.0 вЂ“ 1.0)"),
            ("in_critical_repeat_region", "In BRC repeats, WD40, BRCT, SCD domains"),
            ("in_DNA_binding", "In DNA binding domain"),
            ("in_OB_folds", "In oligonucleotide-binding folds, RING, Walker A/B"),
            ("in_NLS", "In nuclear localization signal"),
            ("in_primary_interaction", "In PALB2/BRCA1 interaction domain"),
        ]),
        ("Group 3: Conservation вЂ” PhyloP (4 features)", [
            ("phylop_score", "PhyloP 100-way vertebrate conservation score (higher = more conserved)"),
            ("high_conservation", "Binary: PhyloP > 4.0"),
            ("ultra_conservation", "Binary: PhyloP > 7.0"),
            ("conserv_x_blosum", "Cross-feature: conservation Г— BLOSUM62 score"),
        ]),
        ("Group 4: Functional Assay вЂ” MAVE (4 features)", [
            ("mave_score", "HDR functional assay score from Hu C et al. 2024"),
            ("has_mave", "Binary: has experimental MAVE data"),
            ("mave_abnormal", "Binary: abnormal function (score 0.01 вЂ“ 1.49)"),
            ("mave_x_blosum", "Cross-feature: MAVE Г— BLOSUM62"),
        ]),
        ("Group 5: AlphaMissense (3 features)", [
            ("am_score", "DeepMind AlphaMissense pathogenicity prediction (0вЂ“1)"),
            ("am_pathogenic", "Binary: AM score > 0.564 (pathogenic threshold)"),
            ("am_x_phylop", "Cross-feature: AlphaMissense Г— PhyloP"),
        ]),
        ("Group 6: 3D Structure вЂ” AlphaFold/PDB (10 features)", [
            ("rsa / is_buried", "Relative solvent accessibility; binary buried flag (<0.25)"),
            ("bfactor", "AlphaFold structural confidence (pLDDT)"),
            ("dist_dna / dist_palb2", "3D distance to DNA binding interface / PALB2 site (Angstroms)"),
            ("is_dna_contact", "Binary: within 5Г… of DNA"),
            ("ss_helix / ss_sheet", "Secondary structure: alpha-helix or beta-sheet"),
            ("buried_x_blosum / dna_contact_x_blosum", "Cross-features with BLOSUM62"),
        ]),
        ("Group 7: Population Frequency вЂ” gnomAD (9 features)", [
            ("gnomad_af / gnomad_af_log", "Global allele frequency and its log-transform"),
            ("gnomad_popmax", "Maximum frequency across all sub-populations"),
            ("gnomad_afr / gnomad_amr / gnomad_eas / gnomad_nfe", "African, American, East Asian, Non-Finnish European AF"),
            ("is_rare", "Binary: AF < 0.001"),
            ("af_x_blosum", "Cross-feature: frequency Г— BLOSUM62"),
        ]),
        ("Group 8: ESM-2 + SpliceAI + Encodings (78 features)", [
            ("esm2_cosine_sim", "Cosine similarity of WT vs mutant ESM-2 embeddings"),
            ("esm2_l2_shift", "L2 norm of embedding difference vector"),
            ("esm2_pca_0 вЂ¦ esm2_pca_19", "20 PCA components of ESM-2 difference vectors"),
            ("spliceai_score / splice_pathogenic", "SpliceAI delta score + binary pathogenic flag"),
            ("Mutation_A>C вЂ¦ Mutation_T>G", "12 one-hot encoded nucleotide mutations"),
            ("AA_ref_Ala вЂ¦ AA_ref_Val", "21 one-hot encoded reference amino acids"),
            ("AA_alt_Ala вЂ¦ AA_alt_Val", "21 one-hot encoded alternate amino acids"),
        ]),
    ]

    for group_title, features in feature_groups:
        S.append(h3(group_title))
        for fname, desc in features:
            S.append(Paragraph(
                f"<b><font face='Courier' size='9'>{fname}</font></b> вЂ” {desc}",
                sBullet))
        S.append(sp(4))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 8. EXTERNAL DATA
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("8. Внешние датасеты и API"))
    S.append(tbl([
        ["Источник", "Тип", "Что предоставляет", "Скрипт пайплайна"],
        ["ClinVar (NCBI)", "Database", "Clinically-annotated variants with pathogenic/benign labels (training labels)", "Direct CSV download"],
        ["gnomAD v4", "Database + API", "Population allele frequencies across 6 sub-populations", "fetch_gnomad.py"],
        ["PhyloP (UCSC)", "Database", "100-way vertebrate conservation scores per nucleotide position", "fetch_phylop.py"],
        ["AlphaMissense", "ML Model (DeepMind)", "Pre-computed pathogenicity scores for all possible missense mutations", "fetch_alphamissense.py"],
        ["MAVE / MaveDB", "Experimental", "HDR functional assay scores measuring actual protein function", "fetch_mave.py"],
        ["AlphaFold / PDB 1MIU", "Structural DB", "3D protein structure for RSA, B-factor, distances, secondary structure", "fetch_alphafold.py"],
        ["ESM-2 (Meta AI)", "Protein LLM", "Contextual protein sequence embeddings (6-layer, 8M param model)", "generate_esm2_embeddings.py"],
        ["SpliceAI", "ML Model", "Deep-learning splice-site disruption predictions", "fetch_spliceai.py"],
        ["dbNSFP", "Aggregated DB", "Pre-computed functional predictions from 30+ tools", "fetch_dbnsfp.py"],
    ], col_widths=[80, 70, 165, 145]))
    S.append(sp())
    S.append(p(
        "The ClinVar and gnomAD APIs are also used at <b>runtime</b> (live lookup endpoints) "
        "to cross-reference predictions against the latest clinical annotations."
    ))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 9. API ENDPOINTS
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("9. Эндпоинты Backend API"))
    S.append(tbl([
        ["Метод", "Путь", "Назначение"],
        ["POST", "/predict", "Single-variant pathogenicity prediction (main endpoint)"],
        ["POST", "/predict/vcf", "Batch VCF file upload вЂ” parses and predicts all missense variants"],
        ["GET", "/", "API status and model info"],
        ["GET", "/health", "Deep health check вЂ” validates all models, scalers, and data are loaded"],
        ["GET", "/lookup/clinvar/{variant}", "Живой поиск ClinVar via NCBI E-utilities API"],
        ["GET", "/lookup/gnomad/{variant}", "Live gnomAD v4 lookup via GraphQL API"],
    ], col_widths=[45, 160, 255]))
    S.append(sp())
    S.append(h2("9.1 Middleware Stack"))
    S.append(b("<b>CORS</b> вЂ” Configurable allowed origins for cross-origin requests"))
    S.append(b("<b>API Key</b> вЂ” Optional header-based authentication (required in production)"))
    S.append(b("<b>Rate Limiter</b> вЂ” In-memory per-IP rate limiting (60 req/min default, separate limit for external lookups)"))
    S.append(b("<b>Request ID</b> вЂ” UUID attached to every request for tracing and debugging"))
    S.append(sp())
    S.append(h2("9.2 Caching"))
    S.append(p("ClinVar and gnomAD lookups use an in-memory <b>LRU cache</b> with 1-hour TTL and 1,000-entry capacity, "
               "avoiding redundant external API calls."))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 10. FRONTEND
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("10. Frontend и Пользовательский интерфейс"))
    S.append(h2("10.1 Design System"))
    S.append(b("Purple-themed colour palette with CSS custom properties (30+ variables)"))
    S.append(b("Light and dark mode with smooth 0.4s cubic-bezier transitions on all elements"))
    S.append(b("Glassmorphism cards with gradient backgrounds and hover shadows"))
    S.append(b("Inter font family (Google Fonts) for premium typography"))
    S.append(b("Responsive grid layout (max-width 920px, 2-column grid)"))
    S.append(b("Floating hero section with animated radial gradients"))
    S.append(sp())
    S.append(h2("10.2 User Input Methods"))
    S.append(b("<b>Manual entry form:</b> Gene selector в†’ cDNA position в†’ Reference AA в†’ Alternate AA в†’ Mutation type в†’ Nucleotide change"))
    S.append(b("<b>VCF file upload:</b> Drag-and-drop or click-to-browse with loading spinner"))
    S.append(sp())
    S.append(h2("10.3 Results Display"))
    S.append(b("Pathogenic/Benign badge with colour-coded probability bar"))
    S.append(b("Confidence interval display (Beta-distribution CI)"))
    S.append(b("Data source cards: PhyloP, MAVE, AlphaMissense, Structure details"))
    S.append(b("SHAP feature attribution bar chart (top 8 features, red/green)"))
    S.append(b("ACMG evidence codes with human-readable rationale"))
    S.append(b("Live ClinVar and gnomAD fetch buttons"))
    S.append(b("VCF batch results table with sortable columns"))
    S.append(sp())
    S.append(h2("10.4 Internationalisation (i18n)"))
    S.append(p("Full Трехъязычный support with crossfade animation on language switch:"))
    S.append(b("<b>English</b> вЂ” Default language"))
    S.append(b("<b>Kazakh (ТљР°Р·Р°Т›С€Р°)</b> вЂ” Full UI translation"))
    S.append(b("<b>Russian (Р СѓСЃСЃРєРёР№)</b> вЂ” Full UI translation"))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 11. Пайплайны данных (Data Pipelines)
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("11. Пайплайны данных (Data Pipelines)"))
    S.append(p(
        "The <font face='Courier'>data_pipelines/</font> directory contains parameterized scripts "
        "that fetch and process external biological data. Each supports a <font face='Courier'>--gene</font> "
        "argument and reads gene-specific configurations from <font face='Courier'>backend/gene_configs/</font>."
    ))
    S.append(b("<b>fetch_alphafold.py</b> вЂ” Загружает структуры PDB, вычисляет RSA, B-фактор, вторичную структуру, расстояния до интерфейсов ДНК/PALB2"))
    S.append(b("<b>fetch_alphamissense.py</b> вЂ” Загружает и индексирует баллы AlphaMissense по ключу варианта"))
    S.append(b("<b>fetch_gnomad.py</b> вЂ” Запрашивает частоты аллелей gnomAD v4 по 6 субпопуляциям"))
    S.append(b("<b>fetch_phylop.py</b> вЂ” Извлекает баллы PhyloP 100-way из файлов UCSC bigWig"))
    S.append(b("<b>fetch_mave.py</b> вЂ” Загружает данные функционального анализа MAVE HDR"))
    S.append(b("<b>fetch_spliceai.py</b> вЂ” Получает дельта-баллы SpliceAI"))
    S.append(b("<b>fetch_dbnsfp.py</b> вЂ” Извлекает предсказания из агрегированной БД dbNSFP"))
    S.append(b("<b>generate_esm2_embeddings.py</b> вЂ” Запускает белковую языковую модель ESM-2 для эмбеддингов вариантов + PCA"))
    S.append(b("<b>prepare_brca1.py / prepare_all_genes.py</b> вЂ” Unifies datasets across genes into training-ready format"))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 12. TESTING
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("12. Тестирование и валидация"))
    S.append(h2("12.1 Unit Tests (pytest)"))
    S.append(b("<b>test_acmg_rules.py</b> вЂ” Verifies PM1, PP3, BP4, BS1 evidence codes fire correctly"))
    S.append(b("<b>test_feature_engineering.py</b> вЂ” Validates feature vector construction, BLOSUM62 lookups, domain mapping"))
    S.append(b("<b>test_negative_cases.py</b> вЂ” Edge cases: invalid amino acids, out-of-range positions, malformed input"))
    S.append(b("<b>test_shap_stability.py</b> вЂ” Ensures SHAP values are deterministic across repeated runs"))
    S.append(sp())
    S.append(h2("12.2 ML Validation"))
    S.append(b("10-fold stratified cross-validation with 1,000 bootstrap Доверительные интервалы"))
    S.append(b("External validation on MAVE holdout set (functional assay data never used in training)"))
    S.append(b("Ablation study: leave-one-feature-group-out to measure each group's contribution"))
    S.append(b("SOTA comparison Бенчмарки против REVEL, CADD, AlphaMissense, PolyPhen-2"))
    S.append(b("Data leakage detection scripts to ensure no test-set contamination"))
    S.append(sp())
    S.append(h2("12.3 VCF Integration Tests"))
    S.append(p("A 224-line comprehensive VCF test suite covers pathogenic, benign, synonymous, "
               "multi-allelic, indel, and edge-case variants to validate end-to-end VCF parsing."))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 13. SECURITY
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("13. Меры безопасности"))
    S.append(b("XSS prevention: HTML escaping in frontend (escapeHtml function)"))
    S.append(b("Input validation: Pydantic validators on all API inputs (gene name, cDNA pos, AA codes, mutation format)"))
    S.append(b("Path traversal protection: os.path.basename normalization on pickle file loads"))
    S.append(b("Rate limiting: Per-IP in-memory rate limiter (general + external API specific)"))
    S.append(b("CORS: Configurable allowed origins (not wildcard in production)"))
    S.append(b("API key: Header-based authentication enforced in production mode"))
    S.append(b("Query injection prevention: Regex validation on ClinVar/gnomAD lookup parameters"))
    S.append(b("File size limits: VCF uploads capped at 50 MB"))
    S.append(b("Atomic file writes: Thread-locked writes to needs_wetlab_assay.csv"))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 14. DEPLOYMENT
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("14. Развертывание (Docker)"))
    S.append(p("The project includes Docker support for reproducible deployment:"))
    S.append(Paragraph("<font face='Courier' size='9'>docker-compose up --build</font>", sCode))
    S.append(b("Base image: python:3.10-slim"))
    S.append(b("Exposes port 8000"))
    S.append(b("Auto-restart policy: unless-stopped"))
    S.append(b("All dependencies installed via requirements.txt"))
    S.append(b("Server: uvicorn backend.main:app --host 0.0.0.0 --port 8000"))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 15. USER-FACING FEATURES
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("15. Полный список функций (пользовательских)"))
    S.append(p("Общее количество: <b>25+ различных пользовательских функций</b>"))
    S.append(sp())
    features_list = [
        ("Предсказание одиночного варианта", "Ввод мутации вручную и мгновенная классификация патогенности"),
        ("Пакетный анализ VCF", "Загрузка файла VCF для предсказания всех миссенс-вариантов"),
        ("Поддержка 5 генов", "BRCA1, BRCA2, PALB2, RAD51C, RAD51D вЂ” all HR-pathway genes"),
        ("Откалиброванные вероятности", "Output is a well-calibrated probability (0вЂ“1), not just a binary label"),
        ("Доверительные интервалы", "Beta-distribution CIs show prediction certainty"),
        ("Объяснения SHAP", "Гистограмма топ 8 признаков, влияющих на каждое предсказание"),
        ("Критерии ACMG/AMP", "Автоматизированные коды доказательств: PVS1, PM1, PP3, BP4, BS1"),
        ("Движок правил 1 уровня", "Мгновенная классификация усекающих мутаций без ML"),
        ("Живой поиск ClinVar", "Real-time cross-reference against NCBI ClinVar database"),
        ("Живой поиск gnomAD", "Real-time allele frequency from gnomAD v4 GraphQL API"),
        ("Индикаторы источников данных", "Показывает информацию PhyloP, MAVE, AlphaMissense, структуры для варианта"),
        ("Dark / Light Mode", "Переключение с плавным переходом 0.4с на всех элементах UI"),
        ("Трехъязычный Interface", "Английский, Казахский, Русский с анимированным переключением"),
        ("Сортировка для Wet-Lab (мокрая лаборатория)", "Низкоуверенные или патогенные предсказания логируются для наблюдения"),
        ("VUS Reclassification", "Массовый анализ ClinVar VUS с многоуровневой по уверенности реклассификацией"),
        ("Доступность с клавиатуры", "Пользовательские выпадающие списки с навигацией по стрелкам/Enter/Escape"),
        ("Авто-капитализация", "Input fields auto-correct amino acid codes (ala в†’ Ala)"),
        ("Валидация с учетом гена", "Диапазон позиций cDNA адаптируется под выбранный ген"),
        ("Маппинг геномной позиции", "cDNA в†’ genomic coordinate mapping for gnomAD lookups"),
        ("Парсинг VCF с учетом цепи", "Handles both + and в€’ strand genes correctly"),
        ("Поддержка мульти-аллельных VCF", "Разделенные запятыми ALT аллели парсятся индивидуально"),
        ("Эндпоинт проверки работоспособности (Health Check)", "Глубокая валидация здоровья всех компонентов ML"),
        ("Трассировка по Request ID", "UUID на каждый запрос для отладки и логирования"),
        ("Кэш ответов LRU", "1-часовой TTL кэш для внешних API ответов"),
        ("Развертывание Docker", "Контейнерное развертывание одной командой"),
    ]
    for name, desc in features_list:
        S.append(Paragraph(f"<b>{name}</b> вЂ” {desc}", sBullet))
    S.append(PageBreak())

    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    # 16. GLOSSARY
    # в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
    S.append(h1("16. Глоссарий ключевых терминов"))
    glossary = [
        ("Missense variant", "Одиночное изменение нуклеотида, приводящее к другой аминокислоте in the protein"),
        ("Pathogenic", "Вариант, вызывающий заболевание (нарушающий функцию белка)"),
        ("Benign", "Безобидный вариант (не влияющий на функцию белка)"),
        ("VUS", "Variant of Uncertain Significance вЂ” not enough evidence to classify"),
        ("HR pathway", "Homologous Recombination вЂ” the DNA repair mechanism these 5 genes are part of"),
        ("SHAP", "SHapley Additive exPlanations вЂ” a method to explain individual ML predictions"),
        ("ACMG/AMP", "American College of Medical Genetics вЂ” clinical variant classification guidelines"),
        ("BLOSUM62", "BLOcks SUbstitution Matrix вЂ” measures evolutionary likelihood of amino acid substitutions"),
        ("ESM-2", "Evolutionary Scale Modeling вЂ” Meta AI's protein language model"),
        ("PhyloP", "Phylogenetic P-value вЂ” measures evolutionary conservation across 100 vertebrate species"),
        ("gnomAD", "Genome Aggregation Database вЂ” population allele frequencies from 800,000+ individuals"),
        ("AlphaMissense", "Предварительно вычисленные предсказания патогенности DeepMind для всех возможных миссенс-мутаций"),
        ("MAVE", "Multiplexed Assays of Variant Effect вЂ” experimental functional data"),
        ("XGBoost", "eXtreme Gradient Boosting вЂ” a tree-based ML algorithm"),
        ("Isotonic Regression", "Непараметрический метод калибровки для согласования вероятностей с истинными исходами"),
        ("CDS", "Coding DNA Sequence вЂ” the protein-coding portion of a gene"),
        ("cDNA", "Complementary DNA вЂ” a DNA copy of mRNA, used for position numbering"),
        ("VCF", "Variant Call Format вЂ” standard file format for storing genetic variants"),
    ]
    S.append(tbl(
        [["Термин", "Определение"]] + [[t, d] for t, d in glossary],
        col_widths=[120, 340]
    ))
    S.append(sp(20))
    S.append(hr())
    S.append(Paragraph(
        "End of Document вЂ” SteppeDNA Comprehensive Project Guide",
        _ps("end", fontSize=10, alignment=TA_CENTER, textColor=TEXT_BODY)))

    return S


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
#  BUILD PDF
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
def main():
    print("=" * 60)
    print("  SteppeDNA вЂ” Comprehensive Project Guide Generator")
    print("=" * 60)

    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        topMargin=2*cm,
        bottomMargin=2*cm,
        leftMargin=2*cm,
        rightMargin=2*cm,
        title="SteppeDNA вЂ” Comprehensive Project Guide",
        author="SteppeDNA Team",
    )

    story = build_content()

    print(f"\n  Building PDF with {len(story)} elements...")
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"  Size:   {size_mb:.2f} MB")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()



