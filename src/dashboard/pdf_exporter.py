# src/dashboard/pdf_exporter.py
# -*- coding: utf-8 -*-

import io
import os
import urllib.request
from pathlib import Path
from datetime import datetime
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- 1. ระบบดาวน์โหลดและติดตั้งฟอนต์ภาษาไทยอัตโนมัติ ---
def setup_thai_font() -> str:
    font_name = "Prompt-Regular"
    font_bold = "Prompt-Bold"
    
    # โฟลเดอร์เก็บฟอนต์ชั่วคราว
    font_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    
    reg_path = font_dir / "Prompt-Regular.ttf"
    bold_path = font_dir / "Prompt-Bold.ttf"

    # ดาวน์โหลดฟอนต์ Prompt จาก Google Fonts CDN ถ้ายังไม่มี
    if not reg_path.exists():
        url_reg = "https://github.com/google/fonts/raw/main/ofl/prompt/Prompt-Regular.ttf"
        try:
            urllib.request.urlretrieve(url_reg, str(reg_path))
        except Exception:
            pass

    if not bold_path.exists():
        url_bold = "https://github.com/google/fonts/raw/main/ofl/prompt/Prompt-Bold.ttf"
        try:
            urllib.request.urlretrieve(url_bold, str(bold_path))
        except Exception:
            pass

    # ลงทะเบียนฟอนต์กับ ReportLab
    try:
        if reg_path.exists():
            pdfmetrics.registerFont(TTFont(font_name, str(reg_path)))
        if bold_path.exists():
            pdfmetrics.registerFont(TTFont(font_bold, str(bold_path)))
        return font_name
    except Exception:
        return "Helvetica"

def generate_pdf_report(metrics: dict, fig_radar=None, df_sample=None) -> bytes:
    # ติดตั้งฟอนต์ไทย
    thai_font = setup_thai_font()
    thai_bold = "Prompt-Bold" if "Prompt-Bold" in pdfmetrics.getRegisteredFontNames() else thai_font

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=thai_bold,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0284c7')
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName=thai_font,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748b')
    )
    sec_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontName=thai_bold,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName=thai_font,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    cell_bold_style = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName=thai_bold,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    elements = []

    # 1. หัวเอกสาร
    elements.append(Paragraph("รายงานผลวิเคราะห์ความคิดเห็นของผู้รับบริการ (Customer Sentiment Report)", title_style))
    elements.append(Paragraph(f"โรงพยาบาลสิริเวช จันทบุรี | ข้อมูล ณ วันที่: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    elements.append(Spacer(1, 14))

    # 2. ตาราง KPI ภาพรวม
    total_val = metrics.get("total", 0)
    pos_val = metrics.get("pos", 0)
    neu_val = metrics.get("neu", 0)
    neg_val = metrics.get("neg", 0)

    kpi_data = [
        [Paragraph("<font color='white'><b>ความคิดเห็นทั้งหมด</b></font>", cell_bold_style),
         Paragraph("<font color='white'><b>เชิงบวก (Positive)</b></font>", cell_bold_style),
         Paragraph("<font color='white'><b>เป็นกลาง (Neutral)</b></font>", cell_bold_style),
         Paragraph("<font color='white'><b>เชิงลบ (Negative)</b></font>", cell_bold_style)],
        [str(total_val), str(pos_val), str(neu_val), str(neg_val)]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 130])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0c192c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), thai_bold),
        ('FONTSIZE', (0, 1), (-1, 1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # 3. กราฟพายและแถบสัดส่วน
    if (pos_val + neu_val + neg_val) > 0:
        elements.append(Paragraph("สัดส่วนความรู้สึกของผู้รับบริการภาพรวม (Sentiment Breakdown Overview):", sec_style))
        
        d = Drawing(520, 110)
        pc = Pie()
        pc.x = 20
        pc.y = 10
        pc.width = 90
        pc.height = 90
        pc.data = [max(0.01, pos_val), max(0.01, neu_val), max(0.01, neg_val)]
        pc.slices[0].fillColor = colors.HexColor('#22c55e')
        pc.slices[1].fillColor = colors.HexColor('#94a3b8')
        pc.slices[2].fillColor = colors.HexColor('#ef4444')
        d.add(pc)

        pos_pct = (pos_val / total_val * 100) if total_val else 0
        neu_pct = (neu_val / total_val * 100) if total_val else 0
        neg_pct = (neg_val / total_val * 100) if total_val else 0

        d.add(Rect(140, 75, 14, 14, fillColor=colors.HexColor('#22c55e'), strokeColor=None))
        d.add(String(165, 78, f"Positive (เชิงบวก): {pos_val} เรื่อง ({pos_pct:.1f}%)", fontName=thai_bold, fontSize=9, fillColor=colors.HexColor('#1e293b')))

        d.add(Rect(140, 50, 14, 14, fillColor=colors.HexColor('#94a3b8'), strokeColor=None))
        d.add(String(165, 53, f"Neutral (เป็นกลาง): {neu_val} เรื่อง ({neu_pct:.1f}%)", fontName=thai_bold, fontSize=9, fillColor=colors.HexColor('#1e293b')))

        d.add(Rect(140, 25, 14, 14, fillColor=colors.HexColor('#ef4444'), strokeColor=None))
        d.add(String(165, 28, f"Negative (เชิงลบ): {neg_val} เรื่อง ({neg_pct:.1f}%)", fontName=thai_bold, fontSize=9, fillColor=colors.HexColor('#1e293b')))

        elements.append(d)
        elements.append(Spacer(1, 10))

    # 4. ตารางคะแนนและสถิติรายแผนก
    if df_sample is not None and not df_sample.empty:
        dept_col = next((c for c in ["แผนกที่เกี่ยวข้อง", "department", "category", "แผนกที่ประเมิน"] if c in df_sample.columns), None)
        sent_col = next((c for c in ["ความรู้สึก", "sentiment", "label", "ผลภาพรวม (Overall)"] if c in df_sample.columns), None)

        if dept_col and sent_col:
            dept_summary = []
            for d_name, group in df_sample.groupby(dept_col):
                # กรองชื่อแผนกที่ไม่ใช่ nan หรือขีด
                clean_name = str(d_name).strip()
                if clean_name.lower() in ["nan", "none", "", "-"]:
                    clean_name = "บริการทั่วไปของโรงพยาบาล"

                d_total = len(group)
                d_pos = int(group[sent_col].astype(str).str.contains("บวก|pos|พอใจ|1").sum())
                d_neg = int(group[sent_col].astype(str).str.contains("ลบ|neg|ปรับปรุง|-1").sum())
                d_score = round((d_pos / (d_pos + d_neg)) * 5.0, 1) if (d_pos + d_neg) > 0 else 3.5
                dept_summary.append({
                    "dept": clean_name[:35],
                    "total": d_total,
                    "pos": d_pos,
                    "neg": d_neg,
                    "score": d_score
                })

            dept_summary.sort(key=lambda x: (x["score"], x["total"]), reverse=True)

            elements.append(Paragraph("คะแนนความพึงพอใจและสถิติจำแนกรายแผนก (Department Performance Scores):", sec_style))
            dept_table_data = [
                [Paragraph("<font color='white'><b>แผนก / ส่วนงานบริการ</b></font>", cell_bold_style),
                 Paragraph("<font color='white'><b>รวม (เรื่อง)</b></font>", cell_bold_style),
                 Paragraph("<font color='white'><b>เชิงบวก</b></font>", cell_bold_style),
                 Paragraph("<font color='white'><b>เชิงลบ</b></font>", cell_bold_style),
                 Paragraph("<font color='white'><b>คะแนน (/5.0)</b></font>", cell_bold_style)]
            ]
            for row in dept_summary[:8]:
                dept_table_data.append([
                    Paragraph(row["dept"], cell_style),
                    str(row["total"]),
                    str(row["pos"]),
                    str(row["neg"]),
                    f"{row['score']:.1f} ★"
                ])

            dept_table = Table(dept_table_data, colWidths=[200, 75, 80, 80, 85])
            dept_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, -1), thai_font),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(dept_table)
            elements.append(Spacer(1, 14))

    # 5. ตารางข้อร้องเรียนล่าสุด Top 5
    if df_sample is not None and not df_sample.empty:
        elements.append(Paragraph("ตัวอย่างข้อคิดเห็น/ข้อร้องเรียนล่าสุด (Recent Complaints Summary - Top 5):", sec_style))
        
        d_col = next((c for c in ["วันที่", "date", "timestamp"] if c in df_sample.columns), None)
        dep_col = next((c for c in ["แผนกที่เกี่ยวข้อง", "department", "category"] if c in df_sample.columns), None)
        feed_col = next((c for c in ["ข้อความความคิดเห็นของลูกค้า", "feedback", "text"] if c in df_sample.columns), None)

        recent_data = [
            [Paragraph("<b>วันที่</b>", cell_bold_style),
             Paragraph("<b>แผนก</b>", cell_bold_style),
             Paragraph("<b>ข้อความความคิดเห็น / ข้อเสนอแนะ</b>", cell_bold_style)]
        ]
        
        # กรองเอาเฉพาะแถวที่มีข้อความจริง ไม่เป็นขีดหรือว่างเปล่า
        valid_rows = df_sample.copy()
        if feed_col:
            valid_rows = valid_rows[~valid_rows[feed_col].astype(str).str.strip().isin(["-", "nan", "None", ""])]
        
        display_sample = valid_rows.head(5) if not valid_rows.empty else df_sample.head(5)

        for _, r in display_sample.iterrows():
            d_val = str(r.get(d_col, "-")) if d_col else "-"
            dep_val = str(r.get(dep_col, "บริการทั่วไป")) if dep_col else "บริการทั่วไป"
            if dep_val.lower() in ["nan", "none", "", "-"]:
                dep_val = "บริการทั่วไป"
            
            feed_val = str(r.get(feed_col, "-"))[:120] if feed_col else "-"
            
            recent_data.append([
                Paragraph(d_val, cell_style),
                Paragraph(dep_val, cell_style),
                Paragraph(feed_val, cell_style)
            ])

        recent_table = Table(recent_data, colWidths=[90, 130, 300])
        recent_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#334155')),
            ('FONTNAME', (0, 0), (-1, -1), thai_font),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(recent_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
