# src/dashboard/pdf_exporter.py
# -*- coding: utf-8 -*-

import io
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

def generate_pdf_report(metrics: dict, fig_radar=None, df_sample=None) -> bytes:
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
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0284c7')
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b')
    )
    sec_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )

    elements = []

    # 1. หัวเอกสาร
    elements.append(Paragraph("Hospital Patient Feedback & Analytics Report", title_style))
    elements.append(Paragraph(f"Exported Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    elements.append(Spacer(1, 14))

    # 2. ตาราง KPI รวม 4 ช่อง
    total_val = metrics.get("total", 0)
    pos_val = metrics.get("pos", 0)
    neu_val = metrics.get("neu", 0)
    neg_val = metrics.get("neg", 0)

    kpi_data = [
        ["Total Feedback", "Positive", "Neutral", "Negative"],
        [str(total_val), str(pos_val), str(neu_val), str(neg_val)]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 130])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0c192c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # 3. กราฟพายและแถบสัดส่วน (สร้างผ่าน ReportLab Native โดยตรง ไม่ต้องพึ่ง Matplotlib)
    if (pos_val + neu_val + neg_val) > 0:
        elements.append(Paragraph("Sentiment Breakdown Overview:", sec_style))
        
        d = Drawing(520, 110)
        
        # วาด Pie Chart ฝั่งซ้าย
        pc = Pie()
        pc.x = 20
        pc.y = 10
        pc.width = 90
        pc.height = 90
        pc.data = [max(0.01, pos_val), max(0.01, neu_val), max(0.01, neg_val)]
        pc.slices[0].fillColor = colors.HexColor('#22c55e')  # เขียว (Pos)
        pc.slices[1].fillColor = colors.HexColor('#94a3b8')  # เทา (Neu)
        pc.slices[2].fillColor = colors.HexColor('#ef4444')  # แดง (Neg)
        d.add(pc)

        # วาด Legend คำอธิบายฝั่งขวา
        pos_pct = (pos_val / total_val * 100) if total_val else 0
        neu_pct = (neu_val / total_val * 100) if total_val else 0
        neg_pct = (neg_val / total_val * 100) if total_val else 0

        # Positive Box & Text
        d.add(Rect(140, 75, 14, 14, fillColor=colors.HexColor('#22c55e'), strokeColor=None))
        d.add(String(165, 78, f"Positive (เชิงบวก): {pos_val} รายการ ({pos_pct:.1f}%)", fontName='Helvetica-Bold', fontSize=9, fillColor=colors.HexColor('#1e293b')))

        # Neutral Box & Text
        d.add(Rect(140, 50, 14, 14, fillColor=colors.HexColor('#94a3b8'), strokeColor=None))
        d.add(String(165, 53, f"Neutral (เป็นกลาง): {neu_val} รายการ ({neu_pct:.1f}%)", fontName='Helvetica-Bold', fontSize=9, fillColor=colors.HexColor('#1e293b')))

        # Negative Box & Text
        d.add(Rect(140, 25, 14, 14, fillColor=colors.HexColor('#ef4444'), strokeColor=None))
        d.add(String(165, 28, f"Negative (เชิงลบ): {neg_val} รายการ ({neg_pct:.1f}%)", fontName='Helvetica-Bold', fontSize=9, fillColor=colors.HexColor('#1e293b')))

        elements.append(d)
        elements.append(Spacer(1, 10))

    # 4. ตารางประเมินผลคะแนนรายแผนก (Department Performance Scores)
    if df_sample is not None and not df_sample.empty:
        dept_col = next((c for c in ["แผนกที่เกี่ยวข้อง", "department", "category", "แผนกที่ประเมิน"] if c in df_sample.columns), None)
        sent_col = next((c for c in ["ความรู้สึก", "sentiment", "label", "ผลภาพรวม (Overall)"] if c in df_sample.columns), None)

        if dept_col and sent_col:
            dept_summary = []
            for d_name, group in df_sample.groupby(dept_col):
                d_total = len(group)
                d_pos = int(group[sent_col].astype(str).str.contains("บวก|pos|พอใจ|1").sum())
                d_neg = int(group[sent_col].astype(str).str.contains("ลบ|neg|ปรับปรุง|-1").sum())
                d_score = round((d_pos / (d_pos + d_neg)) * 5.0, 1) if (d_pos + d_neg) > 0 else 3.5
                dept_summary.append({
                    "dept": str(d_name)[:30],
                    "total": d_total,
                    "pos": d_pos,
                    "neg": d_neg,
                    "score": d_score
                })

            dept_summary.sort(key=lambda x: x["score"], reverse=True)

            elements.append(Paragraph("Department Performance & Satisfaction Scores:", sec_style))
            dept_table_data = [["Department", "Total", "Positive", "Negative", "Score (/5.0)"]]
            for row in dept_summary[:8]:  # แสดง Top 8 แผนก
                dept_table_data.append([
                    row["dept"],
                    str(row["total"]),
                    str(row["pos"]),
                    str(row["neg"]),
                    f"{row['score']:.1f} ★"
                ])

            dept_table = Table(dept_table_data, colWidths=[200, 75, 80, 80, 85])
            dept_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(dept_table)
            elements.append(Spacer(1, 14))

    # 5. ตารางข้อร้องเรียนล่าสุด (Recent Complaints)
    if df_sample is not None and not df_sample.empty:
        elements.append(Paragraph("Recent Complaints Summary (Top 5):", sec_style))
        
        d_col = next((c for c in ["วันที่", "date", "timestamp"] if c in df_sample.columns), None)
        dep_col = next((c for c in ["แผนกที่เกี่ยวข้อง", "department", "category"] if c in df_sample.columns), None)
        feed_col = next((c for c in ["ข้อความความคิดเห็นของลูกค้า", "feedback", "text"] if c in df_sample.columns), None)

        recent_data = [["Date", "Department", "Feedback Summary"]]
        for _, r in df_sample.head(5).iterrows():
            d_val = str(r.get(d_col, "-")) if d_col else "-"
            dep_val = str(r.get(dep_col, "General"))[:22] if dep_col else "General"
            feed_val = str(r.get(feed_col, "-"))[:90] if feed_col else "-"
            
            recent_data.append([
                Paragraph(d_val, cell_style),
                Paragraph(dep_val, cell_style),
                Paragraph(feed_val, cell_style)
            ])

        recent_table = Table(recent_data, colWidths=[90, 130, 300])
        recent_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#334155')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(recent_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
