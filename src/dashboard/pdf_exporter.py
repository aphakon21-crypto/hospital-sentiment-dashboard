import io
import pandas as pd
import plotly.io as pio
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(metrics: dict, fig_radar=None, df_sample=None) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=25, leftMargin=25, topMargin=25, bottomMargin=25)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='TitleStyle',
        fontSize=18,
        leading=22,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0284c7')
    )
    body_style = ParagraphStyle(name='BodyStyle', fontSize=10, leading=14, fontName='Helvetica')
    
    # 1. ส่วนหัวรายงาน
    story.append(Paragraph("Hospital Patient Feedback & Analytics Report", title_style))
    story.append(Paragraph(f"Exported Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 15))
    
    # 2. ตารางสรุปตัวเลขสถิติ (KPIs)
    kpi_data = [
        ["Total Feedback", "Positive", "Neutral", "Negative"],
        [
            str(metrics.get("total", 0)),
            str(metrics.get("pos", 0)),
            str(metrics.get("neu", 0)),
            str(metrics.get("neg", 0))
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[130, 130, 130, 130])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 20))
    
    # 3. แนบรูปชาร์ต Radar Chart
    if fig_radar:
        try:
            radar_img = pio.to_image(fig_radar, format='png', width=520, height=280)
            story.append(RLImage(io.BytesIO(radar_img), width=500, height=270))
            story.append(Spacer(1, 15))
        except Exception:
            pass

    # 4. สรุปรายการล่าสุด (ตารางย่อ)
    if df_sample is not None and not df_sample.empty:
        story.append(Paragraph("Recent Complaints Summary (Top 5):", styles['Heading3']))
        story.append(Spacer(1, 8))
        table_rows = [["Date", "Department", "Feedback"]]
        for _, row in df_sample.head(5).iterrows():
            fb = str(row.get("ข้อความความคิดเห็นของลูกค้า", "-"))[:45] + "..."
            table_rows.append([str(row.get("วันที่", "-")), str(row.get("แผนกที่เกี่ยวข้อง", "-")), fb])
        
        t_sample = Table(table_rows, colWidths=[80, 160, 280])
        t_sample.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6)
        ]))
        story.append(t_sample)

    doc.build(story)
    buffer.seek(0)
    return buffer
