import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

class FinalReportGenerator:
    def __init__(self, output_path="full_analysis.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='AnalysisHeader', fontSize=14, fontName='Helvetica-Bold', textColor=colors.darkblue, spaceAfter=10))
        self.styles.add(ParagraphStyle(name='MetricLabel', fontSize=10, fontName='Helvetica-Bold', textColor=colors.gray))
        self.styles.add(ParagraphStyle(name='MetricValue', fontSize=12, fontName='Helvetica-Bold', textColor=colors.black))
        self.styles.add(ParagraphStyle(name='AnalysisBody', fontSize=10, fontName='Helvetica', leading=12))
        self.styles.add(ParagraphStyle(name='CodeData', fontSize=8, fontName='Courier', leftIndent=20, textColor=colors.darkgreen))
        self.styles.add(ParagraphStyle(name='PageLabel', fontSize=8, fontName='Helvetica-Oblique', textColor=colors.gray, alignment=2))

    def generate(self, data, metrics=None):
        print(f"📄 Generating Full PDF: {self.output_path}...")
        story = []
        
        # --- PAGE 1: SYSTEM METRICS DASHBOARD ---
        story.append(Paragraph("QuickSight AI: Ingestion Metrics Report", self.styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        
        if metrics:
            # 1. High Level Stats Row
            data_summary = [
                [Paragraph("Processing Time", self.styles['MetricLabel']), 
                 Paragraph("Total Pages", self.styles['MetricLabel']),
                 Paragraph("Visual Confidence", self.styles['MetricLabel'])],
                [Paragraph(metrics.get('duration', 'N/A'), self.styles['MetricValue']),
                 Paragraph(str(metrics.get('total_pages', 0)), self.styles['MetricValue']),
                 Paragraph(f"{metrics.get('avg_confidence', 0)*100:.1f}%", self.styles['MetricValue'])]
            ]
            t_summary = Table(data_summary, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
            t_summary.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('topPadding', (0,0), (-1,-1), 10),
                ('bottomPadding', (0,0), (-1,-1), 15),
                ('LINEBELOW', (0,0), (-1,-1), 1, colors.lightgrey),
            ]))
            story.append(t_summary)
            story.append(Spacer(1, 0.4*inch))

            # 2. Detailed Pipeline Comparison (YOLO vs Docling)
            story.append(Paragraph("Pipeline Performance: Detection Audit", self.styles['AnalysisHeader']))
            
            # Simple text summary of the audit
            audit_text = f"""
            <b>Visual Elements Detected:</b> {metrics.get('total_visuals', 0)}<br/>
            <b>Tables Extracted:</b> {metrics.get('total_tables', 0)}<br/>
            <b>Text Segments:</b> {metrics.get('total_text', 0)}<br/><br/>
            This pipeline uses a Hybrid Routing architecture. The 'Visual Elements' count represents 
            complex figures intercepted by the YOLOv8 Auditor and processed by the Vision Agent.
            """
            story.append(Paragraph(audit_text, self.styles['AnalysisBody']))
            story.append(Spacer(1, 0.5*inch))
            
            story.append(PageBreak())

        # --- EXISTING CONTENT GENERATION ---
        # (Paste your existing grouping/sorting logic here exactly as before)
        # --- 1. BUNCH BY PAGE (Grouping) ---
        pages_bucket = {}
        for item in data:
            p = item.get('page', 1)
            if p not in pages_bucket: pages_bucket[p] = []
            pages_bucket[p].append(item)

        sorted_page_nums = sorted(pages_bucket.keys())
        
        for p_num in sorted_page_nums:
            items = pages_bucket[p_num]

            # --- SORT LOGIC ---
            has_bboxes = all('bbox' in x for x in items)
            if has_bboxes:
                items.sort(key=lambda x: x['bbox'][3], reverse=True)
            else:
                items.sort(key=lambda x: x.get('order_id', 0))

            # --- RENDER PAGE ---
            if p_num > sorted_page_nums[0]: # Page break for subsequent pages
                story.append(PageBreak())
                
            story.append(Paragraph(f"--- Source Page {p_num} ---", self.styles['PageLabel']))
            story.append(Spacer(1, 0.2*inch))

            for item in items:
                # TEXT
                if item['type'] == 'text':
                    story.append(Paragraph(item['content'], self.styles['Normal']))
                    story.append(Spacer(1, 0.15*inch))
                
                # HEADER
                elif item['type'] == 'header':
                    story.append(Paragraph(item['content'], self.styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))

                # TABLE
                elif item['type'] == 'table':
                    story.append(Paragraph("<b>[Table extracted]</b>", self.styles['Normal']))
                    story.append(Paragraph(item['content'].replace('\n', '<br/>'), self.styles['CodeData']))
                    story.append(Spacer(1, 0.2*inch))

                # VISUAL
                elif item['type'] == 'visual':
                    img_path = item.get('file_path')
                    if img_path and os.path.exists(img_path):
                        img = RLImage(img_path, width=5.5*inch, height=3.5*inch, kind='proportional')
                        story.append(img)
                        story.append(Spacer(1, 0.1*inch))
                    
                    analysis = item.get('analysis', {})
                    content = analysis.get('content', {})
                    
                    conf_score = analysis.get('confidence_score', 0.0)
                    conf_reason = analysis.get('confidence_reason', '')
                    
                    if conf_score > 0.8:
                        conf_text = f"<font color='green'><b>High Confidence ({int(conf_score*100)}%)</b></font>"
                    elif conf_score > 0.5:
                        conf_text = f"<font color='orange'><b>Medium Confidence ({int(conf_score*100)}%)</b></font>"
                    else:
                        conf_text = f"<font color='red'><b>Low Confidence ({int(conf_score*100)}%)</b></font>"

                    heading = analysis.get('heading', 'Visual Analysis')
                    overview = content.get('overview', 'No description available.')
                    
                    findings_text = ""
                    for f in content.get('key_findings', []):
                        findings_text += f"• {f}<br/>"

                    analysis_content = f"""
                    <b>{heading}</b> &nbsp;&nbsp;|&nbsp;&nbsp; {conf_text}<br/><br/>
                    {overview}<br/><br/>
                    {findings_text}<br/><br/>
                    <i><font size=8 color='grey'>Reasoning: {conf_reason}</font></i>
                    """
                    
                    p = Paragraph(analysis_content, self.styles['AnalysisBody'])
                    t = Table([[p]], colWidths=[6.5*inch])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), colors.aliceblue),
                        ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
                        ('LEFTPADDING', (0,0), (-1,-1), 10),
                        ('RIGHTPADDING', (0,0), (-1,-1), 10),
                        ('TOPPADDING', (0,0), (-1,-1), 10),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.3*inch))

        doc = SimpleDocTemplate(self.output_path, pagesize=letter)
        doc.build(story)
        print(f"✅ Saved Full PDF to {self.output_path}")