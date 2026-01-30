import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class FinalReportGenerator:
    def __init__(self, output_path="full_analysis.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='AnalysisHeader', fontSize=12, fontName='Helvetica-Bold', textColor=colors.darkblue, spaceAfter=6))
        self.styles.add(ParagraphStyle(name='AnalysisBody', fontSize=10, fontName='Helvetica', leading=12))
        self.styles.add(ParagraphStyle(name='CodeData', fontSize=8, fontName='Courier', leftIndent=20, textColor=colors.darkgreen))
        self.styles.add(ParagraphStyle(name='PageLabel', fontSize=8, fontName='Helvetica-Oblique', textColor=colors.gray, alignment=2))

    def generate(self, data):
        print(f"📄 Generating Full PDF: {self.output_path}...")
        story = []
        story.append(Paragraph("Full Document Analysis Report", self.styles['Title']))
        story.append(Spacer(1, 0.5*inch))

        # --- 1. BUNCH BY PAGE (Grouping) ---
        # We group items so we can sort each page individually
        pages_bucket = {}
        for item in data:
            p = item.get('page', 1)
            if p not in pages_bucket: pages_bucket[p] = []
            pages_bucket[p].append(item)

        # --- 2. PROCESS PAGES IN ORDER ---
        sorted_page_nums = sorted(pages_bucket.keys())
        
        for p_num in sorted_page_nums:
            items = pages_bucket[p_num]

            # --- 3. SORT LOGIC (The Fix) ---
            # Try to identify if we need spatial sorting
            # If items have a 'bbox', we sort spatially.
            # Docling Y is Bottom-Left (0 is Bottom, 800 is Top).
            # We want Top-to-Bottom reading order -> Sort by Y DESCENDING.
            
            # Check if bboxes exist
            has_bboxes = all('bbox' in x for x in items)
            
            if has_bboxes:
                # item['bbox'] is [L, B, R, T]. Index 3 is Top Y.
                # Reverse=True gives High Y (Header) first.
                items.sort(key=lambda x: x['bbox'][3], reverse=True)
            else:
                # Fallback to existing order_id if no bboxes (Digital mode fallback)
                items.sort(key=lambda x: x.get('order_id', 0))

            # --- 4. RENDER PAGE ---
            if p_num > sorted_page_nums[0]:
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
                    
                    # Analysis Block
                    analysis = item.get('analysis', {})
                    content = analysis.get('content', {})
                    
                    heading = analysis.get('heading', 'Visual Analysis')
                    overview = content.get('overview', 'No description available.')
                    
                    findings_text = ""
                    for f in content.get('key_findings', []):
                        findings_text += f"• {f}<br/>"

                    analysis_content = f"<b>{heading}</b><br/><br/>{overview}<br/><br/>{findings_text}"
                    
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