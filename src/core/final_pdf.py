import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import json

class FinalReportGenerator:
    def __init__(self, output_path="full_analysis.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        # Custom Styles
        self.styles.add(ParagraphStyle(name='AnalysisHeader', fontSize=12, fontName='Helvetica-Bold', textColor=colors.darkblue, spaceAfter=6))
        self.styles.add(ParagraphStyle(name='AnalysisBody', fontSize=10, fontName='Helvetica', leftIndent=10))
        self.styles.add(ParagraphStyle(name='CodeData', fontSize=8, fontName='Courier', leftIndent=20, textColor=colors.darkgreen))

    def generate(self, data):
        print(f"📄 Generating Full PDF: {self.output_path}...")
        story = []
        
        # Title
        story.append(Paragraph("Full Document Analysis Report", self.styles['Title']))
        story.append(Spacer(1, 0.5*inch))

        for item in data:
            # --- TEXT ---
            if item['type'] == 'text':
                story.append(Paragraph(item['content'], self.styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
            
            # --- HEADERS ---
            elif item['type'] == 'header':
                story.append(Paragraph(item['content'], self.styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))

            # --- TABLES ---
            elif item['type'] == 'table':
                story.append(Paragraph("<b>[Table extracted]</b>", self.styles['Normal']))
                story.append(Paragraph(item['content'].replace('\n', '<br/>'), self.styles['CodeData']))
                story.append(Spacer(1, 0.2*inch))

            # --- VISUALS (The Good Stuff) ---
            elif item['type'] == 'visual':
                # 1. Add the Image
                img_path = item.get('file_path')
                if img_path and os.path.exists(img_path):
                    # Resize to fit width (max 6 inches)
                    img = RLImage(img_path, width=5*inch, height=3*inch, kind='proportional')
                    story.append(img)
                    story.append(Spacer(1, 0.1*inch))
                
                # 2. Add Gemini Analysis
                analysis = item.get('analysis', {})
                content = analysis.get('content', {})
                
                # Heading
                header_text = f"🤖 Analysis: {analysis.get('heading', 'Visual')}"
                story.append(Paragraph(header_text, self.styles['AnalysisHeader']))
                
                # Overview
                story.append(Paragraph(f"<b>Overview:</b> {content.get('overview', '')}", self.styles['AnalysisBody']))
                story.append(Spacer(1, 0.05*inch))
                
                # Findings
                findings = content.get('key_findings', [])
                if findings:
                    story.append(Paragraph("<b>Key Findings:</b>", self.styles['AnalysisBody']))
                    for f in findings:
                        story.append(Paragraph(f"• {f}", self.styles['AnalysisBody']))
                
                # Data
                data_dict = content.get('extracted_data', {})
                if data_dict:
                    data_str = json.dumps(data_dict, indent=2).replace('\n', '<br/>').replace(' ', '&nbsp;')
                    story.append(Spacer(1, 0.05*inch))
                    story.append(Paragraph(f"<b>Data:</b><br/>{data_str}", self.styles['CodeData']))

                story.append(Spacer(1, 0.4*inch)) # Space after visual block

        doc = SimpleDocTemplate(self.output_path, pagesize=letter)
        doc.build(story)
        print(f"✅ Saved Full PDF to {self.output_path}")