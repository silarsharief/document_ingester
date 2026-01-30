import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

class DebugReporter:
    def __init__(self, output_filename="debug_report.pdf"):
        self.output_filename = output_filename
        self.story = []
        self.styles = getSampleStyleSheet()
        
        # Custom Styles
        self.styles.add(ParagraphStyle(name='GeminiText', fontSize=10, leading=12, spaceAfter=10))
        self.styles.add(ParagraphStyle(name='HeaderSmall', fontSize=12, fontName='Helvetica-Bold', spaceAfter=5))

        # Add Title Page
        title = Paragraph(f"Visual Intelligence Debug Report", self.styles['Title'])
        timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal'])
        self.story.append(title)
        self.story.append(timestamp)
        self.story.append(Spacer(1, 0.5 * inch))

    def add_comparison(self, page_num, docling_path, yolo_path, gemini_text):
        """
        Adds a Side-by-Side comparison block to the report.
        """
        # 1. Section Header
        self.story.append(Paragraph(f"Page {page_num} - Detected Figure", self.styles['Heading2']))
        
        # 2. Prepare Images (Resize to fit)
        # We enforce a max width/height to keep the PDF clean
        img_width = 3.5 * inch
        img_height = 2.5 * inch
        
        img_docling = Image(docling_path, width=img_width, height=img_height, kind='proportional')
        img_yolo = Image(yolo_path, width=img_width, height=img_height, kind='proportional')

        # 3. Side-by-Side Table
        data = [
            [Paragraph("<b>Before (Docling Crop)</b>", self.styles['Normal']), Paragraph("<b>After (YOLO Fix)</b>", self.styles['Normal'])],
            [img_docling, img_yolo]
        ]
        
        table = Table(data, colWidths=[4 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0,0), (1,0), colors.whitesmoke),
            ('PADDING', (0,0), (-1,-1), 10),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.2 * inch))
        
        # 4. Gemini Analysis Text
        self.story.append(Paragraph("<b>Gemini Analysis:</b>", self.styles['HeaderSmall']))
        self.story.append(Paragraph(gemini_text, self.styles['GeminiText']))
        
        # 5. Break Line
        self.story.append(Spacer(1, 0.5 * inch))
        self.story.append(Paragraph("_" * 60, self.styles['Normal']))
        self.story.append(Spacer(1, 0.5 * inch))

    def save(self):
        """
        Writes the PDF to disk.
        """
        doc = SimpleDocTemplate(
            self.output_filename,
            pagesize=landscape(letter),
            rightMargin=0.5*inch, leftMargin=0.5*inch,
            topMargin=0.5*inch, bottomMargin=0.5*inch
        )
        doc.build(self.story)
        print(f"📄 Report generated: {self.output_filename}")