import pandas as pd
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER ,TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

def dict_to_text(report_dict):
    raw_text = "Report Title\n\n"  # Assuming a static title for all reports
    for key, value in report_dict.items():
        raw_text += f"{key.capitalize()}\n{value}\n\n"
    return raw_text
# 文本预处理函数
def preprocess_text(raw_text):
    #print(raw_text)
    sections = re.split(r'\n\n+', raw_text)  # 根据多个换行符分割文本
    structured_content = {
        "title": sections[0].strip(),
        "sections": [],
        "references": sections[-1].split('\n') if "https://" in sections[-1] else []
    }
    for section in sections[1:-1]:  # 避开标题和参考文献
        header = section.split('\n')[0].strip()
        text = [line.strip() for line in section.split('\n')[1:] if line.strip()]
        structured_content['sections'].append({"header": header, "text": text})
    return structured_content



def create_pdf(content, filename):


    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Registering a CID font for Chinese characters
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    pdfmetrics.registerFontFamily('STSong-Light', normal='STSong-Light', bold='STSong-Light', italic='STSong-Light', boldItalic='STSong-Light')
    #styles['Heading2'].fontName = 'STSong-Light'
    styles['BodyText'].alignment = TA_LEFT
    # Adjust the 'Title' style as needed or create a new style if you modify it often
    styles['Title'].alignment = TA_CENTER
    styles['Title'].fontSize = 24
    styles['Title'].spaceAfter = 20

    Story = []

    # Title Page
    Story.append(Paragraph('%s' % content['title'], styles['Title']))
    Story.append(Spacer(1, 48))  # Adding more space after the title

    # Main Content Sections
    for section in content['sections']:
        Story.append(Paragraph('%s' % section['header'], styles['BodyText']))
        Story.append(Spacer(1, 12))
        paragraph_text = '<br/>'.join(section['text'])  # Ensure paragraphs are separated properly

        #Story.append(Paragraph(paragraph_text, styles['BodyText']))
        Story.append(Paragraph(paragraph_text, styles['BodyText']))
        Story.append(Spacer(1, 12))  # Adjust spacing as per requirement

    # References
    """
    
    Story.append(Paragraph('References', styles['BodyText']))
    Story.append(Spacer(1, 12))
    for ref in content['references']:
        Story.append(Paragraph(ref, styles['BodyText']))
        Story.append(Spacer(1, 6))
    """

    doc.build(Story)









"""

import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics

from reportlab.pdfbase.cidfonts import UnicodeCIDFont


# 文本预处理函数
def preprocess_text(raw_text):
    sections = re.split(r'\n\n+', raw_text)  # 根据多个换行符分割文本
    structured_content = {
        "title": sections[0].strip(),
        "sections": [],
        "references": sections[-1].split('\n') if "https://" in sections[-1] else []
    }
    for section in sections[1:-1]:  # 避开标题和参考文献
        header = section.split('\n')[0].strip()
        text = [line.strip() for line in section.split('\n')[1:] if line.strip()]
        structured_content['sections'].append({"header": header, "text": text})
    return structured_content



def create_pdf(content, filename):




    # Use the registered CID font
    #styles['BodyText'].fontName = 'STSong-Light'
    #styles['Heading2'].fontName = 'STSong-Light'






    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Registering a CID font for Chinese characters
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    pdfmetrics.registerFontFamily('STSong-Light', normal='STSong-Light', bold='STSong-Light', italic='STSong-Light', boldItalic='STSong-Light')
    styles['Heading2'].fontName = 'STSong-Light'
    # Adjust the 'Title' style as needed or create a new style if you modify it often
    styles['Title'].alignment = TA_CENTER
    styles['Title'].fontSize = 24
    styles['Title'].spaceAfter = 20

    Story = []

    # Title Page
    Story.append(Paragraph('<b>%s</b>' % content['title'], styles['Title']))
    Story.append(Spacer(1, 48))  # Adding more space after the title

    # Main Content Sections
    for section in content['sections']:
        Story.append(Paragraph('<b>%s</b>' % section['header'], styles['Heading2']))
        Story.append(Spacer(1, 12))
        paragraph_text = '<br/>'.join(section['text'])  # Ensure paragraphs are separated properly
        Story.append(Paragraph(paragraph_text, styles['BodyText']))
        Story.append(Spacer(1, 24))  # Adjust spacing as per requirement

    # References
    Story.append(Paragraph('<b>References</b>', styles['Heading2']))
    Story.append(Spacer(1, 12))
    for ref in content['references']:
        Story.append(Paragraph(ref, styles['BodyText']))
        Story.append(Spacer(1, 6))

    doc.build(Story)

"""

