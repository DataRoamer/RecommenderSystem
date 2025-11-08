"""
Create PDF instructions for the EDA Tool
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.colors import HexColor, black, blue, green, orange, red
import os

def create_pdf_instructions():
    """Create a PDF with instructions for using the EDA Tool"""

    # Create PDF document
    doc = SimpleDocTemplate("EDA_Tool_Instructions.pdf", pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)

    # Get sample styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor('#1f77b4'),
        alignment=1  # Center alignment
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=HexColor('#ff7f0e')
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=HexColor('#2ca02c')
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=10,
        textColor=HexColor('#d62728'),
        backColor=HexColor('#f8f8f8'),
        borderColor=HexColor('#cccccc'),
        borderWidth=1,
        borderPadding=8
    )

    # Story elements
    story = []

    # Title
    story.append(Paragraph("📊 EDA Tool - Quick Start Instructions", title_style))
    story.append(Spacer(1, 20))

    # Getting Started Section
    story.append(Paragraph("🚀 Getting Started", heading_style))

    # Step 1
    story.append(Paragraph("Step 1: Navigate to the Tool Directory", subheading_style))
    story.append(Paragraph("Open Command Prompt or Terminal and navigate to the EDA tool folder:", styles['Normal']))
    story.append(Preformatted("cd C:\\Astreon\\eda_tool", code_style))
    story.append(Spacer(1, 12))

    # Step 2
    story.append(Paragraph("Step 2: Install Dependencies (First Time Only)", subheading_style))
    story.append(Preformatted("pip install -r requirements.txt", code_style))
    story.append(Spacer(1, 12))

    # Step 3
    story.append(Paragraph("Step 3: Run the Application", subheading_style))
    story.append(Paragraph("Use any of these methods:", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Method A: Python Module (Recommended)</b>", styles['Normal']))
    story.append(Preformatted("python -m streamlit run app.py", code_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Method B: Double-click</b>", styles['Normal']))
    story.append(Paragraph("• Double-click on <i>run_app.bat</i> file in the folder", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Method C: Python Launcher</b>", styles['Normal']))
    story.append(Preformatted("python run_app.py", code_style))
    story.append(Spacer(1, 20))

    # Accessing the Tool
    story.append(Paragraph("🌐 Accessing the Tool", heading_style))
    story.append(Paragraph("1. After running, Streamlit will start a web server", styles['Normal']))
    story.append(Paragraph("2. Your browser will automatically open to: <b>http://localhost:8501</b>", styles['Normal']))
    story.append(Paragraph("3. If it doesn't open automatically, copy the URL from the terminal", styles['Normal']))
    story.append(Spacer(1, 20))

    # Using the Tool
    story.append(Paragraph("📁 Using the Tool", heading_style))

    story.append(Paragraph("Upload Data", subheading_style))
    story.append(Paragraph("1. Click \"Browse files\" in the upload section", styles['Normal']))
    story.append(Paragraph("2. Select a CSV or Excel file (.csv, .xlsx, .xls)", styles['Normal']))
    story.append(Paragraph("3. The tool will automatically analyze your data", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Navigate Sections", subheading_style))
    story.append(Paragraph("Use the sidebar to navigate between:", styles['Normal']))
    story.append(Paragraph("• <b>📁 Data Upload</b>: Upload and validate files", styles['Normal']))
    story.append(Paragraph("• <b>📊 Data Overview</b>: Basic statistics and preview", styles['Normal']))
    story.append(Paragraph("• <b>📋 Data Quality</b>: Comprehensive quality analysis", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Test with Sample Data", subheading_style))
    story.append(Paragraph("Try the included Titanic dataset:", styles['Normal']))
    story.append(Paragraph("• File location: <i>tests\\test_data\\titanic.csv</i>", styles['Normal']))
    story.append(Paragraph("• This will demonstrate all features", styles['Normal']))
    story.append(Spacer(1, 20))

    # Troubleshooting
    story.append(Paragraph("🔧 Troubleshooting", heading_style))

    story.append(Paragraph("<b>\"streamlit is not recognized\"</b>", styles['Normal']))
    story.append(Paragraph("Use: <i>python -m streamlit run app.py</i> instead", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Module Import Errors</b>", styles['Normal']))
    story.append(Paragraph("Run: <i>pip install -r requirements.txt</i>", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Browser Doesn't Open</b>", styles['Normal']))
    story.append(Paragraph("Manually navigate to: <i>http://localhost:8501</i>", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Permission Errors</b>", styles['Normal']))
    story.append(Paragraph("Run Command Prompt as Administrator", styles['Normal']))
    story.append(Spacer(1, 20))

    # Understanding Results
    story.append(Paragraph("📊 Understanding Results", heading_style))

    story.append(Paragraph("Quality Score (0-100)", subheading_style))
    story.append(Paragraph("• <b>90-100</b>: Excellent 🟢", styles['Normal']))
    story.append(Paragraph("• <b>75-89</b>: Good 🟡", styles['Normal']))
    story.append(Paragraph("• <b>60-74</b>: Fair 🟠", styles['Normal']))
    story.append(Paragraph("• <b>Below 60</b>: Critical Issues 🔴", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analysis Tabs", subheading_style))
    story.append(Paragraph("• <b>Overview</b>: Key metrics and data types", styles['Normal']))
    story.append(Paragraph("• <b>Missing Data</b>: Completeness analysis", styles['Normal']))
    story.append(Paragraph("• <b>Duplicates</b>: Data redundancy check", styles['Normal']))
    story.append(Paragraph("• <b>Outliers</b>: Anomaly detection", styles['Normal']))
    story.append(Paragraph("• <b>Statistics</b>: Detailed column statistics", styles['Normal']))
    story.append(Spacer(1, 20))

    # Stopping the Tool
    story.append(Paragraph("🛑 Stopping the Tool", heading_style))
    story.append(Paragraph("Press <b>Ctrl+C</b> in the terminal to stop the application", styles['Normal']))
    story.append(Spacer(1, 30))

    # Footer
    story.append(Paragraph("Need Help? Check the detailed README.md file for more information.",
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10,
                                       textColor=HexColor('#666666'), alignment=1)))

    # Build PDF
    doc.build(story)
    print("PDF instructions created: EDA_Tool_Instructions.pdf")

if __name__ == "__main__":
    create_pdf_instructions()