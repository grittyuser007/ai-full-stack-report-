import os
import json
import pandas as pd
import fitz
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from llama_index.llms.groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for matplotlib
import matplotlib.pyplot as plt

# Load environment variables from .env 
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"  # Specify the model to use

# Initialize Groq client
groq_client = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility function: Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function: Query Groq API
def use_groq_chat_api(prompt):
    try:
        response = groq_client.complete(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error occurred while processing the data with Groq API: {e}"


# Function: Generate PDF with file content and Groq API response
def generate_pdf(prompt, file_content, groq_response, df):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles for better formatting
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle

    custom_body_style = ParagraphStyle(
        'CustomBodyStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,  # Line spacing
        alignment=TA_LEFT,  # Left-align text
        spaceAfter=10  # Space after each paragraph
    )

    custom_heading_style = ParagraphStyle(
        'CustomHeadingStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        alignment=TA_LEFT,
        spaceAfter=12,
    )

    elements = []

    # Add a bold heading
    elements.append(Paragraph("<b>FULL STACK REPORT GENERATING AGENT</b>", custom_heading_style))
    elements.append(Spacer(1, 12))

    # Add Groq API response
    formatted_response = groq_response.replace("\n", "<br />")  # Replace newlines with HTML line breaks
    elements.append(Paragraph(formatted_response, custom_body_style))
    elements.append(Spacer(1, 12))

    # Add comparison plot if there are exactly two columns
    if df.shape[1] == 2:
        try:
            plt.figure()
            df.plot(x=df.columns[0], y=df.columns[1], kind='bar', figsize=(8, 6))
            plt.title('Comparison Plot')
            comparison_plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'comparison_plot.png')
            plt.savefig(comparison_plot_path)
            plt.close()

            elements.append(Paragraph("Comparison Plot from Data:", custom_heading_style))
            elements.append(Image(comparison_plot_path, width=400, height=300))
            elements.append(Spacer(1, 12))
        except Exception as e:
            print(f"Error creating comparison plot: {e}")

    # Add numeric data visualizations
    if not df.empty:
        numeric_df = df.select_dtypes(include=['number'])  # Select numeric columns

        if not numeric_df.empty:
            # Bar plot
            try:
                plot_path_bar = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_bar.png')
                numeric_df.plot(kind='bar', figsize=(8, 6))
                plt.title("Bar Plot of Numeric Data")
                plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
                plt.tight_layout()
                plt.savefig(plot_path_bar)
                plt.close()

                elements.append(Paragraph("Bar Plot of Numeric Data:", custom_heading_style))
                elements.append(Image(plot_path_bar, width=400, height=300))
                elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error creating bar plot: {e}")

            # Pie chart for the first numeric column
            try:
                plot_path_pie = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_pie.png')
                numeric_df.iloc[:, 0].plot(kind='pie', figsize=(8, 6), autopct='%1.1f%%')
                plt.title("Pie Chart of First Numeric Column")
                plt.ylabel("")  # Remove y-axis label
                plt.tight_layout()
                plt.savefig(plot_path_pie)
                plt.close()

                elements.append(Paragraph("Pie Chart of First Numeric Column:", custom_heading_style))
                elements.append(Image(plot_path_pie, width=400, height=300))
                elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error creating pie chart: {e}")

            # Histogram
            try:
                plot_path_hist = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_hist.png')
                numeric_df.plot(kind='hist', figsize=(8, 6), alpha=0.7, bins=10)
                plt.title("Histogram of Numeric Data")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(plot_path_hist)
                plt.close()

                elements.append(Paragraph("Histogram of Numeric Data:", custom_heading_style))
                elements.append(Image(plot_path_hist, width=400, height=300))
                elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error creating histogram: {e}")

    # Build and save the PDF
    doc.build(elements)
    return pdf_path


# Route: File upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        prompt = request.form.get('prompt', '')  # Default to empty string if not provided

        file_extension = filename.rsplit(".", 1)[1].lower()
        try:
            if file_extension == "csv":
                df = pd.read_csv(file_path)
            elif file_extension == "xlsx":
                df = pd.read_excel(file_path)
            elif file_extension == "json":
                df = pd.read_json(file_path)
            elif file_extension == "pdf":
                doc = fitz.open(file_path)
                text = "".join(page.get_text() for page in doc)
                df = pd.DataFrame([text], columns=["Content"])
            elif file_extension == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                df = pd.DataFrame({"Content": [content]})
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400

        file_content = df.to_string(index=False)
        combined_content = f"{prompt}\n\n{file_content}"
        print("User Prompt:\n", prompt)
        print("Combined Content Sent to Groq API:\n", combined_content)

        try:
            groq_response = use_groq_chat_api(combined_content)
        except Exception as e:
            return jsonify({"error": f"Groq API error: {str(e)}"}), 500

        pdf_path = generate_pdf(prompt, file_content, groq_response, df)
        return send_file(pdf_path, as_attachment=True, download_name='report.pdf')

    return jsonify({"error": "Invalid file type"}), 400

# Route: Home
@app.route("/")
def index():
    return render_template("upload.html")

# Main entry point
if __name__ == "__main__":
    app.run(debug=True)
