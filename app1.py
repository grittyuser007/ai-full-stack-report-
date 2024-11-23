import os
import json
import pandas as pd
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Load API key from environment variable
client = Groq(api_key=GROQ_API_KEY)

# Ensure necessary directories exist
index_storage_dir = 'index_storage'
docstore_path = os.path.join(index_storage_dir, 'docstore.json')

if not os.path.exists(index_storage_dir):
    os.makedirs(index_storage_dir)

if not os.path.exists(docstore_path):
    with open(docstore_path, 'w') as f:
        json.dump({}, f)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def use_groq_chat_api(model, messages):
    """Call the Groq Chat API."""
    try:
        chat_completion = client.chat.completions.create(messages=messages, model=model)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error using Groq Chat API: {e}")
        return "Error occurred while processing the data with Groq API."

def generate_pdf(index_response, groq_response, df):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Llama Index Response:", styles['Heading2']))
    elements.append(Paragraph(index_response, styles['BodyText']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Groq API Response:", styles['Heading2']))
    elements.append(Paragraph(groq_response, styles['BodyText']))
    elements.append(Spacer(1, 12))

    if not df.empty:
        # Ensure numeric data is available for plotting
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            print("No numeric data to plot.")
        else:
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png')
            numeric_df.plot(kind='bar', figsize=(8, 6))
            plt.title("Bar Plot")
            plt.savefig(plot_path)
            plt.close()

            # Add the plot to the PDF
            elements.append(Image(plot_path, width=400, height=300))

    doc.build(elements)
    return pdf_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

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
            return jsonify({"error": f"Error parsing file: {e}"}), 400

        file_content = df.to_string()

        try:
            groq_response = use_groq_chat_api(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": file_content}],
            )
        except Exception as e:
            return jsonify({"error": f"Error querying Groq API: {e}"}), 400

        pdf_path = generate_pdf(file_content, groq_response, df)
        return send_file(pdf_path, as_attachment=True, download_name='report.pdf')
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/generate-report', methods=['POST'])
def generate_report():
    data = request.json
    query = data.get('query')

    storage_context = StorageContext.from_defaults(persist_dir=index_storage_dir)
    index = VectorStoreIndex(storage_context=storage_context)
    index_response = index.query(query)

    groq_response = use_groq_chat_api(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(index_response)},
        ],
    )

    pdf_path = generate_pdf(str(index_response), groq_response, pd.DataFrame())
    return send_file(pdf_path, as_attachment=True, download_name='report.pdf')

@app.route("/")
def index():
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
