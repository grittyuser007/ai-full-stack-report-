import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from groq import Groq
import fitz

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

client = Groq(api_key=GROQ_API_KEY)

# Ensure the index storage directory exists
index_storage_dir = "index_storage"
docstore_path = os.path.join(index_storage_dir, "docstore.json")
os.makedirs(index_storage_dir, exist_ok=True)
if not os.path.exists(docstore_path):
    with open(docstore_path, "w") as f:
        json.dump({}, f)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {"txt", "pdf", "csv", "xls", "json"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def use_groq_chat_api(model, messages):
    """Call the Groq Chat API."""
    try:
        chat_completion = client.chat.completions.create(messages=messages, model=model)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error using Groq Chat API: {e}")
        return "Error occurred while processing the data with Groq API."

@app.route("/test-api-key", methods=["GET"])
def test_api_key():
    """Test if the API key is loaded correctly."""
    return jsonify({"GROQ_API_KEY": GROQ_API_KEY})

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and process with a prompt via Groq API."""
    if "file" not in request.files or "prompt" not in request.form:
        return jsonify({"error": "File or prompt not provided"}), 400

    file = request.files["file"]
    prompt = request.form["prompt"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        file_extension = filename.rsplit(".", 1)[1].lower()
        try:
            # Process the file as before (e.g., CSV, TXT, PDF, etc.)
            if file_extension == "csv":
                df = pd.read_csv(file_path)
            elif file_extension == "xlsx":
                df = pd.read_excel(file_path)
            elif file_extension == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                df = pd.DataFrame({"Content": [content]})
            elif file_extension == "pdf":
                text = ""
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text()
                df = pd.DataFrame({"Content": [text]})
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        except Exception as e:
            return jsonify({"error": f"Error parsing file: {e}"}), 400

        # Combine file content with the user prompt
        data_string = df.to_string(index=False)
        combined_input = f"User prompt: {prompt}\n\nFile content:\n{data_string}"

        # Query Groq API
        groq_response = use_groq_chat_api(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": combined_input}
            ],
        )

        return jsonify({
            "message": "File and prompt processed successfully.",
            "groq_response": groq_response
        }), 200
    return jsonify({"error": "Invalid file type"}), 400

@app.route("/generate-report", methods=["POST"])
def generate_report():
    """Generate a report based on a query."""
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Load index and query
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

    return jsonify({
        "index_response": str(index_response),
        "groq_response": groq_response
    })

@app.route("/")
def index():
    """Render the upload page."""
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
