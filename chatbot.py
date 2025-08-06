from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Llama 3 model and embeddings
llm = Ollama(model="llama3")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global variables
rag_chain = None
chat_history = []

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load and process documents
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

# Create vector store
def create_vector_store(documents):
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Initialize RAG chain
def initialize_rag_chain(vector_store):
    template = """Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."
    Context: {context}
    Question: {question}
    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

# HTML template as a string
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
        }
        .bot-message {
            background-color: #f0f0f0;
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
        }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-4">
        <h1 class="text-2xl font-bold mb-4">RAG Chatbot with Llama 3</h1>
        <form id="upload-form" enctype="multipart/form-data" class="mb-4">
            <input type="file" id="file-input" name="file" accept=".pdf" class="mb-2">
            <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded">Upload PDF</button>
        </form>
        <div id="upload-status" class="mb-4"></div>
        <div class="chat-container">
            {% for message in chat_history %}
                <div class="{{ 'user-message' if message.role == 'user' else 'bot-message' }}">
                    <strong>{{ message.role.capitalize() }}:</strong> {{ message.content }}
                </div>
            {% endfor %}
        </div>
        <div class="flex">
            <input id="chat-input" type="text" class="flex-1 border p-2 rounded-l" placeholder="Ask a question...">
            <button id="send-button" class="bg-blue-500 text-white px-4 py-2 rounded-r">Send</button>
        </div>
    </div>
    <script>
        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const statusDiv = document.getElementById('upload-status');
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = '<p class="text-green-500">File uploaded successfully!</p>';
                } else {
                    statusDiv.innerHTML = `<p class="text-red-500">Error: ${result.error}</p>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
            }
        });

        document.getElementById('send-button').addEventListener('click', async () => {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const result = await response.json();
                if (response.ok) {
                    const chatContainer = document.querySelector('.chat-container');
                    result.chat_history.forEach(msg => {
                        const div = document.createElement('div');
                        div.className = msg.role === 'user' ? 'user-message' : 'bot-message';
                        div.innerHTML = `<strong>${msg.role.charAt(0).toUpperCase() + msg.role.slice(1)}:</strong> ${msg.content}`;
                        chatContainer.appendChild(div);
                    });
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    input.value = '';
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    from flask import render_template_string
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

@app.route('/upload', methods=['POST'])
def upload_file():
    global rag_chain
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            documents = load_documents(file_path)
            vector_store = create_vector_store(documents)
            rag_chain = initialize_rag_chain(vector_store)
            os.remove(file_path)  # Clean up
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
        except Exception as e:
            os.remove(file_path)  # Clean up on error
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global rag_chain
    if not rag_chain:
        return jsonify({'error': 'Please upload a document first'}), 400
    data = request.get_json()
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    try:
        response = rag_chain.run(user_input)
        chat_history.append({'role': 'user', 'content': user_input})
        chat_history.append({'role': 'assistant', 'content': response})
        return jsonify({'response': response, 'chat_history': chat_history}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)