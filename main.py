from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any, List
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Error reading PDF {pdf.filename}: {str(e)}")
            continue
    return text if text else None

# Helper function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, respond with 'answer is not available in the context.'\n
    Context:\n{context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)  # FIXED MODEL NAME
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_resume_info(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "first_name": None,
            "last_name": None,
            "email": None,
            "phone": None,
            "education": [],
            "work_experience_summary": "No text could be extracted from the resume",
            "skills": [],
            "current_position": None,
            "years_of_experience": None
        }

    prompt_template = """
    Analyze the following resume text and extract the requested information in JSON format.
    Be as accurate as possible in identifying each field.

    Required fields:
    - first_name (string): The candidate's first name
    - last_name (string): The candidate's last name
    - email (string): The candidate's email address
    - phone (string): The candidate's phone number
    - education (array of strings): List of educational qualifications
    - work_experience_summary (string): 2-3 sentence summary of work experience
    - skills (array of strings): List of key skills
    - current_position (string): Current or most recent job position
    - years_of_experience (integer): Total years of professional experience

    For any field not found in the resume, set the value to null.

    Resume Text:
    {text}

    Return ONLY the JSON object with no additional text or explanation.
    """

    model = genai.GenerativeModel("gemini-1.5-pro")

    try:
        response = model.generate_content(prompt_template.format(text=text))
        # Clean the response to ensure it's valid JSON
        json_str = response.text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"):
            json_str = json_str[3:-3].strip()

        extracted_data = json.loads(json_str)

        # Validate the extracted data
        required_fields = [
            "first_name", "last_name", "email", "phone",
            "education", "work_experience_summary",
            "skills", "current_position", "years_of_experience"
        ]

        for field in required_fields:
            if field not in extracted_data:
                extracted_data[field] = None

        # Ensure arrays are lists
        if isinstance(extracted_data.get("education"), str):
            extracted_data["education"] = [extracted_data["education"]]
        if isinstance(extracted_data.get("skills"), str):
            extracted_data["skills"] = [extracted_data["skills"]]

        return extracted_data

    except Exception as e:
        print(f"Error parsing resume: {str(e)}")
        return {
            "first_name": None,
            "last_name": None,
            "email": None,
            "phone": None,
            "education": [],
            "work_experience_summary": "Error parsing resume information",
            "skills": [],
            "current_position": None,
            "years_of_experience": None
        }
    
def compare_resumes(pdf_paths: List[str], user_question: str) -> str:
    """
    Compares multiple resumes based on a user's question.

    Args:
        pdf_paths: List of paths to the uploaded PDF files.
        user_question: The question to answer based on the resumes.

    Returns:
        A string containing the comparison and answer.
    """

    all_text = get_pdf_text(pdf_paths)
    if not all_text:
        return "Could not extract text from any of the uploaded PDFs."

    text_chunks = get_text_chunks(all_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    docs = vector_store.similarity_search(user_question, k=len(pdf_paths))  # Retrieve relevant docs

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    if request.method == 'POST':
        user_question = request.form['user_question']
        pdf_files = request.files.getlist("pdf_files")

        # Save uploaded PDFs and extract text
        pdf_paths = []
        for pdf in pdf_files:
            filename = secure_filename(pdf.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf.save(path)
            pdf_paths.append(path)

        raw_text = get_pdf_text(pdf_paths)
        if raw_text:  # Check if raw_text is not None
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_question)

            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            answer = response["output_text"]
        else:
            answer = "Could not extract text from the uploaded PDFs."

        return render_template('pdf_chat.html', answer=answer)

    return render_template('pdf_chat.html', answer=None)

@app.route('/extract_info', methods=['GET', 'POST'])
def extract_info():
    if request.method == 'POST':
        pdf_files = request.files.getlist("pdf_files")

        if not pdf_files:
            flash("Please upload at least one PDF file")
            return redirect(url_for('extract_info'))

        # Limit to one file for resume extraction
        if len(pdf_files) > 1:
            flash("Please upload only one resume at a time")
            return redirect(url_for('extract_info'))

        pdf = pdf_files[0]
        filename = secure_filename(pdf.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf.save(path)

        raw_text = get_pdf_text([path])
        if not raw_text:
            flash("Could not extract any text from the uploaded PDF")
            return redirect(url_for('extract_info'))

        extracted_info = extract_resume_info(raw_text)
        return render_template('extracted_info.html', info=extracted_info)

    return render_template('extract_info.html')


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        user_question = request.form['user_question']
        pdf_files = request.files.getlist('pdf_files')

        if len(pdf_files) < 2:
            flash("Please upload at least two resume files for comparison.")
            return redirect(url_for('compare'))

        pdf_paths = []
        for pdf in pdf_files:
            filename = secure_filename(pdf.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf.save(path)
            pdf_paths.append(path)

        comparison_result = compare_resumes(pdf_paths, user_question)
        return render_template('compare_resumes.html', comparison=comparison_result)

    return render_template('compare_resumes.html', comparison=None)

if __name__ == '__main__':
    app.run(debug=True)