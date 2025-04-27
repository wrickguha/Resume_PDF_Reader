# Resume PDF Reader: AI-Powered Document Analysis

Resume PDF Reader is a web application built with Flask and Python that leverages the power of AI to provide intelligent document analysis, with a particular focus on PDF files. It offers tools for interacting with documents in a conversational manner, extracting key information, and comparing multiple documents, making it a valuable tool for various applications, including resume screening and efficient information retrieval.

## Features

* **Chat with PDF:** Engage in a conversational dialogue with your PDF documents. Ask questions about the content and receive accurate, context-aware answers.
* **Extract Resume Info:** Automatically extract key data fields from resumes, such as name, contact information, education, work experience, and skills, into a structured JSON format.
* **Compare Resumes:** Upload and compare multiple resumes side-by-side based on user queries, enabling efficient candidate evaluation and identification of the most qualified applicants.

## Technologies Used

* Python
* Flask
* PyPDF2
* LangChain
* Google's Gemini API (for LLM and embeddings)
* FAISS (for vector search)
* dotenv (for environment variable management)
* werkzeug (for secure file handling)

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/wrickguha/Resume_PDF_Reader.git
    cd Resume_PDF_Reader
    ```

2.  **Install Dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS and Linux
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**

    * Create a `.env` file in the project's root directory.
    * Add your Google Gemini API key to the `.env` file:

        ```
        GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
        ```

    * Obtain your Google Gemini API key from the Google Cloud Console.

4.  **Run the Application:**

    ```bash
    python main.py
    ```

    The application will typically start on `http://127.0.0.1:5000/`.

## Project Structure

├── main.py           # Main Flask application

├── templates/

│   ├── index.html      # Home page

│   ├── pdf_chat.html   # PDF Chat interface

│   ├── extract_info.html # Resume extraction display

│   ├── compare_resumes.html # Resume comparison display

├── static/

│   └── style.css     # CSS stylesheet

├── uploads/          # Directory to store uploaded files (created at runtime)

├── .env              # Environment variables (API key)

├── requirements.txt  # Python dependencies

└── README.md         # Documentation


## Usage

1.  **Home Page (`/`):** Provides an overview of the application and links to the main features.
2.  **Chat with PDF (`/pdf_chat`):**
    * Upload one or more PDF files.
    * Enter your questions in the provided form.
    * The application will use the uploaded PDFs to generate answers.
3.  **Extract Resume Info (`/extract_info`):**
    * Upload a single resume PDF file.
    * The application will extract key information from the resume and display it.
4.  **Compare Resumes (`/compare`):**
    * Upload two or more resume PDF files.
    * Enter a question or comparison criteria.
    * The application will analyze the resumes and provide a comparative summary.

## Important Notes

* Ensure your Google Gemini API key is correctly set up in the `.env` file.
* The `uploads/` directory will be created automatically to store uploaded files.
* The application uses LangChain and FAISS for efficient question answering and document retrieval.

## Video Description




https://github.com/user-attachments/assets/28fa2edd-62f3-4bc5-8cb2-63dab7c544e3




## Contributing

You can add contribution guidelines here if you want to encourage others to contribute to your project

