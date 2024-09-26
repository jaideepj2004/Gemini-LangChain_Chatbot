from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

app = Flask(__name__)
load_dotenv()

# Set up Google Generative AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Load the API key from .env file
genai.configure(api_key=GOOGLE_API_KEY)

# Load and process the specific PDF file
pdf_loader = PyPDFLoader(r"C:\Users\jaide\OneDrive\Desktop\SS_prototype\SS_prototype\pdfs\Q&A.pdf")  # Point to your PDF file
data = pdf_loader.load()  # Load the content

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

# Create embeddings and vector store for the PDF data
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_index = Chroma.from_texts([t.page_content for t in texts], embeddings).as_retriever()

# Create prompt template
prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    docs = vector_index.get_relevant_documents(question)  # Retrieve relevant documents from PDF
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return jsonify({'response': response['output_text']})

if __name__ == '__main__':
    app.run(debug=True)
