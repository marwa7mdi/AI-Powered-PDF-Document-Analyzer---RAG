"""
Title: AI-Powered PDF Document Analyzer (RAG)
Author: Marwa Hamdi
Role: Data Analytics Consultant & Trainer
Description: This script uses LangChain and Google Gemini to perform 
             semantic search and information extraction from PDF reports.
"""

import os
from google.colab import userdata

# 1. SETUP & ENVIRONMENT
def setup_environment():
    # Set API Key from Colab Secrets
    os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
    print("✅ Environment keys configured.")

# 2. CORE LIBRARIES
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 3. RAG PIPELINE CLASS
class DocumentIntelligence:
    def __init__(self, file_path):
        self.file_path = file_path
        self.chunks = []
        self.vector_db = None
        self.qa_chain = None

    def load_and_split(self, start_page=21, end_page=30):
        """Loads specific pages to optimize for API quota limits."""
        loader = PyPDFLoader(self.file_path)
        data = loader.load()
        # Segmenting pages for efficiency
        selected_data = data[start_page:end_page]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.chunks = splitter.split_documents(selected_data)
        print(f"✅ Document split into {len(self.chunks)} chunks.")

    def build_vector_store(self):
        """Creates a vector database using Google Embeddings."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory="./rag_storage"
        )
        print("✅ Vector DB initialized and persisted.")

    def create_qa_chain(self):
        """Sets up the retrieval-augmented generation chain."""
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer based ONLY on the provided context.

{context}

Question: {question}
Helpful Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        print("✅ QA Chain is ready.")

    def ask(self, query):
        """Executes a query against the document."""
        if not self.qa_chain:
            return "QA Chain not initialized."
        return self.qa_chain.invoke(query)["result"]

# 4. EXECUTION
if __name__ == "__main__":
    setup_environment()
    
    # Initialize the system
    analyzer = DocumentIntelligence("/content/google_report.pdf")
    
    # Run Pipeline
    analyzer.load_and_split()
    analyzer.build_vector_store()
    analyzer.create_qa_chain()
    
    # Example Query
    question = "Summary of approach to climate adaptation and resilience?"
    answer = analyzer.ask(question)
    
    print("\n" + "="*30)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("="*30)
