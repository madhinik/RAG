import os
import pandas as pd
from docx import Document
import PyPDF2
import cohere
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
def extract_text_from_file(path):
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
        return ["\n".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    
    elif path.endswith(".txt"):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().split('\n\n')
    
    elif path.endswith(".docx"):
        doc = Document(path)
        return [para.text for para in doc.paragraphs if para.text.strip()]
    
    elif path.endswith(".pdf"):
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return [page.extract_text() for page in reader.pages if page.extract_text()]
    
    else:
        raise ValueError(" Unsupported file format.")
API_KEY = "x6OWA......................"  

class CohereEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = cohere.Client("1qGko1T3JkYIcZAxXpyWtQRHhL2cxhspcujIk9LF")

    def embed_documents(self, texts):
        return self.client.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings

    def embed_query(self, text):
        return self.client.embed(texts=[text], model="embed-english-v3.0", input_type="search_query").embeddings[0]
def load_or_create_vectorstore(chunks, embedding_model, persist_dir="./vector_db"):
    if os.path.exists(persist_dir):
        print(" Loading existing vector store...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        print(" Creating new vector store...")
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=persist_dir)
        vectorstore.persist()
        return vectorstore
def retrieve_context(vectorstore, query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def get_answer_from_cohere(query, retrieved_docs):
    co = cohere.Client(API_KEY)
    response = co.chat(
        model="command-a-03-2025",
        message=query,
        documents=[{"text": doc} for doc in retrieved_docs]
    )
    return response.text
if __name__ == "__main__":
    file_path = input(" Enter file path (e.g., data.pdf): ").strip()
    if not os.path.exists(file_path):
        print(" File not found.")
        exit()

    print(" Extracting text...")
    chunks = extract_text_from_file(file_path)
    print(f" {len(chunks)} chunks extracted.")

    print(" Initializing embeddings...")
    embedding_model = CohereEmbeddings(API_KEY)

    print(" Vector store setup...")
    vectorstore = load_or_create_vectorstore(chunks, embedding_model)

    while True:
        query = input("\n Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        retrieved_docs = retrieve_context(vectorstore, query)
        print("\n Top Results:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"{i}. {doc}\n")
        answer = get_answer_from_cohere(query, retrieved_docs)
        print(" Answer:\n", answer)

