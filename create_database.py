# load documents
# split documents into chunks
# create embeddings for each chunk
# store the embeddings in a vector database
# query the vector database

from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma



load_dotenv()

#loading the pdf file
file_path_pdf = Path("Document_loaders\\3814959.pdf")
reader = PdfReader(file_path_pdf)
pdf_docs = []

for i, page in enumerate(reader.pages):
    text_content = page.extract_text()
    if text_content:
        doc = Document(
            page_content=text_content,
            metadata={"source": str(file_path_pdf), "page": i}
        )
        pdf_docs.append(doc)

data_pdf = pdf_docs


# text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(data_pdf)


# embedding models 

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# vector database

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="chroma_db_main"
)


