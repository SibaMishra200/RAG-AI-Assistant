\# RAG AI Assistant



A hybrid Retrieval-Augmented Generation (RAG) AI assistant that allows users to upload a PDF and ask questions about its contents while also supporting general-purpose questions.



The application intelligently routes each query either to the document-based RAG pipeline or directly to the LLM based on the nature of the question.



\## Features



\- PDF document upload through a Streamlit interface

\- Automatic PDF text extraction using PyPDF

\- Recursive text chunking with overlapping chunks

\- Semantic embeddings using `sentence-transformers/all-mpnet-base-v2`

\- ChromaDB vector database for document storage and retrieval

\- Maximum Marginal Relevance (MMR) retrieval

\- Intelligent query routing between:

&#x20; - Document-based RAG

&#x20; - General LLM knowledge

\- Source/page information for document-based answers

\- Protection against unsupported document-based answers

\- Chat-style interaction through Streamlit

\- Supports large PDF uploads through Streamlit configuration



\## How It Works



The application follows a hybrid RAG architecture:



```text

&#x20;                   ┌──────────────────┐

&#x20;                   │    PDF Upload    │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │   PDF Parsing    │

&#x20;                   │     PyPDF        │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │ Text Chunking    │

&#x20;                   │ Recursive Split  │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │    Embeddings    │

&#x20;                   │  Hugging Face    │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │    ChromaDB      │

&#x20;                   │  Vector Store    │

&#x20;                   └──────────────────┘





User Query

&#x20;    │

&#x20;    ▼

┌──────────────────────┐

│    Query Router      │

│      LLM-based       │

└──────────┬───────────┘

&#x20;          │

&#x20;     ┌────┴────┐

&#x20;     │            │

&#x20;    RAG    	 GENERAL

&#x20;     │       	   │

&#x20;     ▼        	   ▼

┌──────────┐ ┌──────────────┐

│ ChromaDB │ │ 		Direct LLM   │

│ Retrieval│ │ 		Knowledge    │

└────┬─────┘ └──────┬───────┘

&#x20;      │              	  │

&#x20;      ▼              	  │

┌──────────────┐     │

│ RAG Prompt   │    	  │

│ + Context    │    	  │

└──────┬───────┘     │

&#x20;     	 │                │

&#x20;     	 └─────┬──────┘

&#x20;           	 ▼

&#x20;     ┌──────────────┐

&#x20;    	 │    Answer    │

&#x20;     └──────────────┘



Query Routing



One of the main features of this project is hybrid query routing.



Before answering a question, the system determines whether the query is:



RAG



Used when the question is related to the uploaded document.



Examples:



What is the main conclusion of the document?



According to the report, what methodology was used?



What results were obtained in the study?



The query is sent to ChromaDB, relevant document chunks are retrieved using MMR, and the retrieved context is provided to the LLM.



GENERAL



Used when the question is unrelated to the uploaded document.



Examples:



What is Python?



Explain machine learning.



What is an API?



These questions are answered directly by the LLM without retrieving document content.



This allows the application to behave as both a document assistant and a general AI assistant.



RAG Pipeline



The document processing pipeline consists of:



PDF upload

PDF text extraction using PyPDF

Document creation with page metadata

Recursive text splitting

Text embeddings generation

Vector storage in ChromaDB

MMR-based retrieval

Context construction

LLM response generation



The text splitter uses:



Chunk size: 1000

Chunk overlap: 200



The retriever uses:



Retrieval method: MMR

k = 4

fetch\_k = 10

lambda\_mult = 0.5

Technology Stack

AI / LLM

Groq

openai/gpt-oss-120b

LangChain

Embeddings

Hugging Face

sentence-transformers/all-mpnet-base-v2

Vector Database

ChromaDB

Document Processing

PyPDF

Application

Streamlit

Programming Language

Python

Project Structure

RAG-AI-Assistant/

│

├── .streamlit/

│   └── config.toml

│

├── app.py

├── create\_database.py

├── main.py

├── requirements.txt

├── .gitignore

└── README.md



Local/private files such as API credentials, virtual environments, uploaded documents, and generated vector databases are excluded from the Git repository using .gitignore.



Installation

1\. Clone the repository

git clone https://github.com/SibaMishra200/RAG-AI-Assistant.git

cd RAG-AI-Assistant

2\. Create a virtual environment

python -m venv .venv



Activate it on Windows:



.venv\\Scripts\\activate

3\. Install dependencies

pip install -r requirements.txt

4\. Configure environment variables



Create a .env file in the project root.



Add your API credentials using the following variable names:



GROQ\_API\_KEY=your\_groq\_api\_key

HUGGINGFACEHUB\_API\_TOKEN=your\_huggingface\_token



Never commit your .env file to GitHub.



5\. Run the application

streamlit run app.py



The application will open in your browser.



Usage

Launch the Streamlit application.

Upload a PDF document.

Build the vector database.

Ask questions about the uploaded document.

Ask general questions unrelated to the document.

The query router determines the appropriate processing path.

Document-based responses can display relevant source/page information.

Example

Document Query

User:

What methodology was used in the document?



System:

RAG → Retrieve relevant chunks → Generate answer

General Query

User:

What is the difference between supervised and unsupervised learning?



System:

GENERAL → Direct LLM response

Key Design Decisions

MMR Retrieval



Maximum Marginal Relevance is used instead of simple similarity retrieval to balance relevance and diversity among retrieved chunks.



Page Metadata



Each extracted PDF page is stored with page metadata so that document-based answers can be associated with their source page.



Hybrid Architecture



The application does not force every question through the vector database. General questions can bypass retrieval and go directly to the LLM.



Environment Variables



API keys are stored outside the source code using environment variables and are excluded from version control.



Limitations

Scanned/image-only PDFs may require OCR before text can be extracted.

The quality of document answers depends on PDF text extraction, chunking, retrieval, and embedding quality.

The query router uses an LLM decision before retrieval, which introduces an additional LLM call.

Hugging Face embeddings require access to the Hugging Face inference service.

Future Improvements

Add OCR support for scanned PDFs

Add streaming LLM responses

Add conversation-aware retrieval

Add hybrid keyword + semantic search

Improve query routing using document-aware relevance scoring

Add document management for multiple uploaded files

Add evaluation metrics for retrieval and answer quality

Add deployment support for cloud environments

Author



Siba Prasad Mishra



Data Science and AI Engineer



Interested in Data Science, Machine Learning, Generative AI, RAG systems, and AI application development.



⭐ If you find this project useful, consider giving the repository a star.

