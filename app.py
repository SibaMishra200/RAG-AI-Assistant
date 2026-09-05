import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

# Only one active vector database is maintained.
CHROMA_PATH = Path("chroma_db")

# Hugging Face embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Groq LLM
LLM_MODEL = "openai/gpt-oss-120b"


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD HUGGING FACE EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    huggingface_token = os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )

    if not huggingface_token:

        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN is missing "
            "from your .env file."
        )

    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=huggingface_token,
    )

    return embeddings


# ============================================================
# LOAD GROQ LLM
# ============================================================

@st.cache_resource
def get_llm():

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:

        raise ValueError(
            "GROQ_API_KEY is missing "
            "from your .env file."
        )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=LLM_MODEL,
        temperature=0,
    )

    return llm


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(uploaded_file)

    documents = []

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        if text:

            text = text.strip()

        else:

            text = ""

        if text:

            document = Document(

                page_content=text,

                metadata={
                    "source": uploaded_file.name,
                    "page": page_number + 1,
                },
            )

            documents.append(document)

    return documents, len(reader.pages)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(uploaded_file):

    # --------------------------------------------------------
    # Delete previous ChromaDB
    # --------------------------------------------------------

    if CHROMA_PATH.exists():

        shutil.rmtree(CHROMA_PATH)

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    documents, total_pages = load_pdf(
        uploaded_file
    )

    if not documents:

        raise ValueError(
            "No readable text was found in this PDF.\n\n"
            "The PDF may be scanned or image-based. "
            "OCR may be required."
        )

    # --------------------------------------------------------
    # Text splitting
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = get_embeddings()

    # --------------------------------------------------------
    # Create ChromaDB
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=str(
            CHROMA_PATH
        ),

        collection_name="rag_documents",
    )

    return (
        vectorstore,
        total_pages,
        len(documents),
        len(chunks),
    )


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever(vectorstore):

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    return retriever


# ============================================================
# QUERY ROUTER PROMPT
# ============================================================

router_prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",

            """
You are a query router for a hybrid AI assistant.

The user has uploaded a document and can ask two
types of questions:

1. Questions that require information from the document.
2. General questions that can be answered using your
   own general knowledge and reasoning.

Your task is to decide which type the user's question is.

Return ONLY one word:

RAG

or

GENERAL


Choose RAG when:

- The user asks about the uploaded document.
- The user refers to the document, PDF, report, paper,
  project, article, or its contents.
- The user asks "according to the document".
- The user asks about facts, numbers, methods, results,
  conclusions, names, or details that are likely contained
  in the uploaded document.
- The question clearly relates to the subject matter
  discussed in the document.


Choose GENERAL when:

- The question is unrelated to the uploaded document.
- The user asks general knowledge questions.
- The user asks about programming or technology unrelated
  to the document.
- The user is making casual conversation.
- The user asks for general explanations.
- The answer does not require information from the PDF.


When uncertain, choose GENERAL.

Return ONLY:

RAG

or

GENERAL
"""
        ),

        (
            "human",

            """
User question:

{question}
"""
        ),
    ]
)


# ============================================================
# RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",

            """
You are a helpful AI assistant answering questions
about an uploaded document.

Use ONLY the provided document context.

Do NOT use your general knowledge to answer
document-related questions.

If the answer cannot be found in the provided context,
say:

"I could not find the answer in the document."

Rules:

1. Do not invent information.
2. Do not make unsupported claims.
3. Answer clearly and concisely.
4. When useful, mention the relevant page number.
"""
        ),

        (
            "human",

            """
Document Context:

{context}


Question:

{question}
"""
        ),
    ]
)


# ============================================================
# GENERAL LLM PROMPT
# ============================================================

general_prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",

            """
You are a helpful AI assistant.

Answer the user's question using your own
knowledge and reasoning.

The user may have an uploaded document, but
this question has been determined to be unrelated
to that document.

Do not retrieve information from the document.

Do not mention RAG, retrieval, or document context
unless the user asks about those topics.

Give a natural, helpful, and accurate answer.
"""
        ),

        (
            "human",

            """
User question:

{question}
"""
        ),
    ]
)


# ============================================================
# ROUTE USER QUERY
# ============================================================

def route_query(query):

    llm = get_llm()

    router_input = router_prompt.invoke(
        {
            "question": query
        }
    )

    response = llm.invoke(
        router_input
    )

    route = response.content.strip().upper()

    # --------------------------------------------------------
    # Make the router robust if the model returns extra text.
    # --------------------------------------------------------

    if route.startswith("RAG"):

        return "RAG"

    if route.startswith("GENERAL"):

        return "GENERAL"

    # If router gives unexpected output,
    # default to GENERAL.
    return "GENERAL"


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def format_context(docs):

    context_parts = []

    for index, doc in enumerate(
        docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        page = doc.metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(

            f"""
[Source {index}]
Document: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(query):

    # ========================================================
    # STEP 1 — ROUTE THE QUERY
    # ========================================================

    route = route_query(
        query
    )

    llm = get_llm()

    # ========================================================
    # GENERAL QUESTION
    # ========================================================

    if route == "GENERAL":

        general_input = general_prompt.invoke(

            {
                "question": query
            }
        )

        response = llm.invoke(
            general_input
        )

        return (
            response.content,
            [],
            "GENERAL",
        )

    # ========================================================
    # RAG QUESTION
    # ========================================================

    vectorstore = (
        st.session_state.vectorstore
    )

    retriever = get_retriever(
        vectorstore
    )

    docs = retriever.invoke(
        query
    )

    # --------------------------------------------------------
    # No documents retrieved
    # --------------------------------------------------------

    if not docs:

        return (
            "I could not find the answer in the document.",
            [],
            "RAG",
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = format_context(
        docs
    )

    # --------------------------------------------------------
    # Create RAG prompt
    # --------------------------------------------------------

    rag_input = rag_prompt.invoke(

        {
            "context": context,
            "question": query,
        }
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = llm.invoke(
        rag_input
    )

    return (
        response.content,
        docs,
        "RAG",
    )


# ============================================================
# SIDEBAR — PROJECT INFORMATION
# ============================================================

with st.sidebar:

    st.title("🤖 RAG AI Assistant")

    st.caption(
        "Hybrid Retrieval-Augmented Generation"
    )

    st.divider()

    # ========================================================
    # DEVELOPER
    # ========================================================

    st.header("👨‍💻 Developer")

    st.subheader(
        "Siba Prasad Mishra"
    )

    st.caption(
        "Data Science and AI Engineer"
    )

    st.write(
        "📧 sibamishra200@email.com"
    )

    st.write(
        "📱 +91 9777758678"
    )

    st.write(
        "📍 Bhubaneswar, India"
    )

    st.divider()

    # ========================================================
    # ABOUT
    # ========================================================

    st.header("📖 About")

    st.write(
        """
This project is a Hybrid RAG AI Assistant.

It can understand whether a user's question
requires information from an uploaded document
or whether it can be answered using the LLM's
own knowledge and reasoning.
"""
    )

    st.divider()

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.header("🔗 How It Works")

    st.write(
        """
**1. Upload**

Upload a PDF document.

**2. Extract**

Text is extracted from the PDF.

**3. Chunk**

The document is divided into smaller chunks.

**4. Embed**

Chunks are converted into vectors using
Hugging Face embeddings.

**5. Store**

Embeddings are stored in ChromaDB.

**6. Route**

The LLM determines whether a question is
document-related.

**7. Retrieve**

Only document-related questions are sent
to the vector database.

**8. Generate**

gpt-oss-120b generates the final response.
"""
    )

    st.divider()

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.header("⚙️ Technology Stack")

    st.write(
        "🤖 **LLM**"
    )

    st.caption(
        "gpt-oss-120b · Groq"
    )

    st.write(
        "🧠 **Embeddings**"
    )

    st.caption(
        "all-mpnet-base-v2 · Hugging Face"
    )

    st.write(
        "🗄️ **Vector Database**"
    )

    st.caption(
        "ChromaDB"
    )

    st.write(
        "🔎 **Retriever**"
    )

    st.caption(
        "MMR"
    )

    st.write(
        "🔗 **Framework**"
    )

    st.caption(
        "LangChain"
    )

    st.write(
        "🎨 **Interface**"
    )

    st.caption(
        "Streamlit"
    )

    st.divider()

    # ========================================================
    # TECHNICAL SKILLS
    # ========================================================

    st.header("🛠️ Skills")

    st.write(
        """
**Languages**

Python · SQL

**Data Science**

NumPy · Pandas · Matplotlib · Seaborn ·
Scikit-learn

**Machine Learning**

Regression · Classification · Clustering ·
Feature Engineering · EDA · Model Evaluation

**Generative AI**

LLMs · Prompt Engineering · RAG ·
AI Application Development

**Frameworks & Tools**

Streamlit · FastAPI · Git · GitHub ·
Google Colab · Jupyter Notebook · VS Code
"""
    )

    st.divider()

    # ========================================================
    # PROJECTS
    # ========================================================

    st.header("🚀 Other Projects")

    st.write(
        "**Mental Health Prediction System**"
    )

    st.caption(
        "Machine Learning · Streamlit · XGBoost · "
        "Random Forest"
    )

    st.write(
        "**Dietary Pattern Discovery & "
        "Chronic Disease Risk Prediction**"
    )

    st.caption(
        "Hybrid AI · Machine Learning · Deep Learning · "
        "Generative AI · RAG"
    )

    st.divider()

    # ========================================================
    # CERTIFICATIONS
    # ========================================================

    st.header("🏆 Certifications")

    st.write(
        "**PwC Technology & Transformation "
        "Launchpad Program**"
    )

    st.caption(
        "July 2025 · PwC AC India"
    )

    st.write(
        "**Tata Elxsi Teliport Season 3**"
    )

    st.caption(
        "June 2026 · Team Tech Yuvas"
    )

    st.divider()

    # ========================================================
    # EDUCATION
    # ========================================================

    st.header("🎓 Education")

    st.write(
        "**B.Tech — Computer Science & Engineering**"
    )

    st.caption(
        "ITER, Siksha 'O' Anusandhan University"
    )

    st.caption(
        "CGPA: 9.3 · 2022–2026"
    )

    st.divider()

    # ========================================================
    # PROJECT LINKS
    # ========================================================

    st.header("🔗 Connect")

    st.write(
        "⭐ GitHub"
    )

    st.caption(
        "https://github.com/SibaMishra200"
    )

    st.write(
        "💼 LinkedIn"
    )

    st.caption(
        "https://www.linkedin.com/in/siba-mishra-078b6020a/"
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("📚 Ask Your Documents")

st.write(
    """
Upload a PDF and ask questions about it.
For unrelated questions, the assistant can respond
using its own knowledge without retrieving from the PDF.
"""
)


# ============================================================
# STEP 1 — UPLOAD PDF
# ============================================================

st.header("1️⃣ Upload Your PDF")

uploaded_file = st.file_uploader(

    "Choose a PDF document",

    type=["pdf"],

    help="Maximum supported file size: 150 MB.",
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file:

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.success(
        f"📄 **{uploaded_file.name}** "
        f"· {file_size_mb:.1f} MB"
    )

    # --------------------------------------------------------
    # File size validation
    # --------------------------------------------------------

    if file_size_mb > 150:

        st.error(
            "❌ The uploaded file is larger than "
            "the 150 MB limit."
        )

    else:

        st.header(
            "2️⃣ Build Knowledge Base"
        )

        st.write(
            """
The document will be processed using:

**PDF → Text Extraction → Chunking → "
            "Embeddings → ChromaDB**
"""
        )

        build_database = st.button(

            "🚀 Build Knowledge Base",

            type="primary",

            use_container_width=True,
        )

        if build_database:

            try:

                with st.status(
                    "Building your knowledge base...",
                    expanded=True,
                ):

                    # ----------------------------------------
                    # PDF
                    # ----------------------------------------

                    st.write(
                        "📖 Extracting text from PDF..."
                    )

                    (
                        vectorstore,
                        total_pages,
                        text_pages,
                        chunks,
                    ) = create_vector_database(
                        uploaded_file
                    )

                    # ----------------------------------------
                    # Store state
                    # ----------------------------------------

                    st.session_state.vectorstore = (
                        vectorstore
                    )

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.document_stats = {

                        "pages":
                            total_pages,

                        "text_pages":
                            text_pages,

                        "chunks":
                            chunks,
                    }

                    # New document = new conversation
                    st.session_state.messages = []

                    st.write(
                        "✂️ Document split into chunks."
                    )

                    st.write(
                        "🧠 Hugging Face embeddings created."
                    )

                    st.write(
                        "🗄️ ChromaDB vector database created."
                    )

                st.success(
                    "🎉 Knowledge base created successfully!"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"""
❌ Failed to process the document.

Error:

{e}
"""
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if (
    st.session_state.vectorstore
    and st.session_state.document_stats
):

    st.divider()

    st.header(
        "📊 Document Information"
    )

    stats = st.session_state.document_stats

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Pages",
            stats["pages"],
        )

    with col2:

        st.metric(
            "Text Pages",
            stats["text_pages"],
        )

    with col3:

        st.metric(
            "Chunks",
            stats["chunks"],
        )

    with col4:

        st.metric(
            "Status",
            "🟢 Ready",
        )

    st.caption(
        f"Current document: "
        f"**{st.session_state.document_name}**"
    )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.vectorstore:

    st.divider()

    st.header(
        "🚀 Get Started"
    )

    st.info(
        """
Upload a PDF above and click
**Build Knowledge Base**.

Once the knowledge base is ready,
the chat interface will appear here.
"""
    )

    st.subheader(
        "🔗 RAG Pipeline"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.write("📄")

        st.caption(
            "PDF"
        )

    with col2:

        st.write("✂️")

        st.caption(
            "Chunking"
        )

    with col3:

        st.write("🧠")

        st.caption(
            "Embeddings"
        )

    with col4:

        st.write("🗄️")

        st.caption(
            "ChromaDB"
        )

    with col5:

        st.write("🤖")

        st.caption(
            "LLM"
        )


# ============================================================
# CHAT
# ============================================================

if st.session_state.vectorstore:

    st.divider()

    st.header(
        "3️⃣ Chat with Your Document"
    )

    st.caption(
        "Document questions use RAG. "
        "General questions use the LLM directly."
    )

    # --------------------------------------------------------
    # Display chat history
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # -----------------------------------------------
            # Show sources only for RAG responses
            # -----------------------------------------------

            if (
                message.get("route") == "RAG"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 View Retrieved Sources"
                ):

                    for source in message[
                        "sources"
                    ]:

                        st.markdown(
                            f"### Page {source['page']}"
                        )

                        st.caption(
                            source["source"]
                        )

                        st.write(
                            source["preview"]
                        )

                        st.divider()


    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    query = st.chat_input(
        "Ask a question..."
    )


    if query:

        # ====================================================
        # USER MESSAGE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                query
            )


        # ====================================================
        # ASSISTANT MESSAGE
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    (
                        answer,
                        docs,
                        route,
                    ) = ask_question(
                        query
                    )

                    # ----------------------------------------
                    # Answer
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )


                    # ----------------------------------------
                    # Retrieved sources
                    # ----------------------------------------

                    sources = []

                    if route == "RAG":

                        for doc in docs:

                            preview = (
                                doc.page_content
                                .replace(
                                    "\n",
                                    " ",
                                )
                            )

                            if len(preview) > 300:

                                preview = (
                                    preview[:300]
                                    + "..."
                                )

                            sources.append(
                                {
                                    "page":
                                        doc.metadata.get(
                                            "page",
                                            "?",
                                        ),

                                    "source":
                                        doc.metadata.get(
                                            "source",
                                            "document",
                                        ),

                                    "preview":
                                        preview,
                                }
                            )


                        # ------------------------------------
                        # Display retrieved sources
                        # ------------------------------------

                        if sources:

                            with st.expander(
                                "📚 View Retrieved Sources"
                            ):

                                for source in sources:

                                    st.markdown(
                                        f"### Page "
                                        f"{source['page']}"
                                    )

                                    st.caption(
                                        source["source"]
                                    )

                                    st.write(
                                        source["preview"]
                                    )

                                    st.divider()


                    # ----------------------------------------
                    # Save assistant message
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                answer,

                            "sources":
                                sources,

                            "route":
                                route,
                        }
                    )


                except Exception as e:

                    error_message = (
                        f"❌ Something went wrong:\n\n{e}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                error_message,

                            "sources":
                                [],

                            "route":
                                "GENERAL",
                        }
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 Hybrid RAG AI Assistant · "
    "Built with Streamlit, LangChain, ChromaDB, "
    "Hugging Face & Groq"
)