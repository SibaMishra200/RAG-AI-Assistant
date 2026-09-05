from dotenv import load_dotenv
import os
import groq
from langchain_groq import ChatGroq
#prompt template
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


load_dotenv()  # Load environment variables from .env file


embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

vectorstore = Chroma(
    persist_directory="chroma_db_main",
    embedding_function=embeddings_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4,
                   "fetch_k": 10,
                   "lambda_mult": 0.5}  # 0 is useful for more diverse results, 1 is useful for more relevant results
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document." . then, provide a concise answer to the question based on your knowledge and reasoning.
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag model is ready to use!")

print("enter exit to quit the program")


groq_client = groq.Groq()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b"
)

while True:
    query = input("you : ")
    if query.lower() == "exit":
        print("Exiting the program.")
        break

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("\n AI : ", response.content)

