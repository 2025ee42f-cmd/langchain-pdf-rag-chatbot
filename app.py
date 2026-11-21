import os
from dotenv import load_dotenv

# 1. Load the API key from .env
load_dotenv(override=True)

# 2. LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Import chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Free Local Embeddings + Free Groq LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 3. Run the Project ---
print("\nLoading document...")
if not os.path.exists("data/document.pdf"):
    print("ERROR: 'data/document.pdf' not found. Please place a PDF inside the 'data' folder.")
else:
    loader = PyPDFLoader("data/document.pdf")
    docs = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    print("Creating and populating vector store locally (first run downloads a small ~80MB model)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Building retrieval chain...")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    system_prompt = (
        "You are a helpful domain-specific assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, say 'I don't know based on the provided document.' \n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nChatbot is ready! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down chatbot...")
            break

        try:
            response = rag_chain.invoke({"input": user_input})
            print(f"Bot: {response['answer']}")
        except Exception as e:
            print(f"An error occurred during retrieval: {e}")
