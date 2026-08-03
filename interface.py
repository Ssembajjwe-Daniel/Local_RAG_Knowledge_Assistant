import os
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM


# Works locally (defaults to localhost) and in Docker (set via -e OLLAMA_BASE_URL=...)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


st.title("KI-interner Wissensassistent ")


# Cache the setup so it only runs once, not on every question.
# NOTE: if you add a new PDF to documents/, use the "Clear cache" option
# in Streamlit's menu (or restart the app) so it gets picked up.
@st.cache_resource
def load_database():

    # Load every PDF in the documents folder
    pdf_loader = DirectoryLoader("documents/", glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    # Load every .txt file in the documents folder too
    txt_loader = DirectoryLoader(
        "documents/", glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    txt_documents = txt_loader.load()

    documents = pdf_documents + txt_documents

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings (dedicated embedding model, not llama3.1)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)

    # Store vectors
    database = FAISS.from_documents(chunks, embeddings)

    return database


@st.cache_resource
def load_model():
    return OllamaLLM(model="llama3.1", base_url=OLLAMA_URL)


database = load_database()
model = load_model()


def agent(question):

    docs = database.similarity_search(question, k=3)

    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
    Answer the question using this document:

    {context}

    Question:
    {question}
    """

    return model.invoke(prompt)


question = st.text_input("Ask a question about your notes:")

if question:
    with st.spinner("Thinking..."):
        answer = agent(question)

    st.subheader("Answer:")
    st.write(answer)