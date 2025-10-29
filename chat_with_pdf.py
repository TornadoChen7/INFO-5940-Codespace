import os
import io
import streamlit as st
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from pypdf import PdfReader

# Define API Key and Base URL
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://api.ai.it.cornell.edu/v1"

# Frontend messages
st.set_page_config(page_title="RAG File Q&A", page_icon="📝")
st.title("RAG File Q&A")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None


# Enable multiple files upload and .pdf/.txt extension upload
uploaded_files = st.file_uploader(
    "Upload one or more documents (.txt, .md, .pdf)",
    type=("txt", "md", "pdf"),
    accept_multiple_files=True,
)

# Use sidebar to change the settings of the RAG system and the GenAI model
with st.sidebar:
    st.header("Settings")
    # Controls for text splitting
    chunk_size = st.slider("Chunk size", 300, 3000, 1200, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 10)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 4, 1)

    # LLM behavior setting
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)

    # Model Selection
    model_name = st.selectbox("Model", options=["openai.gpt-4o-mini"], index=0)

    # Button to clear chat history and reset session state
    if st.button("Clear chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": "History cleared."}]
        st.session_state["vectorstore"] = None
        st.session_state["retriever"] = None
        st.session_state["uploaded_filenames"] = []
        st.rerun()

# Display all chat messages stored in the session state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Read and decode a text or markdown file from uploaded bytes
def read_txt_or_md(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-16", errors="ignore")

def read_pdf(file_bytes: bytes) -> str:
    '''
    Read and extract text content from a PDF file.
    '''
    text = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        #Extract text from PDF
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)

def load_documents(files) -> List[Document]:
    '''
    Load and parse uploaded files into a list of Document objects.
    '''
    docs = []
    for f in files:
        name = f.name
        b = f.getvalue()
        # Read .txt and .md files
        if name.lower().endswith((".txt", ".md")):
            c = read_txt_or_md(b)
        # Read .pdf files
        elif name.lower().endswith(".pdf"):
            c = read_pdf(b)
        else:
            continue
        if not c.strip():
            continue
        # Add text
        docs.append(Document(page_content=c, metadata={"source": name}))
    return docs

def build_index(files):
    '''
    Build a vector-based retrieval index from uploaded documents.
    '''
    docs = load_documents(files)
    # Split the documents into smaller overlapping text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    # Generate vector embeddings
    embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-small")
    # Store the embeddings in a Chroma vector database
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
    # Create a retriever object
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return vectorstore, retriever

# When files are uploaded and the user clicks "Build index", build the retrieval index.
if uploaded_files:
    if st.button("Build index"):
        with st.spinner("Building index..."):
            vs, rt = build_index(uploaded_files)
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"] = rt
        st.success("Index built successfully.")

# Initialize the LLM
llm = ChatOpenAI(model=model_name, temperature=temperature)

def answer_stream(question: str):
    '''
    Generate a streamed answer to the user’s question.
    '''
    retriever = st.session_state["retriever"]
    # Use the retriever to get the most relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])
    # Prepare the message sequence for the LLM
    messages = [
        ("system", "Answer using only the provided context. If not found, say you don't know. Provide brief sources list."),
        ("user", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    # Stream the LLM’s response
    for chunk in llm.stream(messages):
        if getattr(chunk, "content", None):
            yield chunk.content
    yield "\n"

def render_sources(question: str):
    '''
    Display the retrieved document sources under the answer.
    '''
    docs = st.session_state["retriever"].get_relevant_documents(question)
    if not docs:
        return
    st.markdown("---")
    st.markdown("**Sources**")
    # Iterate through the retrieved documents and display each one
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        preview = (d.page_content[:200] + "…") if len(d.page_content) > 200 else d.page_content
        # Display each document inside an expandable section
        with st.expander(f"[{i}] {src}", expanded=False):
            st.write(preview)

# Main chat input
question = st.chat_input("Ask something about the uploaded documents...", disabled=st.session_state["retriever"] is None)

if question:
    # Save and display user's message.
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        response = st.write_stream(answer_stream(question))
        render_sources(question)
    #  Save assistant's response to session state.
    st.session_state["messages"].append({"role": "assistant", "content": response})