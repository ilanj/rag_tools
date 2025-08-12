from typing import TypedDict, List
from loguru import logger
import glob
import os
import shutil

from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS, Chroma


# ----------------------
# 1. Define the State
# ----------------------
class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str


# ----------------------
# 2. Load and embed PDFs
# ----------------------

pdf_files = glob.glob("data/*.pdf")  # Adjust path as needed
logger.info(f"Found {len(pdf_files)} PDF files.")

all_docs = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    all_docs.extend(docs)

logger.info(f"✅ Loaded {len(all_docs)} chunks from {len(pdf_files)} PDFs")

# Embeddings
embedding = OllamaEmbeddings(model="llama3.1")

# ----------------------
# 3. Vector store
# ----------------------

# Remove existing Chroma DB
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Create and persist new Chroma DB
vectorstore = Chroma.from_documents(
    all_docs, embedding, persist_directory="./chroma_db"
)
vectorstore.persist()

# Retrieve more than the default 4 chunks
chunk_window = min(all_docs / 2, 20)
retriever = vectorstore.as_retriever(search_kwargs={"k": chunk_window})

# Load the LLM
llm = ChatOllama(model="llama3.1")


# ----------------------
# 4. Graph steps
# ----------------------
def retrieve_step(state: RAGState) -> RAGState:
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    return {"question": question, "docs": docs}


def generate_step(state: RAGState) -> RAGState:
    question = state["question"]
    docs = state["docs"]
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Answer the question based on the following context:\n\n{context}\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    return {"question": question, "docs": docs, "answer": response.content}


# ----------------------
# 5. Build RAG graph
# ----------------------
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_step)
graph.add_node("generate", generate_step)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

app = graph.compile()

# ----------------------
# 6. Interactive Q&A loop
# ----------------------
logger.info("📘 Ask questions based on the PDF. Type 'exit' to quit.\n")

while True:
    question = input("❓ You: ").strip()
    if question.lower() in ["exit", "quit"]:
        break

    result = app.invoke({"question": question})
    logger.info("🤖 Answer:", result["answer"])
