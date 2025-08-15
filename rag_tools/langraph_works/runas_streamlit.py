import streamlit as st
from typing import TypedDict, List
import glob
import os
import shutil

# Minimal friendly imports wrapped in try/except so the app starts even if deps missing
try:
    from loguru import logger
    from langgraph.graph import StateGraph
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.chat_models import ChatOllama
    from langchain.embeddings import OllamaEmbeddings
    from langchain.vectorstores import Chroma
except Exception as e:  # pragma: no cover - runtime dependency handling
    logger = None
    StateGraph = None
    Document = None
    PyPDFLoader = None
    ChatOllama = None
    OllamaEmbeddings = None
    Chroma = None


# ----------------------
# App config / credentials
# ----------------------
USERNAME = "careerdayuser"
PASSWORD = "mleteam"


class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str


def show_import_help():
    st.error(
        "Required libraries are missing. Install: streamlit, loguru, langgraph, langchain-core, \\n        langchain-community, chromadb and ollama (if using Ollama)."
    )
    st.stop()


def login_section():
    st.title("Login")
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if user == USERNAME and pwd == PASSWORD:
                st.session_state["logged_in"] = True
            else:
                st.error("Invalid credentials")


def ensure_imports():
    # If any core import is missing, show helpful message and stop
    missing = []
    if logger is None:
        missing.append("loguru")
    if StateGraph is None or Document is None:
        missing.append("langgraph / langchain-core")
    if PyPDFLoader is None:
        missing.append("langchain-community (document loaders)")
    if ChatOllama is None or OllamaEmbeddings is None:
        missing.append("langchain-community (ollama) / langchain.embeddings")
    if Chroma is None:
        missing.append("langchain.vectorstores (Chroma) / chromadb")
    if missing:
        st.warning("Missing packages: " + ", ".join(missing))
        show_import_help()


def list_project_folders(data_dir: str = "data"):
    """Return a sorted list of folder names inside `data/` (non-recursive)."""
    if not os.path.exists(data_dir):
        return []
    entries = [
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    entries.sort()
    return entries


def list_project_pdfs(project: str, data_dir: str = "data"):
    """Return a tuple of PDF paths for a given project folder (hashable)."""
    folder = os.path.join(data_dir, project)
    pattern = os.path.join(folder, "*.pdf")
    files = sorted(glob.glob(pattern))
    return tuple(files)


@st.cache_resource
def build_vectorstore_and_llm(
    project: str, pdf_paths: tuple, force_rebuild: bool = False
):
    """
    Build or load a vectorstore + llm for a project.
    - project: name (used for chroma subdir)
    - pdf_paths: tuple of PDF file paths (hashable so cache works)
    - force_rebuild: if True, rebuild even if chroma exists
    Returns (retriever, llm)
    """
    persist_dir = os.path.join("./chroma_db", project)
    embedding = OllamaEmbeddings(model="llama3.1")

    # If persisted DB exists and not forcing rebuild, try to load it
    if os.path.exists(persist_dir) and not force_rebuild:
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir, embedding_function=embedding
            )
        except Exception:
            # fallback to rebuild
            shutil.rmtree(persist_dir)
            vectorstore = None
    else:
        vectorstore = None

    # If vectorstore not available, build from PDFs
    if vectorstore is None:
        all_docs = []
        for p in pdf_paths:
            loader = PyPDFLoader(p)
            docs = loader.load_and_split()
            all_docs.extend(docs)

        # ensure directory exists
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore = Chroma.from_documents(
            all_docs, embedding, persist_directory=persist_dir
        )
        try:
            vectorstore.persist()
        except Exception:
            # some Chroma versions persist on creation
            pass

    # choose k based on docs count
    try:
        doc_count = getattr(vectorstore, "len", None)
        if doc_count is None:
            # try to infer from retriever later; fallback to 4
            chunk_window = 4
        else:
            chunk_window = doc_count
    except Exception:
        chunk_window = 4

    retriever = vectorstore.as_retriever(search_kwargs={"k": chunk_window})
    llm = ChatOllama(model="llama3.1")
    return retriever, llm


def main_ui():
    st.title("PDF Q&A — Streamlit (LangGraph style)")
    st.sidebar.button(
        "Logout", on_click=lambda: st.session_state.update({"logged_in": False})
    )

    st.markdown("---")
    projects = list_project_folders("data")
    if not projects:
        st.info(
            "No project folders found in `data/`. Create folders with PDF files inside."
        )
        return

    # Tabs for each project
    tabs = st.tabs(projects)
    for project, tab in zip(projects, tabs):
        with tab:
            st.header(project)
            pdf_paths = list_project_pdfs(project, "data")
            if not pdf_paths:
                st.info(
                    f"No PDFs found for project `{project}`. Put PDFs in data/{project}/ and refresh."
                )
                continue

            persist_dir = os.path.join("./chroma_db", project)
            built = os.path.exists(persist_dir)
            cols = st.columns([3, 1])
            with cols[1]:
                if st.button(f"Rebuild {project}"):
                    if os.path.exists(persist_dir):
                        shutil.rmtree(persist_dir)
                    built = False
            st.write("Status:", "✅ Built" if built else "⚠️ Not built")

            try:
                retriever, llm = build_vectorstore_and_llm(
                    project, pdf_paths, force_rebuild=not built
                )
            except Exception as e:
                st.error(f"Failed to build/load vector store for {project}: {e}")
                continue

            # simple chat UI for this project
            q = st.text_input(f"Question for {project}")
            if st.button(f"Ask {project}") and q:
                try:
                    docs = retriever.get_relevant_documents(q)
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {q}"
                    response = llm.invoke(prompt)
                    answer = getattr(response, "content", str(response))
                    st.markdown("**Answer**")
                    st.write(answer)
                    with st.expander("Context / source snippets"):
                        for i, d in enumerate(docs, start=1):
                            st.write(f"---\n**Doc {i}**:\n{d.page_content[:1000]}")
                except Exception as e:
                    st.error(f"Error while answering for {project}: {e}")


def main():
    # initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_section()
        return

    # ensure required imports
    ensure_imports()

    # show main UI
    main_ui()


if __name__ == "__main__":
    main()
