from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

# 1. Load your PDF
loader = PyPDFLoader("my_utils/langchain_works/leave_policy_test.pdf")
docs = loader.load_and_split()
logger.info("docs loaded and embedding in progress...")

# 2. Use Ollama's built-in embedding model
embedding = OllamaEmbeddings(model="llama3.1")  # or try "nomic-embed-text" if supported

# 3. Vectorstore setup
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# 4. Load the LLM from Ollama
llm = ChatOllama(model="llama3.1")

# 5. Setup RAG pipeline
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Ask question
response = qa.run("What are our leave policies?")
logger.info(response)
