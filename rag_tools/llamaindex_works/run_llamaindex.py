from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from chromadb import PersistentClient
from loguru import logger
import os

CHROMA_DB_DIR = "./chroma_db"  # Chroma embeddings store
PERSIST_DIR = "./storage"  # LlamaIndex metadata store

llm = Ollama(model="llama3.1", request_timeout=1200.0)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = PersistentClient(path=CHROMA_DB_DIR)
chroma_collection = client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Case 1 — No stored index → Build and persist
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    logger.info("No existing index found. Creating new index...")
    documents = SimpleDirectoryReader("data").load_data()

    # No persist_dir here — just create with vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, llm=llm, embed_model=embed_model, storage_context=storage_context
    )

    # Now persist the index metadata
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# Case 2 — Load existing index
else:
    logger.info("Loading existing index from storage...")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=PERSIST_DIR
    )
    index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)

# Chat engine
chat_engine = index.as_chat_engine(llm=llm, chat_mode="condense_question", verbose=True)

while True:
    query = input("\nYour question: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        logger.info("Exiting chat...")
        break

    response = chat_engine.chat(query)
    logger.info(f"Answer: {response}")
