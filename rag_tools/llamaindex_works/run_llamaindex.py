from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from chromadb import PersistentClient
from loguru import logger
import os

# Paths
CHROMA_DB_DIR = "./chroma_db"

# 1. Define your Ollama LLM
llm = Ollama(model="llama3.1", request_timeout=1200.0)

# 2. Define Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create Chroma client (persistent)
client = PersistentClient(path=CHROMA_DB_DIR)
chroma_collection = client.get_or_create_collection("llamaindex_vstore")

# 4. Wrap Chroma in LlamaIndex VectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 5. Check if embeddings exist (skip embedding if already present)
if not os.listdir(CHROMA_DB_DIR):  
    logger.info("No existing embeddings found. Creating new embeddings...")
    documents = SimpleDirectoryReader("data").load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        storage_context=storage_context
    )
else:
    logger.info("Loading existing embeddings from ChromaDB...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)

# 6. Create chat engine
chat_engine = index.as_chat_engine(llm=llm, chat_mode="condense_question", verbose=True)

# 7. Interactive loop
logger.info("Type your query and press Enter. Type 'exit' to quit.")

while True:
    query = input("\nYour question: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        logger.info("Exiting chat...")
        break

    response = chat_engine.chat(query)
    logger.info(f"Answer: {response}")
