from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from loguru import logger

# 1. Define your Ollama LLM
llm = Ollama(model="llama3.1", request_timeout=1200.0)

# 2. Define Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Load documents
documents = SimpleDirectoryReader("data").load_data()

# 4. Create index (do this once)
index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

# 5. Create a chat engine for conversational memory
chat_engine = index.as_chat_engine(llm=llm, chat_mode="condense_question", verbose=True)

# 6. Interactive loop
logger.info("Type your query and press Enter. Type 'exit' to quit.")

while True:
    query = input("\nYour question: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        logger.info("Exiting chat...")
        break

    response = chat_engine.chat(query)
    logger.info(f"Answer: {response}")
