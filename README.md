
# RAG

## Install Ollama & Pull Llama3 on Mac which is used a LLM to test


1. Install Ollama ->
curl -fsSL https://ollama.com/install.sh | sh

2. Start Ollama service ->
ollama serve &

3. Pull the Llama 3 model ->
ollama pull llama3.1

4. Test the  ->
ollama run llama3

#  ubuntu
* snap install ollama

* ollama--version

    ollama pull llama3.1

## Frameworks used with LLM

1. Langchain

2. LangGraph

3. LlamaIndex

## Vector Store
1. Im-Memory ->
        FAISS

2. Disk Store ->
         chromadb 

* No cloud db used. Chroma embeddings stored can be uploaded to any CSP and resued still. 
