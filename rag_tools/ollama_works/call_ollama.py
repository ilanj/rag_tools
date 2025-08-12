import requests

url = "http://localhost:11434/api/chat"

data = {
    # "model": "smallm",
    # "model": "gpt-oss",
    "model": "llama3.1",
    # "model": "deepseek-r1",
    "stream": False,
    "messages": [{"role": "user", "content": "Tell me a joke"}],
}

response = requests.post(url, json=data)

# Now it's valid JSON
print(response.json()["message"]["content"])
