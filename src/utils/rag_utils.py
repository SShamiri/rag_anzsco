import pandas as pd
import numpy as np
import requests
import json
from typing import List

import ollama

# Function to get embeddings using Ollama
def get_ollama_embedding(text, model_name="llama3"):
    try:
        # Call the Ollama embeddings API
        response = ollama.embeddings(model=model_name, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding for text: '{text}' - {e}")
        return None
 
# Function to retrieve top-k similar documents   
def retrieve(query: str, k: int = 5):
    q_emb = np.array([get_ollama_embedding(query)]).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_meta[i] for i in I[0]]

# Function to stream chat responses from Ollama
def ollama_chat_stream(prompt: str, model: str = "llama3"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert on ANZSCO occupations."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
