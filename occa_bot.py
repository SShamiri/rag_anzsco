import streamlit as st
import faiss
import json 
import numpy as np
import pandas as pd
from src.utils.rag_utils import get_ollama_embedding, ollama_chat_stream

from ollama import Client

# 3. Setup Ollama client
ollama_client = Client()
model_name = "llama3"  # or any model you've pulled locally via Ollama

# Load persisted assets
#df = pd.read_csv("anzsco_flat.csv")
embeddings = np.load("./db/anzsco_embeddings.npy")
index = faiss.read_index("./db/anzsco.index")
with open("./db/anzsco_meta.json", "r", encoding="utf-8") as f:
    id_to_meta = json.load(f)

# Retrieval function
def retrieve(query, k=15):
    q_emb = np.array([get_ollama_embedding(query)]).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_meta[i] for i in I[0]]

# Streamlit chatbot
st.title("💬 OCCA chatBot")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Ask me about an occupation...")

if user_query:
    results = retrieve(user_query)
    context = "\n\n".join([r['doc_text'] for r in results])
    # results = retrieve(user_query, k=3)
    # context = "\n\n".join([r["Occupation Name"] + "\n" + r["Occupation Code"] + "\n" + (r["Skill Level"] + "\n" + r["Path"] or "") + "\n" + (r["tasks"] or "") for r in results])

    # prompt = f"""
    # Use the following context to answer the question:

    # {context}

    # Question: {user_query}
    # Answer:
    # """
    
    # Prompt Ollama with context
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:{context}
    Question: {user_query}
    Answer:
    """
    messages_with_context = [{"role": "system", "content": prompt}]
    
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for response in ollama_client.chat(model=model_name, messages=messages_with_context, stream=True):
            full_response += response['message']['content']
            response_container.markdown(full_response)

    # with st.chat_message("assistant"):
    #     response_container = st.empty()
    #     full_response = ""
    #     for token in ollama_chat_stream(messages_with_context, model="llama3"):
    #         full_response += token
    #         response_container.markdown(full_response)
