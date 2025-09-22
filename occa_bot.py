import streamlit as st
import faiss
import json 
import numpy as np
import pandas as pd
from src.utils.rag_utils import get_ollama_embedding, ollama_chat_stream

# Load persisted assets
#df = pd.read_csv("anzsco_flat.csv")
embeddings = np.load("./db/anzsco_embeddings.npy")
index = faiss.read_index("./db/anzsco.index")
with open("./db/anzsco_meta.json", "r", encoding="utf-8") as f:
    id_to_meta = json.load(f)

# Retrieval function
def retrieve(query, k=5):
    q_emb = np.array([get_ollama_embedding(query)]).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_meta[i] for i in I[0]]

# Streamlit chatbot
st.title("💬 OCCA chatBot")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Ask me about an occupation...")

if user_query:
    results = retrieve(user_query, k=3)
    context = "\n\n".join([r["occupation_name"] + "\n" + (r["skill_level"] or "") + "\n" + (r["tasks"] or "") for r in results])

    prompt = f"""
    Use the following context to answer the question:

    {context}

    Question: {user_query}
    Answer:
    """

    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for token in ollama_chat_stream(prompt, model="llama3"):
            full_response += token
            response_container.markdown(full_response)
