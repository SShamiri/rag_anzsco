import streamlit as st
import requests
import json
import faiss
import numpy as np
import pandas as pd


# -------------------------
# Ollama helpers
# -------------------------
def get_ollama_embedding(text: str, model: str = "nomic-embed-text") -> list:
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload)
    r.raise_for_status()
    return r.json()["embedding"]

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

# -------------------------
# Build FAISS index (demo)
# -------------------------
@st.cache_resource
def build_index():
    # Load your flattened ANZSCO DataFrame here
    df = pd.read_csv("./data/anzsco_full.csv")  # from earlier step
    docs = df["occupation_name"].fillna("") + "\n" + df["skill_level"].fillna("") + "\n" + df["tasks"].fillna("")
    embeddings = [get_ollama_embedding(doc) for doc in docs.tolist()]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    id_to_meta = df.to_dict(orient="records")
    return index, id_to_meta, docs.tolist()

index, id_to_meta, docs = build_index()

def retrieve(query, k=5):
    q_emb = np.array([get_ollama_embedding(query)]).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_meta[i] for i in I[0]]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="ANZSCO RAG Chatbot", page_icon="🤖")
st.title("💬 ANZSCO RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Ask me about an occupation...")

if user_query:
    # Retrieve context
    results = retrieve(user_query, k=3)
    context = "\n\n".join([r["occupation_name"] + "\n" + (r["skill_level"] or "") + "\n" + (r["tasks"] or "") for r in results])

    prompt = f"""
    Use the following context to answer the question:

    {context}

    Question: {user_query}
    Answer:
    """

    # Display user message
    st.chat_message("user").write(user_query)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for token in ollama_chat_stream(prompt, model="llama3"):
            full_response += token
            response_container.markdown(full_response)
        st.session_state.history.append({"user": user_query, "assistant": full_response})
