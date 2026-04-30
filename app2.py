import streamlit as st
import os
import re
import uuid
import pickle
import numpy as np
import faiss
import google.generativeai as genai
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. PAGE CONFIG (Must be first)
st.set_page_config(page_title="AI Exam Tutor", layout="wide")

# 2. NLTK DATA DOWNLOAD
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')

download_nltk()

# 3. RESOURCE LOADING (Matches your data.pkl structure)
@st.cache_resource
def load_assets():
    # Model used in your notebook
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load your saved training files
    if os.path.exists("faiss_index.bin") and os.path.exists("data.pkl"):
        index = faiss.read_index("faiss_index.bin")
        with open("data.pkl", "rb") as f:
            stored_data = pickle.load(f)
        return embed_model, index, stored_data
    return embed_model, None, None

embed_model, index, stored_data = load_assets()

# 4. CLEANING & CHUNKING (Copied from your .ipynb)
def clean_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d{1,3}\b(?=\s)', '', text) # Remove page numbers
    return text.strip()

def semantic_chunking(text, threshold=0.55, max_words=200):
    sentences = sent_tokenize(text)
    if not sentences: return []
    
    sentence_emb = embed_model.encode(sentences)
    chunks = []
    current_chunk = []
    current_emb = []
    
    for i, sent in enumerate(sentences):
        emb = sentence_emb[i]
        if not current_chunk:
            current_chunk.append(sent)
            current_emb.append(emb)
            continue
        
        sim = cosine_similarity([current_emb[-1]], [emb])[0][0]
        if sim < threshold or len(" ".join(current_chunk).split()) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_emb = [emb]
        else:
            current_chunk.append(sent)
            current_emb.append(emb)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# 5. UI & CHAT
st.title("📚 AI Exam Tutor - Trained Mode")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

if not index or not stored_data:
    st.error("Missing 'faiss_index.bin' or 'data.pkl'. Please place them in the project folder.")
    st.stop()

if api_key:
    genai.configure(api_key=api_key)
    # Using the 1.5-flash version you verified in Colab
    llm = genai.GenerativeModel("models/gemini-2.5-flash")
    
    query = st.chat_input("Ask about your training documents...")
    
    if query:
        # Step A: Vectorize Query
        q_vec = embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(q_vec)
        
        # Step B: Retrieve relevant chunks
        _, indices = index.search(q_vec, k=3)
        
        # Step C: Construct Context
        context_list = []
        for i in indices[0]:
            # Validation check: ensures the index exists in your data.pkl
            if i != -1 and i < len(stored_data):
                context_list.append(stored_data[i])
        
        # Ensure 'context' is never empty to avoid errors in the prompt
        if context_list:
            context = "\n---\n".join(context_list)
        else:
            context = "No relevant information found in the training documents."
        # Step D: The Prompt (Matches your notebook persona)
        prompt = f"""
        You are an expert exam tutor. 
        Answer using ONLY the provided context. 

        RULES:
        1. Use headings, bullet points, and structure for clarity.
        2. If the answer isn't in the context, say you don't know.
        3. Do NOT mention your internal logic or outside knowledge.
        4.You MAY add headings, explanations, and structure to improve clarity.


        CONTEXT:
        {context}

        QUESTION:
        {query}
        """
        
        st.chat_message("user").write(query)
        with st.chat_message("assistant"):
            try:
                response = llm.generate_content(prompt)
                st.markdown(response.text)
                with st.expander("View Source Context"):
                    st.write(context)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Enter your Gemini API Key in the sidebar to begin.")