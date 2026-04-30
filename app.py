import streamlit as st
import os
import re
import uuid
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import nltk

# STEP 1: MUST BE THE ABSOLUTE FIRST ST COMMAND
st.set_page_config(page_title="AI Exam Tutor", layout="wide")

# STEP 2: UI HEADER
st.title("📚 AI Exam Tutor - RAG Chatbot")
st.sidebar.header("Settings")

# STEP 3: BACKGROUND DOWNLOADS
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')

download_nltk_data()

# --- SETTINGS & INPUTS ---
api_key = st.sidebar.text_input("Enter Google API Key", type="password")
uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()

# --- HELPER FUNCTIONS ---
def clean_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d{1,3}\b(?=\s)', '', text)
    return text.strip()

def semantic_chunking(text, threshold=0.55, max_words=200):
    sentences = sent_tokenize(text)
    if not sentences: return [], []
    sentence_emb = embed_model.encode(sentences)
    chunks, metadata = [], []
    current_chunk, current_emb = [], []
    chunk_id = 0
    for i, sent in enumerate(sentences):
        emb = sentence_emb[i]
        if not current_chunk:
            current_chunk.append(sent)
            current_emb.append(emb)
            continue
        sim = cosine_similarity([current_emb[-1]], [emb])[0][0]
        if sim < threshold or len(" ".join(current_chunk).split()) > max_words:
            text_chunk = " ".join(current_chunk)
            chunks.append(text_chunk)
            metadata.append({"id": str(uuid.uuid4()), "chunk_index": chunk_id, "source": "Unknown"})
            chunk_id += 1
            current_chunk = [sent]
            current_emb = [emb]
        else:
            current_chunk.append(sent)
            current_emb.append(emb)
    return chunks, metadata

# --- PROCESSING & CHAT ---
if uploaded_files and api_key:
    if 'index' not in st.session_state:
        with st.spinner("Processing Documents..."):
            all_chunks, chunk_metadata = [], []
            for uploaded_file in uploaded_files:
                reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() or "" for page in reader.pages])
                chunks, meta = semantic_chunking(clean_text(text))
                for m in meta: m["source"] = uploaded_file.name
                all_chunks.extend(chunks)
                chunk_metadata.extend(meta)
            
            embeddings = embed_model.encode(all_chunks).astype("float32")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            st.session_state.index = index
            st.session_state.all_chunks = all_chunks
            st.session_state.chunk_metadata = chunk_metadata

    if 'index' in st.session_state:
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel("gemini-2.5-flash")
        
        query = st.chat_input("Ask a question about your study materials...")
        
        if query:
            # Retrieval
            q_vec = embed_model.encode([query]).astype("float32")
            faiss.normalize_L2(q_vec)
            _, indices = st.session_state.index.search(q_vec, k=3)
            
            context = ""
            for i in indices[0]:
                if i != -1:
                    context += f"\n[Source: {st.session_state.chunk_metadata[i]['source']}]\n{st.session_state.all_chunks[i]}\n"

            # PROMPT WITH YOUR NEW RULES
            prompt = f"""
            You are an expert exam tutor. 
            Answer using ONLY the provided context.

            STRICT RULES:
            - Do NOT use outside knowledge.
            - You MAY add headings, explanations, and structure to improve clarity.
            - If the context doesn't contain the answer, state that clearly.

            OUTPUT FORMAT:
            1. Definition
            2. Explanation (well-structured)
            3. Key Points (bullets)
            4. Sources

            Context:
            {context}

            Question:
            {query}
            """
            
            st.chat_message("user").write(query)
            
            with st.chat_message("assistant"):
                response = llm.generate_content(prompt)
                st.markdown(response.text)
                with st.expander("Show Sources"):
                    st.text(context)

else:
    st.info("Please provide your API Key and upload PDFs to begin.")