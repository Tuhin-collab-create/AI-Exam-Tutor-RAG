
📚 AI Exam Tutor - RAG Chatbot

1. Project Overview
The AI Exam Tutor is a Retrieval-Augmented Generation (RAG) system built to provide grounded, accurate answers to natural language questions based on a specific collection of documents. This project was developed as part of an AI Engineering Internship assignment, focusing on the end-to-end implementation of a RAG pipeline—from document ingestion and chunking to vector storage and LLM-based generation.

2. Tech Stack
- Language: Python 3.11+
- Framework: Streamlit (UI)
- Vector Database: FAISS (Facebook AI Similarity Search)
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- LLM: Google Gemini (Generative AI)
- Document Processing: PyPDF (for PDF extraction), NLTK (for sentence tokenization)

3. Knowledge Base & Dataset
The model's knowledge base is built upon 5 meaningful and non-trivial documents. These documents serve as the grounded context for the chatbot:

- Aftermath of World War II: A comprehensive study of global restructuring, the decline of European colonial empires, and the rise of the US and USSR as superpowers.
- Ethics of Artificial Intelligence (UNESCO Standards): The first-ever global standard-setting instrument on AI ethics, focusing on human rights and dignity.
- Attention Is All You Need: The seminal research paper by Google researchers introducing the Transformer architecture, which dispensed with recurrence and convolutions in favor of attention mechanisms.
- AI Ethics Framework (Detailed): A deeper dive into the UNESCO recommendation, highlighting principles like proportionality, safety, and accountability.
- Global Post-War Order: Documentation regarding international cooperation, the UN Charter, and decolonization processes following 1945.

4. Project Folder Structure
My project directory is organized as follows:

my project/
├── data/                 # Source PDF documents (the 5 datasets)
├── app.py                # Streamlit app with Live PDF Uploader functionality
├── app2.py               # Streamlit app using the Pre-Trained FAISS index
├── chatbot_final.ipynb   # Development notebook with original indexing & testing logic
├── faiss_index.bin       # The persisted FAISS vector index (binary file)
├── data.pkl              # Persisted metadata and text chunks (pickle file)
├── requirements.txt      # List of required Python libraries
└── README.txt            # Project documentation

About my dataset:
1. Aftermath of World War II
Focus: Global restructuring and political shifts post-1945.

Key Content: Detailed analysis of the decline of European colonial empires, the emergence of the US and USSR as global superpowers, and the early tensions that led to the Cold War.

2. Ethics of Artificial Intelligence (UNESCO Standards)
Focus: Global governance and human rights in AI.

Key Content: The official framework adopted by UNESCO members. it establishes the first global standard-setting instrument on the ethics of AI, emphasizing human dignity, privacy, and social well-being.

3. "Attention Is All You Need" (Research Paper)
Focus: The technical foundation of modern AI.

Key Content: The seminal 2017 paper by Vaswani et al. that introduced the Transformer architecture. This is the literal blueprint for models like GPT and Gemini, explaining how "Self-Attention" allows machines to process language effectively.

4. AI Ethics Framework (Detailed Analysis)
Focus: Operationalizing AI safety.

Key Content: An in-depth exploration of the UNESCO recommendations, focusing on the practical application of principles like proportionality, individual safety, and environmental sustainability in AI development.

5. Global Post-War Order
Focus: International cooperation and law.

Key Content: Documentation regarding the formation of the United Nations, the drafting of the UN Charter, and the massive decolonization processes that reshaped the map of the world in the mid-20th century.

5. Application Versions

App 1: Live Uploader (app.py)
- Purpose: Dynamic document processing.
- Workflow: Users upload PDF documents directly. The app extracts text, generates embeddings, and builds a FAISS index in real-time.
- Key Feature: Tracks source filenames dynamically to provide precise citations for any new document provided.

App 2: Pre-Trained Bot (app2.py)
- Purpose: High-performance retrieval from the core 5-document dataset.
- Workflow: Loads the pre-existing faiss_index.bin and data.pkl generated during the development phase.
- Key Feature: Optimized for speed; allows for immediate querying without the need for manual file uploads.

6. Architecture & Strategy
- Chunking Strategy: I implemented a Sentence-Based Chunking Strategy. Instead of fixed character limits, sentence tokenization ensures the bot retrieves complete thoughts and maintains semantic integrity.
- Grounding: The system is programmed to answer using ONLY the provided context. If an answer is not found in the documents, the bot will state it clearly to prevent hallucinations.

7. Setup & Installation
1. Clone the Repository:
   git clone <your-repository-url>
   cd <repository-folder>
2. Install Dependencies:
   pip install -r requirements.txt
3. Run the Application:
   - To use the live uploader: streamlit run app.py
   - To query the pre-trained knowledge base: streamlit run app2.py

8. Example Queries
1. "What are the core values of the UNESCO AI Ethics framework?"
2. "Explain the impact of World War II on global economics."
3. "What are the key safety requirements for AI systems?"
4. "Summarize the role of the Nazi Party in Germany's rise to power."
5. The "Unanswerable" Test: "What is the recipe for chocolate cake?" (Expected: The bot will state it cannot find the answer in the provided context).
