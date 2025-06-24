# Medical RAG QA App

A Retrieval-Augmented Generation (RAG) application for answering medical questions using your own PDF documents. This app leverages open-source models and frameworks to provide accurate, context-aware answers from your medical literature.

---

## Features
- **PDF Ingestion:** Load and process your own medical PDFs.
- **State-of-the-Art Embeddings:** Uses PubMedBERT for domain-specific embeddings.
- **Vector Search:** Qdrant as a high-performance vector database.
- **Local LLM:** Meditron-7B (quantized) for efficient, private inference.
- **Modern Web UI:** FastAPI backend with a Bootstrap-powered frontend.
- **Source Attribution:** See the context and source document for every answer.

---

## Architecture
```
PDFs (data/) → Ingestion (ingest.py) → Embeddings (PubMedBERT) → Qdrant Vector DB
                                                        ↓
                                                FastAPI (rag.py)
                                                        ↓
                                            Meditron-7B LLM (local)
                                                        ↓
                                                Web UI (index.html)
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- [Qdrant](https://qdrant.tech/) running locally (Docker recommended)
- Sufficient RAM/CPU for running Meditron-7B (quantized)

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd Medical-RAG-APP
```

### 3. Install Dependencies
```bash
pip install -r requirments.txt
```

### 4. Download/Place Model & Data
- Place your **quantized Meditron-7B GGUF** model in `meditron-7b-Q4_K_M-GGUF/`.
- Place your **PDFs** in the `data/` directory.

### 5. Start Qdrant (if not already running)
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 6. Ingest Your Data
```bash
python ingest.py
```
This will process your PDFs and populate the Qdrant vector database.

### 7. Run the Web App
```bash
uvicorn rag:app --reload
```
Visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## Usage
- Enter your medical question in the web UI.
- The app retrieves relevant context from your PDFs and generates an answer using the local LLM.
- The response includes the answer, the context used, and the source document.

---

## File/Folder Structure
```
Medical RAG APP/
├── data/                        # Your medical PDFs
├── ingest.py                    # Script to ingest and embed PDFs
├── rag.py                       # FastAPI backend and LLM orchestration
├── retriever.py                 # Standalone script for vector search testing
├── meditron-7b-Q4_K_M-GGUF/     # Quantized Meditron-7B model
├── templates/
│   └── index.html               # Web UI template
├── static/                      # (Optional) Static files for UI
├── requirments.txt              # Python dependencies
```

---

## Customization
- **Change Embedding Model:** Edit `ingest.py` and `rag.py` to use a different HuggingFace model.
- **Swap LLM:** Update the model path and type in `rag.py` to use another GGUF-compatible model.
- **Tune Chunk Size:** Adjust `chunk_size` and `chunk_overlap` in `ingest.py` for different document granularity.

---

## Troubleshooting
- **Qdrant Connection Errors:** Ensure Qdrant is running on `localhost:6333`.
- **Model Loading Issues:** Confirm the GGUF model path and compatibility with CTransformers.
- **No Answers Returned:** Check that your PDFs are in the `data/` folder and ingestion completed successfully.
- **Web UI Not Loading:** Make sure FastAPI is running and accessible at the correct port.

---

## License
This project is for research and educational purposes. Please check the licenses of the models and datasets you use. 