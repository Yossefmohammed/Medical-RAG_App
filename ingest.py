import os
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings  # optional alternative
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Load PDF files from the 'data' directory
loader = DirectoryLoader(
    path="data/",
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Store chunks in Qdrant vector DB
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db_Medical"
)

print("✅ Vector DB 'vector_db_Medical' created successfully!")
