from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

import os
import json

app = FastAPI()

# FastAPI setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === LLM Setup ===
local_llm_path = r"C:\Users\youssef\Desktop\Medical RAG APP\meditron-7b-Q4_K_M-GGUF\meditron-7b.Q4_K_M.gguf"

config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm_path,
    model_type="llama",
    lib="avx2",
    **config
)
print("✅ LLM Initialized...")

# === Prompt Template ===
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# === Embeddings & Qdrant ===
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db_Medical")

retriever = db.as_retriever(search_kwargs={'k': 1})

# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )

    response = qa(query)
    print(response)

    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata.get('source', 'N/A')

    response_data = jsonable_encoder({"answer": answer, "source_document": source_document, "doc": doc})
    return Response(content=json.dumps(response_data), media_type="application/json")
