
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"
client = QdrantClient (url=url,prefer_grpc=False)
print(client)
print("#################")
db =Qdrant(client=client,embeddings=embeddings,collection_name="vector_db_Medical")
print(db)
print("#######")
query = "What is Metastatic disease?"
docs =db.similarity_search_with_score(query=query,k=2)
for i in docs :
    doc,score=i
    print({"score":score,"conetnt":doc.page_content,"metadata":doc.metadata})