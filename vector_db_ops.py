from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

import os
import json 
from uuid import uuid4

# Environment Setup
os.environ["USER_AGENT"] = "user_agent"
open_ai_key = os.environ.get("OPENAI_API_KEY")
embd = OpenAIEmbeddings(api_key=open_ai_key)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
db_file = "db.json"

# Check if index exists else create index 
index_name = "adaptive-rag-index"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,  
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     print(f"New index created: {index_name}")
index = pc.Index(index_name)
print(f"Using index: {index}")

def text_loader(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    document = Document(
        page_content=content,
        metadata={
            "source": file_path,
            "type": "text",
            "size": os.path.getsize(file_path),
            "encoding": "utf-8",
            "last_modified": os.path.getmtime(file_path),
        }
    )

    return [document]

def get_retriever():
    vectorstore = PineconeVectorStore(
        embedding=embd,  
        index_name=index_name,
    )
    return vectorstore, vectorstore.as_retriever()

class VectorDB:
    """Handles document embeddings in Pinecone vector database"""

    def __init__(self, index_name=index_name, db_file=db_file):
        """Initialize vector store connection"""
        self.index_name = index_name
        self.db_file = db_file

    def load_db(self):
        """Load stored document IDs from JSON"""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                return json.load(f)
        return {}

    def save_db(self, data):
        """Save updated document IDs to JSON"""
        with open(self.db_file, "w") as f:
            json.dump(data, f, indent=4)

    def add_document_db(self, doc_name, chunk_ids):
        """Add a file and its chunk IDs to the database"""
        db = self.load_db()
        db[doc_name] = {"ids": chunk_ids}
        self.save_db(db)

    def store_vectordb(self, docs, group=None):
        """Load documents, split them, store in Pinecone, and update JSON database"""
        vectorstore = PineconeVectorStore(embedding=embd, index_name=self.index_name)

        for doc in docs:
            try:
                loaded = None 
                if doc.startswith(('http://', 'https://')):
                    loader = WebBaseLoader(doc)
                    loaded = loader.load()
                
                elif doc.lower().endswith('.pdf'):
                    loader = PyPDFLoader(doc)
                    loaded = loader.load()
                
                elif doc.lower().endswith('.txt'):
                    loaded = text_loader(doc)
                
                else:
                    print(f"Unsupported file type: {doc}")
                    continue
            
            except Exception as e:
                print(f"Error loading {doc}: {e}")
                continue
        
            if loaded:
                for d in loaded:
                    d.metadata['source_name'] = doc

                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
                doc_splits = text_splitter.split_documents(loaded) 
                uuids = [f"{str(uuid4())}" for _ in range(len(doc_splits))]

                vectorstore.add_documents(
                    documents=doc_splits, 
                    ids=uuids,
                    # namespace=group
                )
                self.add_document_db(doc, uuids)
                print(f"Stored {len(doc_splits)} chunks from {doc}")

    def delete_document_vectordb(self, doc_name):
        """Remove a file entry from the JSON and Pinecone databases"""
        db = self.load_db()
        if doc_name in db:
            chunk_ids = db[doc_name]["ids"]
            del db[doc_name]
            self.save_db(db)

            vectorstore = PineconeVectorStore(embedding=embd, index_name=self.index_name)
            vectorstore.delete(ids=chunk_ids) 

            print(f"Deleted {doc_name} and {len(chunk_ids)} chunks from Pinecone.")
            return True

        print(f"File {doc_name} not found in database.")
        return False  
