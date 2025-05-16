import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class EmbeddingManager:
    def __init__(self, api_key, vector_db_path="vectordb"):
        self.api_key = api_key
        self.vector_db_path = vector_db_path
        self.chunks_dir = "data/chunks"
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key
        
    def create_vector_database(self):
        """Create vector database from chunks"""
        # Read all chunk files
        all_documents = []
        all_metadatas = []
        
        for filename in os.listdir(self.chunks_dir):
            if filename.endswith("_chunks.json"):
                filepath = os.path.join(self.chunks_dir, filename)
                
                with open(filepath, "r", encoding="utf-8") as file:
                    chunks = json.load(file)
                
                for chunk in chunks:
                    all_documents.append(chunk["content"])
                    all_metadatas.append(chunk["metadata"])
        
        print(f"Creating embeddings for {len(all_documents)} chunks...")
        
        # Create embedding model
        embeddings = OpenAIEmbeddings()
        
        # Create and persist vector database
        vectordb = Chroma.from_texts(
            texts=all_documents,
            metadatas=all_metadatas,
            embedding=embeddings,
            persist_directory=self.vector_db_path
        )
        
        vectordb.persist()
        print("Vector database created and persisted successfully")
        
        return vectordb