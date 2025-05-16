# Add this code to a new file called test_retrieval.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

def test_vector_retrieval():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create embeddings model
    embeddings = OpenAIEmbeddings()
    
    # Load vector database
    vector_db_path = "vectordb"
    if not os.path.exists(vector_db_path):
        print(f"Error: Vector database at {vector_db_path} not found")
        return
    
    try:
        vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        
        # Test queries
        test_queries = [
            "What is a Hidden Markov Model?",
            "Explain how transformers work in NLP",
            "What are the components of a neural language model?",
            "How does attention mechanism work?",
            "What is the difference between CRF and HMM?"
        ]
        
        for query in test_queries:
            print(f"\n\nTesting query: {query}")
            # Retrieve documents
            docs = vectordb.similarity_search(query, k=3)
            
            print(f"Found {len(docs)} relevant documents")
            
            # Print document content
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content preview: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
                print(f"Token count: {doc.metadata.get('token_count', 'Unknown')}")
    
    except Exception as e:
        print(f"Error testing vector retrieval: {e}")

if __name__ == "__main__":
    test_vector_retrieval()