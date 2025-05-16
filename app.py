from src.utils.env_loader import load_environment
from src.data_processor import DataProcessor
from src.embeddings import EmbeddingManager
from src.rag_system import RAGSystem
from src.interface import Interface

def process_data():
    """Process textbook data and create vector database"""
    # Load API key
    api_key = load_environment()
    
    # Process data
    processor = DataProcessor()
    print("Downloading chapters...")
    downloaded_chapters = processor.download_all_chapters()
    
    print("Processing chapters...")
    processed_files = processor.process_all_chapters(downloaded_chapters)
    
    print("Chunking text...")
    total_chunks = processor.chunk_all_processed_files(processed_files)
    
    # Create vector database
    print("Creating vector database...")
    embedding_manager = EmbeddingManager(api_key)
    embedding_manager.create_vector_database()
    
    print("Data processing complete!")

def run_interface():
    """Run the RAG interface"""
    # Load API key
    api_key = load_environment()
    
    # Create RAG system
    rag_system = RAGSystem(api_key)
    
    # Create interface
    interface = Interface(rag_system)
    demo = interface.create_interface()
    
    # Launch interface
    demo.launch(share=True)

if __name__ == "__main__":
    # If you need to process data, uncomment this line
    # process_data()
    
    # Run interface
    run_interface()