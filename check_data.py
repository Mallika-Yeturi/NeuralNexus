# Add this code to a new file called check_data.py
import os

def check_data_processing():
    # Check raw data
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} directory doesn't exist")
        return False
    
    raw_files = os.listdir(raw_dir)
    print(f"Found {len(raw_files)} raw files")
    
    # Check processed data
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} directory doesn't exist")
        return False
    
    processed_files = os.listdir(processed_dir)
    print(f"Found {len(processed_files)} processed files")
    
    # Check chunks
    chunks_dir = "data/chunks"
    if not os.path.exists(chunks_dir):
        print(f"Error: {chunks_dir} directory doesn't exist")
        return False
    
    chunk_files = os.listdir(chunks_dir)
    print(f"Found {len(chunk_files)} chunk files")
    
    # Check vector database
    vector_db_path = "vectordb"
    if not os.path.exists(vector_db_path):
        print(f"Error: {vector_db_path} directory doesn't exist")
        return False
    
    vector_files = os.listdir(vector_db_path)
    print(f"Found {len(vector_files)} files in vector database")
    
    # Check sample content of a chunk file
    if chunk_files:
        import json
        sample_chunk_file = os.path.join(chunks_dir, chunk_files[0])
        print(f"Examining sample chunk file: {sample_chunk_file}")
        
        try:
            with open(sample_chunk_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                print(f"Found {len(chunks)} chunks in the sample file")
                
                # Print first chunk preview
                if chunks:
                    print("\nSample content preview:")
                    content = chunks[0]['content']
                    print(content[:300] + "..." if len(content) > 300 else content)
                else:
                    print("No chunks found in the sample file")
        except Exception as e:
            print(f"Error reading chunk file: {e}")
    
    return True

if __name__ == "__main__":
    check_data_processing()