import os
import re
import requests
from bs4 import BeautifulSoup
import json
from PyPDF2 import PdfReader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    def __init__(self, base_url="https://web.stanford.edu/~jurafsky/slp3/"):
        self.base_url = base_url
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        self.chunks_dir = "data/chunks"
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        
    def get_chapter_links(self):
        """Fetch all chapter links from the textbook website"""
        response = requests.get(self.base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        chapter_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and (href.endswith('.html') or href.endswith('.pdf')) and not href.startswith('http'):
                chapter_links.append(href)
        
        return chapter_links
        
    def download_chapter(self, chapter_link):
        """Download a chapter from the textbook"""
        url = self.base_url + chapter_link
        print(f"Downloading: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            chapter_filename = os.path.basename(chapter_link)
            filepath = os.path.join(self.raw_dir, chapter_filename)
            
            with open(filepath, "wb") as file:
                file.write(response.content)
            
            return chapter_filename
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
            
    def download_all_chapters(self):
        """Download all chapters from the textbook"""
        chapter_links = self.get_chapter_links()
        downloaded_chapters = []
        
        for link in chapter_links:
            chapter_file = self.download_chapter(link)
            if chapter_file:
                downloaded_chapters.append(chapter_file)
        
        return downloaded_chapters
        
    def process_html_chapter(self, filename):
        """Process HTML chapter to extract text"""
        filepath = os.path.join(self.raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        main_content = soup.find('body')
        
        if not main_content:
            return None
            
        text = ""
        # Process headings and paragraphs
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'pre', 'table']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                text += "\n" + "#" * level + " " + element.get_text().strip() + "\n\n"
            elif element.name == 'p':
                text += element.get_text().strip() + "\n\n"
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    text += "• " + li.get_text().strip() + "\n"
                text += "\n"
            elif element.name == 'pre':
                text += "```\n" + element.get_text() + "\n```\n\n"
            elif element.name == 'table':
                text += "[TABLE]\n"
                for row in element.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        text += " | ".join([cell.get_text().strip() for cell in cells]) + "\n"
                text += "\n"
        
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Save processed text
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(self.processed_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)
            
        return output_filename
        
    def process_pdf_chapter(self, filename):
        """Process PDF chapter to extract text"""
        filepath = os.path.join(self.raw_dir, filename)
        try:
            reader = PdfReader(filepath)
            text = ""
            
            # Get chapter title from filename
            chapter_title = os.path.splitext(filename)[0]
            text += f"# {chapter_title}\n\n"
            
            # Extract text from each page
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"## Page {i+1}\n\n"
                    text += page_text + "\n\n"
            
            # Clean up the text
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Save processed text
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(self.processed_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(text)
                
            return output_filename
        except Exception as e:
            print(f"Error processing PDF {filename}: {e}")
            return None
            
    def process_all_chapters(self, downloaded_chapters):
        """Process all downloaded chapters"""
        processed_files = []
        
        for chapter in downloaded_chapters:
            if chapter.endswith('.html'):
                processed_file = self.process_html_chapter(chapter)
            elif chapter.endswith('.pdf'):
                processed_file = self.process_pdf_chapter(chapter)
            else:
                print(f"Skipping unknown file format: {chapter}")
                continue
                
            if processed_file:
                processed_files.append(processed_file)
                
        return processed_files
        
    def count_tokens(self, text):
        """Count tokens in text using tiktoken"""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
        
    def chunk_text(self, filename, chunk_size=800, chunk_overlap=300):
        """Split text into chunks with intelligent boundaries"""
        filepath = os.path.join(self.processed_dir, filename)
        
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Create text splitter with improved parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Smaller chunks for more precise retrieval
            chunk_overlap=chunk_overlap,  # Larger overlap to maintain context
            length_function=self.count_tokens,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Get filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        # Store chunks with enhanced metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            # Extract chapter title if available
            chapter_title = ""
            if chunk and "\n" in chunk and chunk.split("\n")[0].startswith("# "):
                chapter_title = chunk.split("\n")[0].replace("# ", "")
            
            chunk_info = {
                "content": chunk,
                "metadata": {
                    "source": base_filename,
                    "chunk_id": i,
                    "token_count": self.count_tokens(chunk),
                    "chapter_title": chapter_title
                }
            }
            chunk_data.append(chunk_info)
        
        # Save chunks to file
        output_path = os.path.join(self.chunks_dir, f"{base_filename}_chunks.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(chunk_data, file, indent=2)
        
        return len(chunks)
        
    def chunk_all_processed_files(self, processed_files):
        """Chunk all processed files"""
        total_chunks = 0
        
        for file in processed_files:
            num_chunks = self.chunk_text(file)
            total_chunks += num_chunks
            print(f"Created {num_chunks} chunks from {file}")
            
        return total_chunks