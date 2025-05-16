import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
import json

class RAGSystem:
    def __init__(self, api_key, vector_db_path="vectordb", model_name="gpt-4o"):
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.vector_db_path = vector_db_path
        self.model_name = model_name
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create prompt template
        self.template = """You are an expert NLP tutor helping students understand concepts from Jurafsky & Martin's "Speech & Language Processing" textbook.

Answer the question based on the following context from the textbook and the chat history:

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Important instructions:
1. If the context doesn't contain enough information to answer the question fully, say so clearly.
2. Include specific references to sections, chapters, or page numbers mentioned in the context.
3. Format any technical terms, algorithms, or equations appropriately.
4. Keep your answer comprehensive but concise, focusing on the most relevant information.
5. If the question follows up on a previous question, use the chat history to provide a coherent response.
6. Structure your response with sections or bullet points where appropriate.
7. At the end of your answer, include a "Sources:" section that lists the specific chapters and sections used.
8. Important: For each paragraph or bullet point in your answer, add the source identifier at the end in square brackets, like [Source: Chapter 8, Page 5].

Answer:
"""
        
        self.PROMPT = PromptTemplate(
            template=self.template,
            input_variables=["context", "chat_history", "question"]
        )
        
    def initialize_chain(self):
        """Initialize the QA chain with conversation memory"""
        # Create LLM
        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.2
        )
        
        # Load vector database
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=embeddings
        )
        
        # Create retrieval chain with memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.PROMPT},
            return_source_documents=True,
            output_key="answer"
        )
        
        return self.qa_chain
        
    def query(self, question):
        """Query the RAG system"""
        # Initialize chain if not already initialized
        if self.qa_chain is None:
            self.initialize_chain()
            
        # Get answer and source documents
        response = self.qa_chain({"question": question})
        answer = response["answer"]
        source_docs = response["source_documents"]
        
        # Create source map for visualization
        sources = []
        for i, doc in enumerate(source_docs):
            source_info = {
                "id": i,
                "source": doc.metadata.get("source", "Unknown"),
                "chapter_title": doc.metadata.get("chapter_title", ""),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            }
            sources.append(source_info)
        
        # Return both the answer and sources
        return {
            "answer": answer,
            "sources": sources
        }