import gradio as gr
from src.rag_system import RAGSystem
import re

class Interface:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
    def respond(self, message, history):
        """Process user message and generate response with conversation context"""
        result = self.rag_system.query(message)
        
        # Extract answer and sources
        answer = result["answer"]
        sources = result["sources"]
        
        # Format answer with HTML highlighting
        formatted_answer = self.format_answer_with_sources(answer, sources)
        
        return formatted_answer
    
    def format_answer_with_sources(self, answer, sources):
        """Format answer with HTML source highlighting"""
        # Create a lookup dictionary for sources
        source_lookup = {f"Chapter {s['source']}, Page {s.get('chunk_id', '')}" : s for s in sources}
        source_lookup.update({s['source']: s for s in sources})
        
        # Look for source citations in the answer
        pattern = r'\[Source: ([^\]]+)\]'
        
        # Replace them with HTML highlighting
        def replacement(match):
            source_text = match.group(1).strip()
            if source_text in source_lookup:
                source = source_lookup[source_text]
                return f'<span style="background-color: #f0f8ff; border-bottom: 2px solid #4682b4; padding: 2px; font-size: 0.9em;"> [{source_text}] </span>'
            else:
                return match.group(0)
                
        highlighted_answer = re.sub(pattern, replacement, answer)
        
        # Add source list at the bottom
        if sources:
            highlighted_answer += "\n\n<div style='border-top: 1px solid #ccc; margin-top: 20px; padding-top: 10px;'><b>Source Documents:</b><ol>"
            for source in sources:
                highlighted_answer += f"<li><b>{source['source']}</b> {source.get('chapter_title', '')}<br/><small>{source['preview']}</small></li>"
            highlighted_answer += "</ol></div>"
            
        return highlighted_answer
        
    def create_interface(self):
        """Create Gradio interface with conversation memory"""
        demo = gr.ChatInterface(
            self.respond,
            title="NeuralNexus: NLP Textbook Tutor",
            description="Ask questions about concepts from Jurafsky & Martin's 'Speech & Language Processing' textbook. You can ask follow-up questions and the system will remember the conversation context.",
            theme="soft",
            examples=[
                "What is a Hidden Markov Model?",
                "Explain the differences between CRFs and HMMs.",
                "How do transformers handle long-range dependencies?",
                "What are the components of a neural language model?",
                "Explain the concept of attention in NLP."
            ],
            css="""
            .source-highlight {
                background-color: #f0f8ff;
                border-bottom: 2px solid #4682b4;
                padding: 2px;
                font-size: 0.9em;
            }
            """
        )
        
        return demo