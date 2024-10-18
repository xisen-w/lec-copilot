import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Agents.basicAgents import LLMAgent
from fundations.LLMResponsePro import LLMResponsePro
from fundations.open_ai_RAG import Retriever
from pydantic import BaseModel

class IdeaCardSchema(BaseModel):
    idea_name: str
    idea_explanation: str

class LectureAgent(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.retriever = Retriever()
        self.llm_pro = LLMResponsePro(model_name)

    def record_lecture(self, audio_file_path):
        """
        Use the whisper function to transcribe a lecture from an audio file.
        """
        transcription = self.llm_pro.whisper(audio_file_path)
        return transcription

    def explain(self, lecture_content: str):
        """
        Check for terms in the lecture content that need explanation and provide explanations.
        Outputs a schema of list of idea cards.
        """
        # Split lecture content into chunks (e.g., sentences or paragraphs)
        chunks = lecture_content.split('. ')
        idea_cards = []

        for chunk in chunks:
            # Use LLM to identify and explain terms
            response = self.perform_action(
                system_prompt="Identify and explain key terms in the following text.",
                user_prompt=chunk,
                schema_class=IdeaCardSchema
            )
            if response:
                # Assuming response is a list of IdeaCardSchema objects
                idea_cards.extend(response)

        return idea_cards

    def build_knowledge(self, pdf_paths: list[str]):
        """
        Build a knowledge base by uploading PDFs or text, storing them in chunks and vector store.
        """
        for pdf_path in pdf_paths:
            self.retriever.create_embedding_for_pdf(pdf_path)

    def retrieve_knowledge(self, query: str, top_n: int = 5):
        """
        Retrieve relevant knowledge for the given query.
        """
        return self.retriever.retrieve_and_ask(query, top_n=top_n)

    def summarise_lec(self, lecture_content: str):
        """
        Summarize the whole lecture into a hierarchical level of knowledge graph.
        """
        response = self.perform_action(
            system_prompt="Summarize the following lecture content into a hierarchical knowledge graph.",
            user_prompt=lecture_content
        )
        return response.knowledge_graph if response else None

# Example usage
if __name__ == "__main__":
    agent = LectureAgent(model_name="gpt-4o-mini")
    
    # Record lecture
    transcription = agent.record_lecture("/Users/wangxiang/Desktop/my_workspace/lec-copilot/test/demo_recording/test-lecture.m4a")
    print("Transcription:", transcription)
    
    # Explain terms
    idea_cards = agent.explain(transcription)
    print("Idea Cards:", idea_cards)
    
    # # Build knowledge base
    # agent.build_knowledge(["/path/to/pdf1.pdf", "/path/to/pdf2.pdf"])
    
    # # Retrieve knowledge
    # knowledge, context = agent.retrieve_knowledge("Explain the concept of AI ethics.")
    # print("Retrieved Knowledge:", knowledge)
    
    # # Summarize lecture
    # summary = agent.summarise_lec(transcription)
    # print("Lecture Summary:", summary)
