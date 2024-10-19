import sys
import os
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Agents.basicAgents import LLMAgent
from fundations.LLMResponsePro import LLMResponsePro
from fundations.open_ai_RAG import Retriever
from pydantic import BaseModel, Field

class IdeaCardSchema(BaseModel):
    idea_name: str = Field(..., description="The name of the idea.")
    idea_explanation: str = Field(..., description="The explanation of the idea.")
    idea_context: str = Field(..., description="Context of the idea & introducing the idea in greater depth, with at least 3 bullet points.")

class IdeaCardsSchema(BaseModel):
    idea_cards: list[IdeaCardSchema] = Field(..., description="List of idea cards.")

class LectureAgent(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.retriever = Retriever()
        self.llm_pro = LLMResponsePro(model_name)
        self.student_info = None

    def record_lecture(self, audio_file_path):
        """
        Use the whisper function to transcribe a lecture from an audio file.
        """
        transcription = self.llm_pro.whisper(audio_file_path)
        return transcription
    

    def explain(self, lecture_content: str) -> IdeaCardsSchema:
        """
        Analyze the lecture content to identify and explain key terms.
        Outputs a schema of list of idea cards.
        """
        system_prompt = """
        You are a helpful assistant. You are professional. The student is an undergraduate student majoring in a subject at Oxford University. Explain concepts clearly and concisely.

        For each key idea, provide:
        1. Idea Name: A concise name for the idea.
        2. Idea Explanation: A brief explanation of the idea. 
        3. Idea Context: Detailed explanation with at least 3 bullet points. Focus on how things are related.

        Note: if the idea is/involves a formula, please use LaTeX to format it.
        For inline formulas, use single dollar signs: $formula$
        For display formulas (on their own line), use double dollar signs: $$formula$$

        Example:
        - The quadratic formula is given by $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$
        - The area of a circle is:
          $$A = \\pi r^2$$

        Ensure that your explanations are clear and appropriate for an undergraduate level of understanding.
        """
        
        user_prompt = lecture_content

        idea_cards = self.perform_action(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_class=IdeaCardsSchema
        )

        # Log the received idea cards
        logger.info("Received idea cards:")
        for card in idea_cards.idea_cards:
            logger.info(f"Idea Name: {card.idea_name}")
            logger.info(f"Idea Explanation: {card.idea_explanation}")
            logger.info(f"Idea Context: {card.idea_context}")
            logger.info("---")

        return idea_cards

    def build_knowledge(self, pdf_paths: List[str]):
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
