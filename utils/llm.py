"""
LLM query module.
Handles communication with the Groq API for question answering.
Uses a two-model pipeline: reasoner (hidden) + formatter (clean output).
"""

from typing import List, Tuple, Optional
import os
import re
from groq import Groq


class LLMClient:
    """
    Client for querying the Groq API with context from retrieved documents.
    Uses a two-stage pipeline:
      1. Reasoner model (hidden) — can think freely
      2. Formatter model — outputs ONLY the final cleaned answer
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq client.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
        """
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY in .env file "
                "or pass it directly."
            )
        self.client = Groq(api_key=key)
        self.model = "qwen/qwen3.6-27b"

    def _call_llm(self, system: str, user: str, temp: float = 0.0) -> str:
        """Make a single LLM call and return the raw response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temp,
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def _post_process(self, text: str) -> str:
        """Strip any reasoning/thinking traces from output."""
        # Remove  thinking... response blocks
        text = re.sub(r'(?is)<[^>]*think[^>]*>.*?</[^>]*think[^>]*>', '', text)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
        # Remove numbered step patterns
        text = re.sub(r'(?m)^(?:Step\s+\d+|步骤\s+\d+)[\s\S]*?(?=\n(?:Step\s+\d+|步骤\s+\d+)|\Z)', '', text)
        # Remove meta-commentary lines
        text = re.sub(r'(?i)(?:check\s+(?:against\s+)?constraints|analyze\s+(?:the\s+)?requirements?).*?\n', '', text)
        text = re.sub(r'(?m)^(?:Draft|Refined|Revised|Let me|I\'ll|I will|Here\'s|Here is) .*?\n', '', text)
        # Remove any leftover lines containing reasoning keywords
        filtered = []
        for line in text.split('\n'):
            lower = line.lower()
            if any(x in lower for x in ['analysis', 'step', 'check', 'verify', 'draft', 'refined', 'thinking']):
                continue
            filtered.append(line)
        text = '\n'.join(filtered)
        # Clean up blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Tuple[str, float]],
        temperature: float = 0.0
    ) -> str:
        """
        Generate an answer to a question using retrieved context.

        Pipeline:
          1. Reasoner (hidden) — thinks freely, output discarded
          2. Formatter — outputs ONLY the final clean answer

        Args:
            question: The user's question
            context_chunks: List of (chunk_text, relevance_score) tuples
            temperature: LLM temperature for response generation

        Returns:
            Generated answer text
        """
        if not context_chunks:
            return "No relevant context found in the document to answer this question."

        # Build context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(context_chunks, 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.1%})\n{chunk}")

        context = "\n\n".join(context_parts)

        # --- Stage 1: Reasoner (hidden, can think freely) ---
        reasoner_system = (
            "You are a reasoning engine. Think step-by-step about the question "
            "and the provided context. Your output will NOT be shown to the user."
        )
        reasoner_prompt = f"""Context:
{context}

Question: {question}

Think through this carefully. What does the context say?"""

        try:
            reasoning = self._call_llm(reasoner_system, reasoner_prompt, temp=0.3)
        except Exception:
            reasoning = ""

        # --- Stage 2: Formatter (outputs ONLY clean answer) ---
        formatter_system = (
            "You are a strict answer formatting engine.\n\n"
            "You MUST output ONLY the final answer.\n\n"
            "ABSOLUTE RULES:\n"
            "- Do NOT output reasoning\n"
            "- Do NOT output analysis\n"
            "- Do NOT output thoughts, hidden steps, or planning\n"
            "- Do NOT use tags like  thinking, <reasoning>, or explanations of your process\n"
            "- Do NOT evaluate or verify your answer\n"
            "- Do NOT include meta commentary\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a clean, final response in natural language.\n\n"
            "If context is insufficient, respond EXACTLY:\n"
            '"The document doesn\'t contain information about that."'
        )

        formatter_prompt = f"""Context:
{context}

Question: {question}

Reasoning (internal, do not repeat):
{reasoning}

Now output ONLY the final answer to the user's question. No reasoning, no steps, no analysis."""

        try:
            raw = self._call_llm(formatter_system, formatter_prompt, temp=0.0)
            return self._post_process(raw)
        except Exception as e:
            return f"Error generating answer: {str(e)}"