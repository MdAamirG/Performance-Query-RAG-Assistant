"""
AI Analysis Layer: Gemini Integration
Analyzes database queries and performance issues using Google Gemini API
"""
import json
from typing import Dict, Any, List

from google import genai
from google.genai import types
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

try:
    from .config import (
        GEMINI_API_KEY,
        GEMINI_MODELS,
        SYSTEM_PROMPT,
        ANALYSIS_PROMPT_TEMPLATE,
        DEFAULT_CONFIDENCE,
    )
except ImportError:
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODELS,
        SYSTEM_PROMPT,
        ANALYSIS_PROMPT_TEMPLATE,
        DEFAULT_CONFIDENCE,
    )


class PerformanceAnalysis(BaseModel):
    problem: str
    root_causes: List[str]
    suggestions: List[str]
    confidence: str
    expected_improvement: str


class AIAnalyzer:
    def __init__(self):
        """Initialize Gemini API client"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.parser = PydanticOutputParser(pydantic_object=PerformanceAnalysis)
        self.prompt_template = PromptTemplate(
            template=ANALYSIS_PROMPT_TEMPLATE,
            input_variables=["query", "context"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
            },
        )

    def analyze_query(self, user_query: str, context: str) -> Dict[str, Any]:
        """
        Analyze a database query using Gemini with RAG context
        
        Args:
            user_query: The user's question about performance
            context: Retrieved context from similar cases
        
        Returns:
            Structured analysis response
        """
        try:
            # Prepare the prompt
            prompt = self.prompt_template.format(
                query=user_query,
                context=context,
            )
            
            # Call Gemini API with fallback models for transient capacity issues
            response = self._generate_content_with_fallback(prompt)
            response_text = response.text or ""
            
            # Parse response with strict schema validation
            analysis = self.parser.parse(response_text).model_dump()
            
            return {
                "status": "success",
                "analysis": analysis,
                "raw_response": response_text,
            }
        
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis": self._create_fallback_response(user_query, context),
            }

    def _generate_content_with_fallback(self, prompt: str):
        """Try multiple Gemini models so a temporary outage does not fail the request."""
        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                return self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                    ),
                )
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if "503" not in message and "UNAVAILABLE" not in message and "resource exhausted" not in message.lower():
                    raise
                print(f"Gemini model {model_name} unavailable, trying next fallback...")

        if last_error:
            raise last_error
        raise RuntimeError("No Gemini models available")

    def _create_fallback_response(self, query: str, context: str) -> Dict[str, Any]:
        """Create a fallback response when API fails"""
        return {
            "problem": "Database performance issue requiring optimization",
            "root_causes": [
                "Potential missing indexes",
                "Inefficient query structure",
                "Lack of result filtering",
            ],
            "suggestions": [
                "Review query EXPLAIN PLAN analysis",
                "Add appropriate indexes on filter columns",
                "Consider query restructuring or denormalization",
                "Implement caching for frequently accessed data",
            ],
            "confidence": DEFAULT_CONFIDENCE,
            "expected_improvement": "Moderate improvement expected with indexing and query tuning.",
        }

    def evaluate_query_correctness(self, query: str) -> Dict[str, Any]:
        """Evaluate if a SQL query looks valid and reasonable"""
        prompt = f"""Evaluate this SQL query for potential issues:
Query: {query}

Check for:
1. Syntax validity
2. Common performance anti-patterns
3. Security concerns (SQL injection vectors)
4. Data structure assumptions

Respond with JSON: {{"valid": bool, "issues": [list], "confidence": "high/medium/low"}}"""
        
        try:
            response = self._generate_content_with_fallback(prompt)
            return json.loads(response.text or "{}")
        except Exception as e:
            return {
                "valid": True,
                "issues": [],
                "confidence": "low",
                "error": str(e),
            }
