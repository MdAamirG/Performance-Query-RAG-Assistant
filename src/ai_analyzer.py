"""
AI Analysis Layer: Gemini Integration
Analyzes database queries and performance issues using Google Gemini API
"""
import json
import google.generativeai as genai
from typing import Dict, Any
import re
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    DEFAULT_CONFIDENCE,
)


class AIAnalyzer:
    def __init__(self):
        """Initialize Gemini API client"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT
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
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                query=user_query,
                context=context
            )
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse response
            analysis = self._parse_response(response_text)
            
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

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured JSON from Gemini response
        Handles cases where JSON might be wrapped in markdown code blocks
        """
        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Try direct JSON parsing
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
            
            # Fallback: create structured response from text
            return self._parse_text_response(response_text)
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return self._parse_text_response(response_text)

    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse unstructured text response into structured format"""
        return {
            "problem": self._extract_section(text, "Problem", "likely performance issue"),
            "root_causes": self._extract_list(text, "Root Cause"),
            "suggestions": self._extract_list(text, "Optimization"),
            "confidence": DEFAULT_CONFIDENCE,
            "expected_improvement_percent": 50,
            "priority": "high",
            "reasoning": text[:500] + "..." if len(text) > 500 else text,
        }

    def _extract_section(self, text: str, keyword: str, default: str = "") -> str:
        """Extract section from text after a keyword"""
        lines = text.split('\n')
        found = False
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                found = True
                # Get the next non-empty line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        return lines[j].strip()
                break
        return default

    def _extract_list(self, text: str, keyword: str) -> list:
        """Extract list items after a keyword"""
        items = []
        lines = text.split('\n')
        found = False
        
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                found = True
                continue
            
            if found:
                # Stop at next section header or empty after collecting items
                if line.startswith('#') or line.startswith('**'):
                    break
                if line.strip().startswith('- ') or line.strip().startswith('• '):
                    items.append(line.strip()[2:].strip())
                elif line.strip() and not line.startswith(' '):
                    if len(items) > 0:
                        break
        
        return items if items else ["Unable to extract suggestions"]

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
            "confidence": "low",
            "expected_improvement_percent": 30,
            "priority": "medium",
            "reasoning": f"Fallback response: Unable to analyze specific query. Please review database logs and consider: {context[:200]}",
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
            response = self.model.generate_content(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            return {
                "valid": True,
                "issues": [],
                "confidence": "low",
                "error": str(e),
            }
