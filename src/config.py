"""
Configuration and constants for the AI System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODELS = [
	"gemini-2.5-flash",
	"gemini-flash-latest",
	"gemini-flash-lite-latest",
	"gemini-2.0-flash-lite-001",
]
GEMINI_MODEL = GEMINI_MODELS[0]

# Database Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
DATASET_PATH = str(BASE_DIR / "data" / "performance_cases.json")

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Collection Names
COLLECTION_NAME = "db_performance_cases"

# Prompt Templates
SYSTEM_PROMPT = """You are an expert database and system performance analyst. 
Your role is to:
1. Understand database query performance issues
2. Identify root causes from provided context
3. Suggest specific, actionable optimizations
4. Explain your reasoning clearly

Always provide structured responses with:
- Problem identification
- Root cause analysis
- Specific optimization suggestions
- Expected improvement percentage
- Confidence level (high/medium/low)

Be concise, technical, and practical."""

ANALYSIS_PROMPT_TEMPLATE = """Based on the following database performance context and similar cases from our knowledge base:

USER QUERY: {query}

CONTEXT FROM SIMILAR CASES:
{context}

Provide a concise analysis of the likely issue and optimizations.

Follow these format instructions exactly:
{format_instructions}
"""

# Retrieval Configuration
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.3

# Response Configuration
DEFAULT_CONFIDENCE = "medium"
