"""
Main Application: Orchestrates RAG Pipeline + AI Analysis
End-to-End performance query analysis system
"""
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from rag_pipeline import RAGPipeline
from ai_analyzer import AIAnalyzer


class PerformanceAnalysisSystem:
    def __init__(self):
        """Initialize the complete system"""
        print("🚀 Initializing Performance Analysis System...")
        self.rag = RAGPipeline()
        self.analyzer = AIAnalyzer()
        print("✓ System ready!\n")

    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point: Analyze a performance query
        
        Flow:
        1. Retrieve similar cases from vector DB
        2. Format context
        3. Send to Gemini for analysis
        4. Return structured response
        """
        print(f"\n📋 Analyzing query: {user_query}\n")
        
        # Step 1: Retrieve similar cases
        print("🔍 Retrieving similar cases from knowledge base...")
        similar_cases = self.rag.retrieve_similar_cases(user_query, k=3)
        
        if similar_cases:
            print(f"✓ Found {len(similar_cases)} similar cases:")
            for case in similar_cases:
                print(f"  - {case['name']} (similarity: {case.get('similarity_score', 0):.2%})")
        else:
            print("⚠ No highly similar cases found, using general knowledge")
        
        # Step 2: Format context for AI
        context = self.rag.format_cases_for_context(similar_cases)
        
        # Step 3: Analyze with Gemini
        print("\n🤖 Analyzing with AI...")
        result = self.analyzer.analyze_query(user_query, context)
        
        if result["status"] == "success":
            print("✓ Analysis complete\n")
        else:
            print(f"⚠ Analysis completed with warnings: {result.get('error', 'Unknown error')}\n")
        
        return result


class QueryRequest(BaseModel):
    query: str


class AnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem: Optional[str] = None
    root_causes: List[str] = []
    suggestions: List[str] = []
    confidence: Optional[str] = None
    priority: Optional[str] = None
    expected_improvement_percent: Optional[float] = None
    reasoning: Optional[str] = None


class AnalyzeQueryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    analysis: AnalysisPayload
    raw_response: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


app = FastAPI(title="Solartis Performance Analyzer API")
system = PerformanceAnalysisSystem()


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/analyze/query",
    response_model=AnalyzeQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
def analyze_query(request: QueryRequest) -> AnalyzeQueryResponse:
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        response = system.analyze_query(user_query)
        return AnalyzeQueryResponse.model_validate(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
