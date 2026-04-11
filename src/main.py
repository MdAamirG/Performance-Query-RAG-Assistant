"""
Main Application: Orchestrates RAG Pipeline + AI Analysis
End-to-End performance query analysis system
"""
from contextlib import asynccontextmanager
import socket
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

try:
    from .rag_pipeline import RAGPipeline
    from .ai_analyzer import AIAnalyzer
except ImportError:
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

    def detect_anomaly(self, user_query: str) -> Dict[str, Any]:
        """Lightweight anomaly detector using query intent and nearest known cases."""
        similar_cases = self.rag.retrieve_similar_cases(user_query, k=3)
        query_text = user_query.lower()

        anomaly_terms = ["anomaly", "spike", "sudden", "abnormal", "regression"]
        query_signals = sum(1 for term in anomaly_terms if term in query_text)

        best_case_name = None
        best_similarity = 0.0
        case_signals = 0
        for case in similar_cases:
            case_name = str(case.get("name", ""))
            similarity = float(case.get("similarity_score", 0))
            if similarity > best_similarity:
                best_similarity = similarity
                best_case_name = case_name

            case_text = (
                case_name
                + " "
                + " ".join(case.get("root_causes", []))
                + " "
                + str(case.get("context", {}).get("description", ""))
            ).lower()
            case_signals += sum(1 for term in anomaly_terms if term in case_text)

        is_anomaly = (query_signals > 0 and best_similarity >= 0.35) or case_signals > 0
        confidence = "high" if best_similarity >= 0.65 else "medium" if best_similarity >= 0.4 else "low"

        reason = (
            f"Query/case anomaly signals={query_signals + case_signals}, "
            f"best_similarity={best_similarity:.2f}, matched_case={best_case_name or 'none'}."
        )

        return {
            "status": "success",
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "reason": reason,
            "matched_case": best_case_name,
        }


class QueryRequest(BaseModel):
    query: str


class AnomalyRequest(BaseModel):
    query: str


class AnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem: Optional[str] = None
    root_causes: List[str] = []
    suggestions: List[str] = []
    confidence: Optional[str] = None
    priority: Optional[str] = None
    expected_improvement: Optional[str] = None
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


class RootResponse(BaseModel):
    message: str
    docs: str
    health: str


class AnomalyResponse(BaseModel):
    status: str
    is_anomaly: bool
    confidence: str
    reason: str
    matched_case: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.system = PerformanceAnalysisSystem()
    yield


app = FastAPI(title="Solartis Performance Analyzer API", lifespan=lifespan)


def get_system() -> PerformanceAnalysisSystem:
    system = getattr(app.state, "system", None)
    if system is None:
        raise RuntimeError("PerformanceAnalysisSystem is not initialized")
    return system


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        message="Solartis API is running",
        docs="/docs",
        health="/health",
    )


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
        response = get_system().analyze_query(user_query)
        return AnalyzeQueryResponse.model_validate(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/detect/anomaly",
    response_model=AnomalyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
def detect_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        response = get_system().detect_anomaly(user_query)
        return AnomalyResponse.model_validate(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    preferred_port = 8000

    for candidate_port in range(preferred_port, preferred_port + 11):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate_port))
            except OSError:
                continue
            preferred_port = candidate_port
            break

    uvicorn.run(app, host="127.0.0.1", port=preferred_port, reload=False)
