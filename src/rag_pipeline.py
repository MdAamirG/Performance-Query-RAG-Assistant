"""
RAG Pipeline: Vector Database and Context Retrieval
Handles embeddings, storage, and retrieval of performance cases
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from config import (
    CHROMA_PERSIST_DIR,
    DATASET_PATH,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline with embeddings and vector DB"""
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Chroma client
        if not os.path.exists(CHROMA_PERSIST_DIR):
            os.makedirs(CHROMA_PERSIST_DIR)
        
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.performance_cases = []
        self._load_dataset()

    def _load_dataset(self):
        """Load performance cases from JSON file"""
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
        with open(DATASET_PATH, 'r') as f:
            self.performance_cases = json.load(f)
        
        # Populate vector DB if empty
        if self.collection.count() == 0:
            self._populate_vector_db()

    def _populate_vector_db(self):
        """Populate vector database with performance cases"""
        print(f"Populating vector database with {len(self.performance_cases)} cases...")
        
        for case in self.performance_cases:
            case_id = f"case_{case['case_id']}"
            
            # Create searchable text from case
            searchable_text = self._create_searchable_text(case)
            
            # Store in Chroma
            self.collection.add(
                ids=[case_id],
                documents=[searchable_text],
                metadatas=[{
                    "case_id": case["case_id"],
                    "name": case["name"],
                    "severity": case["severity"],
                    "execution_time_ms": case["execution_time_ms"],
                    "frequency_per_hour": case["frequency_per_hour"],
                }],
                embeddings=[self.embedding_model.encode(searchable_text).tolist()]
            )
        
        print(f"✓ Vector database populated with {len(self.performance_cases)} cases")

    def _create_searchable_text(self, case: Dict[str, Any]) -> str:
        """Convert case to searchable text for embeddings"""
        parts = [
            case["name"],
            case["query"],
            " ".join(case.get("root_causes", [])),
            " ".join(case.get("suggested_optimizations", [])),
            case.get("context", {}).get("description", ""),
        ]
        return " ".join(parts)

    def retrieve_similar_cases(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Retrieve the k most similar cases from the vector database
        Returns: List of similar performance cases
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances", "documents"]
        )
        
        retrieved_cases = []
        
        if results["ids"] and len(results["ids"]) > 0:
            for i, case_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Find full case data
                full_case = next(
                    (c for c in self.performance_cases if c["case_id"] == metadata["case_id"]),
                    None
                )
                
                if full_case:
                    retrieved_cases.append({
                        **full_case,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                    })
        
        return retrieved_cases

    def get_case_by_id(self, case_id: int) -> Dict[str, Any]:
        """Get a specific case by ID"""
        return next((c for c in self.performance_cases if c["case_id"] == case_id), None)

    def get_all_cases(self) -> List[Dict[str, Any]]:
        """Get all performance cases"""
        return self.performance_cases

    def format_cases_for_context(self, cases: List[Dict[str, Any]]) -> str:
        """Format retrieved cases into context string for LLM"""
        if not cases:
            return "No similar cases found in knowledge base."
        
        context_parts = []
        for i, case in enumerate(cases, 1):
            context_parts.append(f"""
Case {i}: {case['name']}
Query: {case['query']}
Execution Time: {case['execution_time_ms']}ms
Frequency: {case['frequency_per_hour']}/hour
Severity: {case['severity']}
Root Causes: {', '.join(case.get('root_causes', []))}
Suggested Optimizations: {', '.join(case.get('suggested_optimizations', [:3]))}
Expected Improvement: {case.get('expected_improvement_percent', 0)}%
Similarity Score: {case.get('similarity_score', 0):.2%}
""")
        
        return "\n".join(context_parts)
