"""
Test Suite: Validates RAG system without API calls
Run before adding Gemini integration for full pipeline
"""
import sys
import os
import json

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pipeline import RAGPipeline


def create_rag_pipeline():
    """Create a RAG pipeline instance for tests and standalone execution"""
    print("\n" + "=" * 80)
    print("TEST 1: RAG Pipeline Initialization")
    print("=" * 80)
    
    try:
        rag = RAGPipeline()
        print("✓ RAG pipeline initialized successfully")
        return rag
    except Exception as e:
        print(f"✗ Failed to initialize RAG: {e}")
        return None


@pytest.fixture(scope="module")
def rag():
    rag_instance = create_rag_pipeline()
    assert rag_instance is not None, "Failed to initialize RAG pipeline"
    return rag_instance


def test_rag_initialization(rag):
    """Test RAG pipeline initialization"""
    assert rag is not None


def test_dataset_loading(rag):
    """Test dataset loading"""
    print("\n" + "=" * 80)
    print("TEST 2: Dataset Loading")
    print("=" * 80)
    
    cases = rag.get_all_cases()
    print(f"✓ Loaded {len(cases)} performance cases")
    
    required_fields = ["case_id", "name", "query", "execution_time_ms", "root_causes", "suggested_optimizations"]
    for case in cases:
        for field in required_fields:
            assert field in case, f"Missing field '{field}' in case {case.get('case_id')}"
    
    print(f"✓ All cases have required fields")
    print("\nSample cases:")
    for case in cases[:3]:
        print(f"  Case {case['case_id']}: {case['name']}")


def test_query_retrieval(rag):
    """Test semantic query retrieval"""
    print("\n" + "=" * 80)
    print("TEST 3: Semantic Query Retrieval")
    print("=" * 80)
    
    test_queries = [
        "Why is my query slow?",
        "JSON field filtering performance",
        "High frequency queries",
        "Database anomalies",
        "Table joins optimization"
    ]
    
    for query in test_queries:
        results = rag.retrieve_similar_cases(query, k=2)
        print(f"\nQuery: '{query}'")
        print(f"  Retrieved {len(results)} cases:")
        for result in results:
            score = result.get('similarity_score', 0)
            print(f"    - {result['name']} (similarity: {score:.2%})")


def test_context_formatting(rag):
    """Test context formatting for LLM"""
    print("\n" + "=" * 80)
    print("TEST 4: Context Formatting")
    print("=" * 80)
    
    test_query = "Why does my JSON extraction query take 25 seconds?"
    cases = rag.retrieve_similar_cases(test_query, k=3)
    context = rag.format_cases_for_context(cases)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Formatted context ({len(context)} characters):")
    print("-" * 80)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 80)
    
    assert "Query:" in context, "Context missing 'Query' field"
    assert "Root Causes:" in context, "Context missing 'Root Causes' field"
    assert "Suggested Optimizations:" in context, "Context missing 'Suggested Optimizations' field"
    print("✓ Context formatting valid")


def test_case_retrieval_by_id(rag):
    """Test direct case retrieval by ID"""
    print("\n" + "=" * 80)
    print("TEST 5: Case Retrieval by ID")
    print("=" * 80)
    
    case = rag.get_case_by_id(1)
    assert case is not None, "Failed to retrieve case by ID"
    print(f"✓ Retrieved case: {case['name']}")
    print(f"  Severity: {case['severity']}")
    print(f"  Execution Time: {case['execution_time_ms']}ms")


def test_retrieval_consistency(rag):
    """Test that retrieval is deterministic"""
    print("\n" + "=" * 80)
    print("TEST 6: Retrieval Consistency")
    print("=" * 80)
    
    query = "Performance optimization"
    
    results1 = rag.retrieve_similar_cases(query, k=2)
    results2 = rag.retrieve_similar_cases(query, k=2)
    
    ids1 = [r['case_id'] for r in results1]
    ids2 = [r['case_id'] for r in results2]
    
    assert ids1 == ids2, "Retrieval is not consistent"
    print(f"✓ Retrieval is consistent")
    print(f"  Same cases retrieved: {ids1}")


def test_empty_retrieval(rag):
    """Test edge case: no results"""
    print("\n" + "=" * 80)
    print("TEST 7: Edge Cases")
    print("=" * 80)
    
    # Test with empty context
    empty_cases = []
    context = rag.format_cases_for_context(empty_cases)
    assert "No similar cases" in context, "Empty case handling failed"
    print("✓ Handles empty case list gracefully")


def test_performance_metrics(rag):
    """Test performance characteristics"""
    print("\n" + "=" * 80)
    print("TEST 8: Performance Metrics")
    print("=" * 80)
    
    import time
    
    # Time embedding + retrieval
    start = time.time()
    for _ in range(5):
        rag.retrieve_similar_cases("test query")
    elapsed = time.time() - start
    avg_time = elapsed / 5
    
    print(f"✓ Average retrieval time: {avg_time*1000:.2f}ms (5 iterations)")
    print(f"  Total time: {elapsed:.2f}s")
    
    if avg_time < 1.0:
        print("  Status: ✓ Acceptable (< 1s)")
    else:
        print("  Status: ⚠ Slow (consider optimization)")


def run_all_tests():
    """Run full test suite"""
    print("\n" + "=" * 80)
    print("SOLARTIS RAG SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Initialize
    rag = create_rag_pipeline()
    if not rag:
        print("\n❌ Tests failed: Cannot initialize RAG")
        return False
    
    try:
        # Test 2-8: Other tests
        test_dataset_loading(rag)
        test_query_retrieval(rag)
        test_context_formatting(rag)
        test_case_retrieval_by_id(rag)
        test_retrieval_consistency(rag)
        test_empty_retrieval(rag)
        test_performance_metrics(rag)
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nNext Step: Integrate Gemini API for AI analysis")
        print("(See Phase 2: AI Analysis Layer)\n")
        return True
    
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
