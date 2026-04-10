"""
Main Application: Orchestrates RAG Pipeline + AI Analysis
End-to-End performance query analysis system
"""
import sys
import json
from typing import Dict, Any
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

    def format_response(self, result: Dict[str, Any]) -> str:
        """Format analysis result for display"""
        analysis = result.get("analysis", {})
        
        output = []
        output.append("=" * 80)
        output.append("PERFORMANCE ANALYSIS REPORT")
        output.append("=" * 80)
        
        # Problem
        output.append(f"\n🔴 PROBLEM:")
        output.append(f"  {analysis.get('problem', 'Unknown issue')}")
        
        # Root Causes
        output.append(f"\n🔍 ROOT CAUSES:")
        for cause in analysis.get('root_causes', []):
            output.append(f"  • {cause}")
        
        # Suggestions
        output.append(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in analysis.get('suggestions', []):
            output.append(f"  • {suggestion}")
        
        # Metrics
        output.append(f"\n📊 METRICS:")
        output.append(f"  Priority: {analysis.get('priority', 'Unknown').upper()}")
        output.append(f"  Confidence: {analysis.get('confidence', 'Unknown').upper()}")
        output.append(f"  Expected Improvement: {analysis.get('expected_improvement_percent', 0)}%")
        
        # Reasoning
        output.append(f"\n📝 REASONING:")
        output.append(f"  {analysis.get('reasoning', 'N/A')[:300]}...")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)

    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "=" * 80)
        print("SOLARTIS DATABASE PERFORMANCE ANALYZER - Interactive Mode")
        print("=" * 80)
        print("\nEnter your database performance questions.")
        print("Type 'exit' or 'quit' to leave.")
        print("Type 'list' to see available cases.")
        print("\n" + "=" * 80 + "\n")
        
        while True:
            try:
                user_input = input("Your question: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\n👋 Thank you for using the Performance Analyzer. Goodbye!")
                    break
                
                if user_input.lower() == 'list':
                    self._list_cases()
                    continue
                
                if not user_input:
                    print("Please enter a question.\n")
                    continue
                
                # Analyze the query
                result = self.analyze_query(user_input)
                
                # Display formatted response
                print(self.format_response(result))
                
                # Option to see raw response
                if input("\nShow raw AI response? (y/n): ").lower() == 'y':
                    print("\n--- RAW RESPONSE ---")
                    print(result.get("raw_response", "N/A"))
                    print("--- END RAW RESPONSE ---\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")

    def _list_cases(self):
        """List all available performance cases"""
        cases = self.rag.get_all_cases()
        print("\n" + "=" * 80)
        print("AVAILABLE PERFORMANCE CASES")
        print("=" * 80)
        
        for case in cases:
            print(f"\nCase {case['case_id']}: {case['name']}")
            print(f"  Query: {case['query'][:60]}...")
            print(f"  Severity: {case['severity']} | Execution: {case['execution_time_ms']}ms")
        
        print("\n" + "=" * 80 + "\n")

    def batch_analyze(self, queries: list) -> list:
        """Analyze multiple queries and return results"""
        results = []
        for query in queries:
            result = self.analyze_query(query)
            results.append({
                "query": query,
                "result": result
            })
        return results


def main():
    """Main entry point"""
    # Initialize system
    system = PerformanceAnalysisSystem()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Batch mode: analyze provided queries
        queries = sys.argv[1:]
        print(f"\n📊 Analyzing {len(queries)} queries...\n")
        
        results = system.batch_analyze(queries)
        
        for item in results:
            print(system.format_response(item["result"]))
    else:
        # Interactive mode
        system.interactive_mode()


if __name__ == "__main__":
    main()
