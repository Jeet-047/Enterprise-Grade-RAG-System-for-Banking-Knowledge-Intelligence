import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.pipeline import RAGPipeline

def test_rag_pipeline():
    try:
        print("Initializing RAG Pipeline...")
        pipeline = RAGPipeline()

        print("Preparing vector store...")
        pipeline.prepare_vector_store()

        print("Vector store prepared successfully.")

        # Sample query
        query = "tell me about rag?"

        print(f"Testing with query: {query}")

        # Generate answer (this internally calls retrieve)
        print("Generating answer...")
        response = pipeline.answer(query)

        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(f"Query: {query}")
        print(f"Answer: {response["final_answer"]}")
        print(f"Confident Score: {response["confidence_score"]}")
        print(f"Source: {response["source"]}")
        print("="*50)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_pipeline()
