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
        query = "Tell me about risk and return?"

        print(f"Testing with query: {query}")

        # Retrieve documents
        print("Retrieving relevant documents...")
        documents = pipeline.retrieve(query)
        print(f"Retrieved {len(documents)} documents.")

        # Generate answer
        print("Generating answer...")
        answer = pipeline.answer(query)

        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(f"Query: {query}")
        print(f"Number of retrieved documents: {len(documents)}")
        if documents:
            print("Sample document content (first 200 chars):")
            print(documents[0].page_content[:200] + "...")
        print(f"Answer: {answer}")
        print("="*50)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_pipeline()
