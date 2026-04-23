import numpy as np

class HallucinationDetector:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def compute_similarity(self, answer, context):
        from sklearn.metrics.pairwise import cosine_similarity

        # Encode the answer and context into embeddings
        answer_embedding = self.model.encode([answer])
        context_embedding = self.model.encode(context)
        
        # Compute cosine similarity between the answer and each context chunk
        similarities = cosine_similarity(answer_embedding, context_embedding)[0]
        
        return similarities
    
    def detect_hallucination(self, answer, context, threshold=0.4):
        """
        Detect if the answer is hallucinated by comparing it to the context.
        
        Args:
            answer: The generated answer (string)
            context: The context chunks (string or list of strings)
            threshold: Similarity threshold for hallucination detection
        
        Returns:
            Dictionary with hallucination detection result and similarity score
        """
        # Split context into chunks if it's a single string
        if isinstance(context, str):
            context_chunks = [context]
        else:
            context_chunks = context
        
        # Compute similarity between answer and context chunks
        similarities = self.compute_similarity(answer, context_chunks)
        
        # Take the maximum similarity (best match with any context chunk)
        max_similarity = float(np.max(similarities))
        
        return {
            "is_hallucinated": max_similarity < threshold,
            "similarity_score": max_similarity,
        }