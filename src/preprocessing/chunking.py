import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.observability.logger import logging
from src.exception import MyException
from src.embedding.embedder import NVIDIAEmbedder
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


class DocumentChunker:
    def __init__(self):
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = NVIDIAEmbedder().get_embedder()
        return self._embedder

    def structure_aware_splitter(self, extracted_doc_dict):
        if 'text' not in extracted_doc_dict or 'metadata' not in extracted_doc_dict:
            raise MyException("Input dictionary must contain 'text' and 'metadata' keys.")

        try:
            doc_type = extracted_doc_dict['metadata']['doc_type']
            logging.info(f"Applying structure-aware splitting for document type: {doc_type}")
            return structure_aware_splitter_standalone(extracted_doc_dict)
        except Exception as e:
            raise MyException(e, sys)
    
    def chunk_document(self, cleaned_doc_list: list, similarity_threshold: float = 0.8) -> list:
        logging.info("Starting document chunking process with parallel processing...")
        all_final_chunks = []

        if len(cleaned_doc_list) == 1:
            return self._sequential_chunking(cleaned_doc_list, similarity_threshold)

        try:
            num_workers = min(len(cleaned_doc_list), 4)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.process_single_document, doc, similarity_threshold): doc for doc in cleaned_doc_list}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_final_chunks.extend(result)
                    except Exception as e:
                        logging.error(f"Error processing document: {e}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        continue

            logging.info(f"Document chunking process completed. Generated {len(all_final_chunks)} total final chunks from all documents.")
            return all_final_chunks
        except Exception as e:
            logging.error(f"Parallel processing failed, falling back to sequential processing: {e}")
            return self._sequential_chunking(cleaned_doc_list, similarity_threshold)

    def _sequential_chunking(self, cleaned_doc_list: list, similarity_threshold: float = 0.8) -> list:
        """
        Fallback sequential processing when parallel processing fails.
        """
        logging.info("Using sequential processing as fallback...")
        all_final_chunks = []
        try:
            for doc in cleaned_doc_list:
                structural_chunks = self.structure_aware_splitter(doc)
                refined_chunks = self._semantic_refinement(structural_chunks, similarity_threshold)
                all_final_chunks.extend(refined_chunks)
            return all_final_chunks
        except Exception as e:
            raise MyException(e, sys)

    def _semantic_refinement(self, structural_chunks, similarity_threshold):
        """
        Perform semantic refinement on structural chunks.
        """
        return semantic_refinement_worker(structural_chunks, similarity_threshold, self.embedder)


    # Standalone functions for parallel processing (must be at module level for pickling)
    def process_single_document(self, extracted_doc_dict, similarity_threshold):
        """
        Standalone function to process a single document in parallel.
        This function is completely independent and can be pickled for multiprocessing.
        """
        try:
            # Perform structure-aware splitting
            structural_chunks = structure_aware_splitter_standalone(extracted_doc_dict)
            
            # Load model and perform semantic refinement
            refined_chunks = semantic_refinement_worker(structural_chunks, similarity_threshold, self.embedder)
            
            return refined_chunks
        except Exception as e:
            # Re-raise with more context for debugging
            raise Exception(f"Error processing document: {e}")


def structure_aware_splitter_standalone(extracted_doc_dict):
    """
    Standalone version of structure-aware splitting that doesn't depend on class instance.
    """
    if 'text' not in extracted_doc_dict or 'metadata' not in extracted_doc_dict:
        raise ValueError("Input dictionary must contain 'text' and 'metadata' keys.")
    
    raw_text = extracted_doc_dict['text']
    metadata = extracted_doc_dict['metadata']

    # Define default separators for the documents
    separators = ['\n\n', '\n', ' ', '']

    # Initialize RecursiveCharacterTextSplitter for initial structural chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=2000,  # Larger chunks for initial structural split
        chunk_overlap=0,
        length_function=len,  # Character count for initial split
        add_start_index=True
    )

    # Split the document's text
    doc_for_splitting = [Document(page_content=raw_text, metadata=metadata)]
    split_documents = text_splitter.split_documents(doc_for_splitting)

    # Format the split documents into the desired dictionary structure
    formatted_chunks = []
    for i, split_doc in enumerate(split_documents):
        chunk_metadata = split_doc.metadata.copy()
        chunk_metadata['chunk_id'] = i
        formatted_chunks.append({
            'text': split_doc.page_content,
            'metadata': chunk_metadata
        })
    
    return formatted_chunks


def semantic_refinement_worker(structural_chunks, similarity_threshold, model):
    refined_chunks = []
    for i, structural_chunk in enumerate(structural_chunks):
        text = structural_chunk['text']
        metadata = structural_chunk['metadata'].copy()

        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if len(sentences) <= 1:
            refined_chunks.append(structural_chunk)
            continue

        embeddings = model.embed_documents(sentences)
        embeddings_matrix = np.asarray(embeddings, dtype=float)

        if embeddings_matrix.ndim != 2 or embeddings_matrix.shape[0] < 2:
            refined_chunks.append(structural_chunk)
            continue

        # Pairwise adjacent similarity as scalar per boundary (n-1 values).
        similarities = np.einsum("ij,ij->i", embeddings_matrix[:-1], embeddings_matrix[1:])

        semantic_chunks = []
        current_group = [sentences[0]]
        for j in range(1, len(sentences)):
            if float(similarities[j - 1]) < similarity_threshold:
                semantic_chunks.append('. '.join(current_group) + '.')
                current_group = [sentences[j]]
            else:
                current_group.append(sentences[j])
        if current_group:
            semantic_chunks.append('. '.join(current_group) + '.')

        for k, chunk_text in enumerate(semantic_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = f"{metadata.get('chunk_id', i)}-{k}"
            refined_chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
    return refined_chunks
