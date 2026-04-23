from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.observability.logger import logging
from src.exception import MyException
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


class DocumentChunker:
    def __init__(self):
        pass

    def structure_aware_splitter(self, extracted_doc_dict):
        """
        Performs initial splitting based on natural document boundaries.
        Args:
            extracted_doc_dict (dict): A dictionary containing 'text' and 'metadata'.
                                    'metadata' must contain 'doc_type', 'source', 'page', 'section'.
        Returns:
            list: A list of dictionaries, each representing a structurally aware chunk
                with 'text' and updated 'metadata'.
        """
        if 'text' not in extracted_doc_dict or 'metadata' not in extracted_doc_dict:
            raise MyException("Input dictionary must contain 'text' and 'metadata' keys.")
        
        try:
            raw_text = extracted_doc_dict['text']
            metadata = extracted_doc_dict['metadata']
            doc_type = metadata['doc_type']

            logging.info(f"Applying structure-aware splitting for document type: {doc_type}")

            # Define default separators for the documents
            separators = ['\n\n', '\n', ' ', '']

            # Initialize RecursiveCharacterTextSplitter for initial structural chunks
            # Larger chunk_size and no overlap for initial structural split
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=2000,  # Larger chunks for initial structural split
                chunk_overlap=0,
                length_function=len, # Character count for initial split
                add_start_index=True
            )

            # Split the document's text
            # The splitter expects a list of Document objects, so create one from the raw text.
            doc_for_splitting = [Document(page_content=raw_text, metadata=metadata)]
            split_documents = text_splitter.split_documents(doc_for_splitting)

            # Format the split documents into the desired dictionary structure
            formatted_chunks = []
            for i, split_doc in enumerate(split_documents):
                chunk_metadata = split_doc.metadata.copy()
                # Update chunk metadata with more specific chunk information
                chunk_metadata['chunk_id'] = i
                formatted_chunks.append({
                    'text': split_doc.page_content,
                    'metadata': chunk_metadata
                })
            logging.info(f"Original text split into {len(formatted_chunks)} structural chunks.")
            return formatted_chunks
        except Exception as e:
            raise MyException(e, sys)
    
    def chunk_document(self, cleaned_doc_list: list, similarity_threshold: float = 0.8) -> list:
        """
        Combines structure-aware splitting and semantic refinement into a single function
        for processing cleaned documents into final chunks, using parallel processing.

        Args:
            cleaned_doc_list (list): A list of dictionaries, each containing 'text' and 'metadata'
                                    from the cleaned documents.
            similarity_threshold (float): Threshold for semantic similarity in refinement.

        Returns:
            list: A list of dictionaries, each representing a final chunk with 'text' and 'metadata'.
        """
        logging.info("Starting document chunking process with parallel processing...")
        all_final_chunks = []
        try:
            # Use parallel processing for multiple documents
            with ThreadPoolExecutor(max_workers=min(len(cleaned_doc_list), 4)) as executor:
                # Submit tasks for each document
                futures = [executor.submit(process_single_document, doc, similarity_threshold) for doc in cleaned_doc_list]
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_final_chunks.extend(result)
                    except Exception as e:
                        logging.error(f"Error processing document in parallel: {e}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        # Continue with other documents instead of failing completely
                        continue

            logging.info(f"Document chunking process completed. Generated {len(all_final_chunks)} total final chunks from all documents.")
            return all_final_chunks
        except Exception as e:
            logging.error(f"Parallel processing failed, falling back to sequential processing: {e}")
            # Fallback to sequential processing
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
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return semantic_refinement_worker(structural_chunks, similarity_threshold, model)


# Standalone functions for parallel processing (must be at module level for pickling)
def process_single_document(extracted_doc_dict, similarity_threshold):
    """
    Standalone function to process a single document in parallel.
    This function is completely independent and can be pickled for multiprocessing.
    """
    try:
        # Perform structure-aware splitting
        structural_chunks = structure_aware_splitter_standalone(extracted_doc_dict)
        
        # Load model and perform semantic refinement
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        refined_chunks = semantic_refinement_worker(structural_chunks, similarity_threshold, model)
        
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
    """
    Worker function for semantic refinement that can be used in parallel processes.
    """
    refined_chunks = []
    for i, structural_chunk in enumerate(structural_chunks):
        text = structural_chunk['text']
        metadata = structural_chunk['metadata'].copy()

        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if len(sentences) <= 1:
            refined_chunks.append(structural_chunk)
            continue

        embeddings = model.encode(sentences)

        semantic_chunks = []
        current_group = [sentences[0]]
        for j in range(1, len(sentences)):
            similarity = cosine_similarity([embeddings[j-1]], [embeddings[j]])[0][0]
            if similarity < similarity_threshold:
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
