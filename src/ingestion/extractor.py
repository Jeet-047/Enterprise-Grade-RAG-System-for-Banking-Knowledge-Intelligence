import os
import sys
from src.observability.logger import logging
from src.exception import MyException

class DocumentExtractor:
    def extract_document_info(self, document: list, doc_path: str) -> list:
        if not document:
            return []

        logging.info("Starting the data extraction process")

        if doc_path.startswith(('http://', 'https://')):
            doc_type = "web"
        else:
            ext = os.path.splitext(doc_path)[1].lstrip('.').lower()
            doc_type = 'docx' if ext == 'doc' else ext

        all_extracted_data = []
        for i, doc in enumerate(document):
            current_doc_info = doc.metadata.copy()
            source_path = current_doc_info.get('source', doc_path)

            metadata = {
                'doc_type': doc_type,
                'source': source_path if source_path.startswith(('http://', 'https://')) else os.path.basename(source_path),
                'page': current_doc_info.get('page', i) + 1,
                'section': current_doc_info.get('section', 'N/A')
            }

            all_extracted_data.append({
                'text': doc.page_content,
                'metadata': metadata
            })
            logging.info(f"Extracted text and metadata for page/part {i+1}")

        return all_extracted_data