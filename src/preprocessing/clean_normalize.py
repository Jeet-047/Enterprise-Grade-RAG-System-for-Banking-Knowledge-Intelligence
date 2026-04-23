import re
from bs4 import BeautifulSoup
from src.observability.logger import logging
from src.exception import MyException
import os, sys

class DocumentNormalizationAndCleaning:
    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """
        General-purpose text normalization:
        - unify line breaks
        - collapse excessive spaces
        - collapse many blank lines into single paragraph breaks
        - trim leading/trailing whitespace

        This should be applied AFTER structure-specific cleaning.
        """
        # Normalize Windows-style newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse tabs into spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ blank lines into just 2 (paragraph separation)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Remove extra spaces around newlines
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        # Finally, collapse multiple spaces again (in case we introduced any)
        text = re.sub(r" {2,}", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()

        return text
    
    def clean_document_structure(self, extracted_doc: list) -> list:
        """
        Cleans the document text based on its document type.
        Args:
            extracted_doc (list): A list of dictionaries, each containing 'text' and 'metadata'.
                                    'metadata' must contain 'doc_type'.
        Returns:
            list: An updated list of dictionaries with cleaned text for each document.
        """
        cleaned_document_list = []

        for extracted_doc_dict in extracted_doc:
            if 'text' not in extracted_doc_dict or 'metadata' not in extracted_doc_dict:
                raise MyException("Each input dictionary must contain 'text' and 'metadata' keys.", sys)
            if 'doc_type' not in extracted_doc_dict['metadata']:
                raise MyException("Metadata must contain 'doc_type' key.", sys)

            raw_text = extracted_doc_dict['text']
            doc_type = extracted_doc_dict['metadata']['doc_type']
            cleaned_text = raw_text # Initialize with raw text, cleaning methods will modify this

            logging.info(f"Cleaning document of type: {doc_type}")
            logging.info(f"Original text length: {len(raw_text)}")

            if doc_type == 'web':
                logging.info("Applying Web specific cleaning with BeautifulSoup...")
                soup = BeautifulSoup(raw_text, 'html.parser')

                # Remove script and style elements
                for script_or_style in soup(['script', 'style']):
                    script_or_style.extract() # Remove them from the soup

                # Get text
                cleaned_text = soup.get_text()
            elif doc_type == 'csv':
                logging.info("Applying CSV specific cleaning...")

                # For CSV, we might want to preserve line breaks but remove excessive whitespace
                cleaned_text = re.sub(r"[ \t]+", " ", raw_text)  # Collapse spaces and tabs
                cleaned_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_text)  # Collapse multiple blank lines
                
                # Trim leading/trailing whitespace
                cleaned_text = cleaned_text.strip()
            else: # General text (pdf, docx). No specific structural cleaning needed before normalization.
                logging.info(f"No specific structural cleaning for {doc_type}. Applying general text normalization.")
                cleaned_text = raw_text

            # Update the text in the current dictionary and append to the new list
            extracted_doc_dict['text'] = cleaned_text
            cleaned_document_list.append(extracted_doc_dict)

        return cleaned_document_list
    
    def initialize_document_normalizer(self, extracted_doc: list):
        # first, clean the extracted document structure
        cleaned_document = self.clean_document_structure(extracted_doc)
        # second, normalize the document text
        for cleaned_doc_dict in cleaned_document:
            cleaned_doc_dict["text"] = self.normalize_text(cleaned_doc_dict["text"])
        return cleaned_document
