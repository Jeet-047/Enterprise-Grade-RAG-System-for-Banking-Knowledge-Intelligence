import re
from bs4 import BeautifulSoup
from src.observability.logger import logging
from src.exception import MyException
import os, sys

_RE_SPACE_TAB = re.compile(r"[ \t]+")
_RE_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")
_RE_SPACE_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")
_RE_MULTI_SPACE = re.compile(r" {2,}")

class DocumentNormalizationAndCleaning:
    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _RE_SPACE_TAB.sub(" ", text)
        text = _RE_BLANK_LINES.sub("\n\n", text)
        text = _RE_SPACE_NEWLINE.sub("\n", text)
        text = _RE_MULTI_SPACE.sub(" ", text)
        return text.strip()
    
    def clean_document_structure(self, extracted_doc: list) -> list:
        cleaned_document_list = []

        for extracted_doc_dict in extracted_doc:
            if 'text' not in extracted_doc_dict or 'metadata' not in extracted_doc_dict:
                raise MyException("Each input dictionary must contain 'text' and 'metadata' keys.", sys)
            if 'doc_type' not in extracted_doc_dict['metadata']:
                raise MyException("Metadata must contain 'doc_type' key.", sys)

            raw_text = extracted_doc_dict['text']
            doc_type = extracted_doc_dict['metadata']['doc_type']

            logging.info(f"Cleaning document of type: {doc_type}")
            logging.info(f"Original text length: {len(raw_text)}")

            if doc_type == 'web':
                logging.info("Applying Web specific cleaning with BeautifulSoup...")
                try:
                    soup = BeautifulSoup(raw_text, 'lxml')
                except Exception:
                    logging.warning("lxml parser not available. Falling back to html.parser.")
                    soup = BeautifulSoup(raw_text, 'html.parser')
                for script_or_style in soup(['script', 'style']):
                    script_or_style.extract()
                cleaned_text = soup.get_text()
            elif doc_type == 'csv':
                logging.info("Applying CSV specific cleaning...")
                cleaned_text = _RE_SPACE_TAB.sub(" ", raw_text)
                cleaned_text = _RE_BLANK_LINES.sub("\n\n", cleaned_text)
                cleaned_text = cleaned_text.strip()
            else:
                logging.info(f"No specific structural cleaning for {doc_type}. Applying general text normalization.")
                cleaned_text = raw_text

            extracted_doc_dict['text'] = cleaned_text
            cleaned_document_list.append(extracted_doc_dict)

        return cleaned_document_list
    
    def initialize_document_normalizer(self, extracted_doc: list):
        cleaned_document = self.clean_document_structure(extracted_doc)
        return [{**doc, "text": self.normalize_text(doc["text"])} for doc in cleaned_document]
