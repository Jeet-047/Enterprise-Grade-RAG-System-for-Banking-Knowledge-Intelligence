from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, WebBaseLoader
from src.observability.logger import logging
from src.exception import MyException
import sys
import os

_LOADER_MAP = {
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.doc': Docx2txtLoader,
    '.csv': CSVLoader,
}

class DocumentLoader:
    def load_document(self, document_path: str):
        logging.info(f"Attempting to load document from: {document_path}")

        if not document_path.startswith(('http://', 'https://')):
            if not os.path.exists(document_path):
                raise MyException(f"File does not exist: {document_path}", sys)
            if not os.path.isfile(document_path):
                raise MyException(f"Path exists but is not a regular file: {document_path}", sys)

            file_size = os.path.getsize(document_path)
            if file_size == 0:
                raise MyException(f"File is empty: {document_path}", sys)
            if file_size > 50 * 1024 * 1024:
                logging.warning("File %s is large (%d bytes). This may take longer to process.", document_path, file_size)

        if document_path.startswith(('http://', 'https://')):
            loader = WebBaseLoader(document_path)
        else:
            ext = os.path.splitext(document_path)[1].lower()
            loader_cls = _LOADER_MAP.get(ext)
            if loader_cls is None:
                raise MyException(f"Unsupported document type: {document_path}. Please provide a PDF, DOCX, CSV file or a URL.", sys)
            loader = loader_cls(document_path)

        try:
            document = loader.load()
            logging.info(f"Successfully loaded {len(document)} pages/parts from {document_path}")
            return document
        except MyException:
            raise
        except FileNotFoundError as fnf:
            logging.exception("File not found when loading %s: %s", document_path, fnf)
            raise MyException(f"Could not load document {document_path}. File not found.", sys)
        except PermissionError as pe:
            logging.exception("Permission denied when loading %s: %s", document_path, pe)
            raise MyException(f"Could not load document {document_path}. Permission denied.", sys)
        except Exception as e:
            logging.exception("Error loading document %s: %s", document_path, e)
            raise MyException(f"Could not load document {document_path}. Error: {type(e).__name__}: {e}", sys)

