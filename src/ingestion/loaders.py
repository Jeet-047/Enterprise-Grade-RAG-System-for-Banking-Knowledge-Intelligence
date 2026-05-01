from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader , WebBaseLoader
from src.observability.logger import logging
from src.exception import MyException
import sys
import os

class DocumentLoader:
    def __init__(self):
        pass
    
    def load_document(self, document_path):
        """
        Loads a document from a given path or URL using the appropriate Langchain loader.
        Args:
            document_path (str): The path to the document file or a URL.
        Returns:
            list: A list of loaded documents.
        Raises:
            ValueError: If the document type is unsupported or the path is invalid.
        """
        logging.info(f"Attempting to load document from: {document_path}")
        try:
            # If this is a local file, do some pre-checks to provide better diagnostics
            if not document_path.startswith(('http://', 'https://')):
                if not os.path.exists(document_path):
                    raise MyException(f"File does not exist: {document_path}", sys)
                if not os.path.isfile(document_path):
                    raise MyException(f"Path exists but is not a regular file: {document_path}", sys)
                file_size = os.path.getsize(document_path)
                if file_size == 0:
                    raise MyException(f"File is empty: {document_path}", sys)
                if file_size > 50 * 1024 * 1024:  # 50 MB
                    logging.warning("File %s is large (size=%d bytes). This may take longer to process.", document_path, file_size)

            # Choose an appropriate loader based on file extension or URL
            if document_path.startswith(('http://', 'https://')):
                loader = WebBaseLoader(document_path)
            elif document_path.endswith('.pdf'):
                loader = PyPDFLoader(document_path)
            elif document_path.endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(document_path)
            elif document_path.endswith('.csv'):
                loader = CSVLoader(document_path)
            else:
                raise MyException(f"Unsupported document type: {document_path}. Please provide a PDF, DOCX, CSV file or a URL.", sys)

            # If we haven't already loaded via the TXT fallback, call loader.load()
            if 'document' not in locals():
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
            # Log full exception and traceback to help diagnose file-specific failures
            logging.exception("Error loading document %s: %s", document_path, e)
            raise MyException(f"Could not load document {document_path}. Error: {type(e).__name__}: {e}", sys)

