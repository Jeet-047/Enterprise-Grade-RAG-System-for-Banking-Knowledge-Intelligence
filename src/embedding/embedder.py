import sys
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from src.observability.logger import logging
from src.exception import MyException
from src.utils import read_yaml_file
from dotenv import load_dotenv

load_dotenv()

class NVIDIAEmbedder:
    """
    An wrapper class that provides a consistent interface for embedding operations using NVIDIA's embedding service.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        # Initialize the config for embedder requirements
        self.config = read_yaml_file(config_path)

        emb_cfg = self.config.get("embedder", {})
        self.model_name = emb_cfg["model"]
        self._embedder = None
    
    def get_embedder(self) -> NVIDIAEmbeddings:
        """Create (once) and return the NVIDIA embedding model."""
        if self._embedder is None:
            try:
                logging.info("Initializing the NVIDIA embedder.")
                self._embedder = NVIDIAEmbeddings(
                    model=self.model_name,
                    truncate="NONE"
                )
            except Exception as e:
                raise MyException(e, sys)
        return self._embedder
