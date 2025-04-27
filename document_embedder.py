"""
ChromaDB Document Embedder

This module provides functionality to load documents from various file formats
and embed them into a ChromaDB collection using specified embedding models.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Type

import chromadb
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    JSONLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chroma_embedder")


class DocumentEmbedder:
    """Manages document loading and embedding into ChromaDB."""

    LOADER_MAP: Dict[str, Type] = {
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".json": JSONLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }

    def __init__(
            self,
            collection_name: str,
            openai_api_key: Optional[str] = None,
            embedding_model_name: str = "text-embedding-3-small",
            chroma_host: str = "127.0.0.1",
            chroma_port: int = 8000,
            chroma_ssl: bool = False,
            chroma_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the DocumentEmbedder.

        Args:
            collection_name: Name of the ChromaDB collection
            openai_api_key: OpenAI API key for embeddings (will use environment variable if None)
            embedding_model_name: OpenAI embedding model name
            chroma_host: ChromaDB host address (IP or domain)
            chroma_port: ChromaDB port (default: 8000)
            chroma_ssl: Whether to use HTTPS for ChromaDB connection
            chroma_headers: Optional headers for ChromaDB HTTP client

        Returns:
            None
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Initialize OpenAI embedding model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            dimensions=1536,  # text-embedding-3-small has 1536 dimensions
        )

        protocol = "https" if chroma_ssl else "http"
        logger.info(f"Connecting to ChromaDB at {protocol}://{chroma_host}:{chroma_port}")

        # Set up ChromaDB HTTP client
        try:
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                ssl=chroma_ssl,
                headers=chroma_headers,
            )

            self.collection = self.client.get_collection(self.collection_name)

            logger.info(
                f"Initialized DocumentEmbedder with collection '{collection_name}' "
                f"and OpenAI embedding model '{embedding_model_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise ConnectionError(f"Could not connect to ChromaDB: {str(e)}")

    def _get_loader_for_file(self, file_path: Union[str, Path]) -> Optional[Callable]:
        """
        Get the appropriate document loader for the given file type.

        Args:
            file_path: Path to the file

        Returns:
            A loader instance for the file type or None if unsupported
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        if file_extension in self.LOADER_MAP:
            logger.debug(f"Using {self.LOADER_MAP[file_extension].__name__} for {file_path}")
            return self.LOADER_MAP[file_extension](str(file_path))

        logger.warning(f"Unsupported file type: {file_extension} for file {file_path}")
        return None

    def load_directory(
            self,
            directory_path: Union[str, Path],
            glob_pattern: str = "**/*.*",
            recursive: bool = True,
            use_multithreading: bool = True,
            max_threads: int = 4,
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to the directory containing documents
            glob_pattern: Pattern to match files
            recursive: Whether to search recursively
            use_multithreading: Whether to use multithreading for loading
            max_threads: Maximum number of threads to use

        Returns:
            List of loaded documents
        """
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory does not exist: {directory_path}")
            raise ValueError(f"Directory does not exist: {directory_path}")

        all_documents = []
        supported_extensions = tuple(self.LOADER_MAP.keys())

        matched_files = [
            file for file in directory_path.glob(glob_pattern)
            if file.is_file() and file.suffix.lower() in supported_extensions
        ]

        if not matched_files:
            logger.warning(f"No supported files found in {directory_path}")
            return all_documents

        logger.info(f"Found {len(matched_files)} supported files in {directory_path}")

        for file_path in matched_files:
            try:
                loader = self._get_loader_for_file(file_path)
                if loader:
                    documents = loader.load()
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")

        return all_documents

    def embed_documents(
            self,
            documents: List[Document],
            batch_size: int = 100,
            overwrite: bool = False,
    ) -> bool:
        """
        Embed documents into ChromaDB collection.

        Args:
            documents: List of documents to embed
            batch_size: Number of documents to process in each batch
            overwrite: Whether to clear an existing collection before adding

        Returns:
            Boolean indicating success
        """
        if not documents:
            logger.warning("No documents to embed")
            return False

        if overwrite:
            logger.info(f"Clearing collection '{self.collection_name}'")
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(self.collection_name)
            except Exception as e:
                logger.error(f"Error clearing collection: {str(e)}")
                return False

        total_docs = len(documents)
        logger.info(f"Embedding {total_docs} documents in batches of {batch_size}")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]

            try:
                ids = [f"doc_{i + idx}" for idx in range(len(batch))]
                texts = [doc.page_content for doc in batch]

                embeddings = []
                for text in texts:
                    embedding = self.embeddings.embed_query(text)
                    embeddings.append(embedding)

                metadatas = []
                for doc in batch:
                    metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)
                    metadatas.append(metadata)

                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )

                logger.info(
                    f"Embedded batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} documents)")

            except Exception as e:
                logger.error(f"Error embedding batch: {str(e)}")
                continue

        logger.info(f"Successfully embedded {total_docs} documents into collection '{self.collection_name}'")
        return True

    def process_directory(
            self,
            directory_path: Union[str, Path],
            glob_pattern: str = "**/*.*",
            recursive: bool = True,
            overwrite: bool = False,
            batch_size: int = 100,
    ) -> bool:
        """
        Process all documents in a directory and embed them into ChromaDB.

        Args:
            directory_path: Path to the directory containing documents
            glob_pattern: Pattern to match files
            recursive: Whether to search recursively
            overwrite: Whether to clear an existing collection before adding
            batch_size: Number of documents to process in each batch

        Returns:
            Boolean indicating success
        """
        try:
            # Load documents
            documents = self.load_directory(
                directory_path=directory_path,
                glob_pattern=glob_pattern,
                recursive=recursive,
            )

            if not documents:
                return False

            # Embed documents
            success = self.embed_documents(
                documents=documents,
                batch_size=batch_size,
                overwrite=overwrite,
            )

            return success
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return False


def main() -> None:
    """Main function to demonstrate usage."""
    global success
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Embed documents into ChromaDB using OpenAI embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "directory",
        help="Directory containing documents to embed"
    )

    # Collection arguments
    collection_group = parser.add_argument_group("Collection Configuration")
    collection_group.add_argument(
        "-c", "--collection",
        default="documents",
        help="Name of the ChromaDB collection"
    )
    collection_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing collection"
    )

    # ChromaDB connection arguments
    chroma_group = parser.add_argument_group("ChromaDB Connection")
    chroma_group.add_argument(
        "--host",
        default="127.0.0.1",
        help="ChromaDB host IP address or domain"
    )
    chroma_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="ChromaDB port"
    )
    chroma_group.add_argument(
        "--ssl",
        action="store_true",
        help="Use HTTPS for ChromaDB connection"
    )
    chroma_group.add_argument(
        "--auth-token",
        help="Authorization token for ChromaDB (will be added as a Bearer token)"
    )
    chroma_group.add_argument(
        "--auth-type",
        choices=["Bearer", "Basic", "None"],
        default="None",
        help="Authentication type to use with auth-token"
    )

    # OpenAI embedding arguments
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "-m", "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name"
    )
    embedding_group.add_argument(
        "-k", "--api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)"
    )

    # Processing arguments
    processing_group = parser.add_argument_group("Document Processing")
    processing_group.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding documents"
    )
    processing_group.add_argument(
        "-p", "--pattern",
        default="**/*.*",
        help="Glob pattern for matching files"
    )
    processing_group.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search directory recursively"
    )
    processing_group.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Don't search directory recursively"
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    log_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all but warning and error logs"
    )

    args = parser.parse_args()

    # Configure logging based on arguments
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)

    # Set up authentication headers if provided
    headers = None
    if args.auth_token:
        headers = {}
        if args.auth_type == "Bearer":
            headers["Authorization"] = f"Bearer {args.auth_token}"
        elif args.auth_type == "Basic":
            headers["Authorization"] = f"Basic {args.auth_token}"
        else:
            headers["Authorization"] = args.auth_token

    directory_path = Path(args.directory)
    if not directory_path.exists() or not directory_path.is_dir():
        logger.error(f"Directory does not exist: {args.directory}")
        return

    # Create embedder
    try:
        embedder = DocumentEmbedder(
            collection_name=args.collection,
            openai_api_key=args.api_key,
            embedding_model_name=args.model,
            chroma_host=args.host,
            chroma_port=args.port,
            chroma_ssl=args.ssl,
            chroma_headers=headers,
        )

        # Process directory
        success = embedder.process_directory(
            directory_path=args.directory,
            glob_pattern=args.pattern,
            recursive=args.recursive,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        )

        if success:
            logger.info("✅ Document embedding completed successfully")
        else:
            logger.error("❌ Document embedding failed")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())

    if success:
        logger.info("✅ Document embedding completed successfully")
    else:
        logger.error("❌ Document embedding failed")


if __name__ == "__main__":
    main()
