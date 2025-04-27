# ChromaDB Document Embedder

A streamlined Python utility for embedding document collections into ChromaDB using OpenAI's embedding models. This tool
bridges the gap between unstructured document repositories and vector-based semantic search capabilities.

## Core Functionality

The ChromaDB Document Embedder provides an end-to-end pipeline for:

1. **Document Loading**: Processes multiple file formats from directory structures
2. **Semantic Embedding**: Generates vector representations using OpenAI embedding models
3. **Vector Database Integration**: Stores document vectors in a ChromaDB collection for retrieval
4. **Remote Connection Support**: Works with both local and remote ChromaDB instances

## Features

- **Multi-Format Support**: Processes PDF, TXT, and CSV files
- **Batched Processing**: Handles large document collections with controlled batch sizes
- **Remote ChromaDB Integration**: Connects to HTTP-based ChromaDB endpoints
- **Authentication Support**: Bearer, Basic, and custom authentication headers
- **Comprehensive CLI**: Exposes all configuration options through a command-line interface
- **Robust Error Handling**: Gracefully manages connection and processing failures
- **Configurable Logging**: Adjustable verbosity for operational needs

## Installation

```bash
# Clone the repository
git clone https://github.com/poacosta/chromadb-document-embedder.git
cd chromadb-document-embedder
```

### Requirements & Dependencies

- Python 3.12+
- openai
- chromadb
- langchain
- langchain-openai

### Environment Configuration

Before using the embedder, you must set up your OpenAI API key:

```bash
# Set OpenAI API key as environment variable (required)
export OPENAI_API_KEY=your-openai-api-key
```

## Usage

### Command-Line Interface

The most straightforward way to use the embedder is through its CLI:

```bash
python document_embedder.py ./docs \
  --collection "knowledge_base" \
  --host "192.144.1.144" \
  --port 8000 \
  --model "text-embedding-3-small" \
  --batch-size 50
```

#### Key Parameters

- `directory`: Path containing documents to process
- `--collection`, `-c`: ChromaDB collection name
- `--host`: ChromaDB host IP address or domain name
- `--port`: ChromaDB server port (default: 8000)
- `--model`, `-m`: OpenAI embedding model name
- `--batch-size`, `-b`: Number of documents to process in each batch
- `--pattern`, `-p`: Glob pattern for matching files (e.g., "**/*.pdf")
- `--overwrite`: Replace an existing collection if it exists
- `--ssl`: Use HTTPS for ChromaDB connection
- `--verbose`, `-v`: Enable detailed logging
- `--quiet`, `-q`: Suppress all but warning and error logs

### Programmatic Usage

The embedder can also be imported and used programmatically:

```python
from document_embedder import DocumentEmbedder

# Initialize with connection details
embedder = DocumentEmbedder(
    collection_name="my_knowledge_base",
    embedding_model_name="text-embedding-3-small",
    chroma_host="192.144.1.144",
    chroma_port=8000
)

# Process a directory of documents
embedder.process_directory(
    directory_path="./data/documents",
    glob_pattern="**/*.pdf",
    batch_size=50,
    overwrite=False
)
```

## Technical Implementation

The embedder implements a three-phase process:

1. **Document Discovery & Loading**
    - Traverse directory structures to locate supported files
    - Maps file extensions to appropriate LangChain document loaders
    - Extracts text content and metadata from documents

2. **Vector Embedding Generation**
    - Connects to OpenAI's API to generate embeddings
    - Processes documents in batches to manage memory and API load
    - Handles serialization of complex metadata structures

3. **ChromaDB Integration**
    - Establishes connection to ChromaDB HTTP endpoint
    - Creates or retrieves the specified collection
    - Stores document texts, metadata, and embeddings

The architectural design prioritizes:

- Fault tolerance through comprehensive exception handling
- Scalability via batched processing
- Configurability through parameterization
- Usability through CLI and programmatic interfaces

## Supported File Types

| Extension | Loader      |
|-----------|-------------|
| .pdf      | PyPDFLoader |
| .txt      | TextLoader  |
| .csv      | CSVLoader   |

## Limitations

- While multithreading support is defined in the interface, the current implementation processes files sequentially
- OpenAI API rate limits may affect processing speed for large document collections
- Embedding dimensions are fixed in 1536 for text-embedding-3-small

## License

[MIT License](LICENSE)
