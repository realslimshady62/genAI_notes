```python
"""
Data ingestion module for handling multiple document types.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import pandas as pd
# from langchain. import Document
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""
    source: str
    source_type: str  # pdf, html, json, csv, text, etc.
    title: Optional[str] = None
    date_created: Optional[str] = None
    author: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "date_created": self.date_created,
            "author": self.author,
            "version": self.version,
            "tags": self.tags or [],
        }


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents and return as list of Document objects."""
        pass


class PDFDocumentLoader(DocumentLoader):
    """Loader for PDF documents."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.info(f"Initializing PDF loader for {file_path}")
    
    def load(self) -> List[Document]:
        """Load PDF and return documents."""
        try:
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF: {self.file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {self.file_path}: {str(e)}")
            return []


class TextDocumentLoader(DocumentLoader):
    """Loader for plain text documents."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.info(f"Initializing text loader for {file_path}")
    
    def load(self) -> List[Document]:
        """Load text file and return document."""
        try:
            loader = TextLoader(self.file_path)
            documents = loader.load()
            logger.info(f"Loaded text document: {self.file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {self.file_path}: {str(e)}")
            return []


class CSVDocumentLoader(DocumentLoader):
    """Loader for CSV files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.info(f"Initializing CSV loader for {file_path}")
    
    def load(self) -> List[Document]:
        """Load CSV and convert to documents."""
        try:
            df = pd.read_csv(self.file_path)
            documents = []
            
            for idx, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=content,
                    metadata={"source": self.file_path, "row_number": idx}
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} rows from CSV: {self.file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading CSV {self.file_path}: {str(e)}")
            return []


class JSONDocumentLoader(DocumentLoader):
    """Loader for JSON files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.info(f"Initializing JSON loader for {file_path}")
    
    def load(self) -> List[Document]:
        """Load JSON and convert to documents."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            documents = []
            if isinstance(data, list):
                for item in data:
                    content = json.dumps(item, indent=2)
                    doc = Document(
                        page_content=content,
                        metadata={"source": self.file_path}
                    )
                    documents.append(doc)
            else:
                content = json.dumps(data, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={"source": self.file_path}
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} items from JSON: {self.file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading JSON {self.file_path}: {str(e)}")
            return []


class HTMLDocumentLoader(DocumentLoader):
    """Loader for HTML documents and web pages."""
    
    def __init__(self, url_or_path: str, is_url: bool = False):
        self.url_or_path = url_or_path
        self.is_url = is_url
        logger.info(f"Initializing HTML loader for {url_or_path}")
    
    def load(self) -> List[Document]:
        """Load HTML and extract text content."""
        try:
            if self.is_url:
                response = requests.get(self.url_or_path, timeout=10)
                response.raise_for_status()
                html_content = response.text
            else:
                with open(self.url_or_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator="\n")
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
            
            doc = Document(
                page_content=text,
                metadata={"source": self.url_or_path, "source_type": "html"}
            )
            
            logger.info(f"Loaded HTML document: {self.url_or_path}")
            return [doc]
        except Exception as e:
            logger.error(f"Error loading HTML {self.url_or_path}: {str(e)}")
            return []


class DocumentIngestor:
    """Main class for ingesting documents from various sources."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"Initialized DocumentIngestor with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on file type."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == ".pdf":
            loader = PDFDocumentLoader(str(file_path))
        elif file_ext == ".txt":
            loader = TextDocumentLoader(str(file_path))
        elif file_ext == ".csv":
            loader = CSVDocumentLoader(str(file_path))
        elif file_ext == ".json":
            loader = JSONDocumentLoader(str(file_path))
        elif file_ext in [".html", ".htm"]:
            loader = HTMLDocumentLoader(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []
        
        return loader.load()
    
    def load_directory(self, directory_path: str, file_pattern: str = "**/*") -> List[Document]:
        """Load all documents from a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        documents = []
        for file_path in directory_path.glob(file_pattern):
            if file_path.is_file():
                docs = self.load_document(str(file_path))
                documents.extend(docs)
        
        logger.info(f"Loaded {len(documents)} documents from directory: {directory_path}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def ingest(self, source_path: str, is_directory: bool = False) -> List[Document]:
        """Main ingestion pipeline."""
        logger.info(f"Starting ingestion of {source_path}")
        
        if is_directory:
            documents = self.load_directory(source_path)
        else:
            documents = self.load_document(source_path)
        
        if not documents:
            logger.warning(f"No documents loaded from {source_path}")
            return []
        
        chunked_docs = self.chunk_documents(documents)
        logger.info(f"Ingestion complete: {len(chunked_docs)} chunks ready")
        return chunked_docs

```


from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
    DirectoryLoader
)

All these loaders are designed to ingest different file types (PDF, text, JSON, CSV, directories of files, etc.) and convert their contents into a unified format called a Document (or a list of Document objects). This makes it easier for downstream processing, such as chunking, embedding, or querying with LLMs.

# can you explain this with example , whereas LangChain's generic loaders are more general-purpose and less customizable.

Great question! Let me explain with concrete examples showing how LangChain's generic loaders differ from the custom ones in your code.

### Example 1: CSV Loading

**LangChain's `CSVLoader` (Generic):**
```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="products.csv")
documents = loader.load()
```

Output for a CSV with columns `[id, name, price]`:
```
Document(page_content="id: 1\nname: Laptop\nprice: 999", metadata={...})
Document(page_content="id: 2\nname: Mouse\nprice: 25", metadata={...})
```

**Problems with LangChain's approach:**
- Fixed formatting (newline-separated, no control over separators)
- Generic metadata—no row-specific tracking
- Can't easily add custom preprocessing

---

**Your Custom `CSVDocumentLoader`:**
```python
loader = CSVDocumentLoader("products.csv")
documents = loader.load()
```

Output:
```
Document(page_content="id: 1 | name: Laptop | price: 999", metadata={"source": "products.csv", "row_number": 0})
Document(page_content="id: 2 | name: Mouse | price: 25", metadata={"source": "products.csv", "row_number": 1})
```

**Advantages:**
- Customized formatting (pipe-separated for easier parsing)
- Row number tracking in metadata (useful for tracing which row produced a result)
- Full control over the format

---

### Example 2: JSON Loading

**LangChain's `JSONLoader` (Generic, requires additional parameters):**
```python
from langchain_community.document_loaders import JSONLoader

# Must specify the JSON path to extract from (complex configuration)
loader = JSONLoader(
    file_path="company_info.json",
    jq_schema=".[]",  # Extracts array items
    text_content_key="description"  # Must tell it which key has the text
)
documents = loader.load()
```

This is **inflexible** because:
- You must pre-know the JSON structure
- Requires understanding `jq` syntax
- Only works if there's a specific key containing text

---

**Your Custom `JSONDocumentLoader`:**
```python
loader = JSONDocumentLoader("company_info.json")
documents = loader.load()
```

Output (handles any JSON structure):
```
If JSON is:
{
    "company": "Acme Corp",
    "employees": 50,
    "founded": 2020
}

Creates:
Document(
    page_content='{\n  "company": "Acme Corp",\n  "employees": 50,\n  "founded": 2020\n}',
    metadata={"source": "company_info.json"}
)
```

**Advantages:**
- Works with ANY JSON structure (no configuration needed)
- Pretty-prints JSON for readability
- Handles both objects and arrays automatically

---

### Example 3: Real-World Customization Scenario

Let's say you want to **skip rows with missing values** in your CSV. Here's what you'd need to do:

**With LangChain's `CSVLoader`:**
```python
# NOT POSSIBLE - No built-in filtering
# You'd have to:
# 1. Load documents
# 2. Manually filter them after
loader = CSVLoader("products.csv")
documents = loader.load()
documents = [doc for doc in documents if "None" not in doc.page_content]  # Hacky!
```

**With Your Custom `CSVDocumentLoader`:**
```python
# EASY - Add logic directly in the loader
class CSVDocumentLoader(DocumentLoader):
    def load(self) -> List[Document]:
        try:
            df = pd.read_csv(self.file_path)
            df = df.dropna()  # Skip rows with missing values - BUILT IN!
            documents = []
            
            for idx, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=content,
                    metadata={"source": self.file_path, "row_number": idx}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error loading CSV {self.file_path}: {str(e)}")
            return []
```

---

### Summary Table:

| Feature | LangChain Loaders | Your Custom Loaders |
|---------|------------------|-------------------|
| **Flexibility** | Fixed behavior | Fully customizable |
| **Metadata** | Generic | Specific (e.g., row_number) |
| **Unified Interface** | Different interfaces per loader | Consistent `DocumentLoader` interface |
| **Preprocessing** | Requires post-processing | Built into the loader |
| **Integration with DocumentMetadata** | Not designed for it | Easy to integrate |
| **Learning Curve** | Simple for basic use | Requires more code but much more control |

Your custom approach is **better for a RAG system** because you need precise control over how documents are formatted and metadata is tracked for accurate retrieval and attribution.