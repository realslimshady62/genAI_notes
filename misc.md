1. tell what dataclass decorator is?

2. whats the purpose of below code in src/data_ingestion/__init__.py

```python
from .loader import (
DocumentLoader,
PDFDocumentLoader
)

__all__ = [
    "DocumentLoader",
    "PDFDocumentLoader"
]
```

- this imports specific classes and functions from the loader.py file into the module namespace. without the first code block, you will write from src.data_ingestion.loader import DocumentLoader. but sincy you defined the fucntion in init, you can use from src.data_ingestion import DocumentLoader.

- __all__ is a special python list that defines the public API of the module. when someone uses ```from data_ingestion import *``` only the items listed in __all__ would be imported, and nothing else. it acts as a document for what the module intentionally exposes to external code.

3. diff in python dict and json?
dict is the actual object in the memory while json is a string used mainly for data exchange. you got a json dict, to load it into memory, you will use json.loads('{"a":12}')
