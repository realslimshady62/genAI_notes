In the context of RAG (Retrieval-Augmented Generation), **chunking** is the process of breaking down large documents into smaller, manageable pieces before embedding them. The strategy you choose directly impacts retrieval accuracy, context preservation, and the quality of the final answer.

Here are the main chunking strategies, from basic to advanced:

### 1. Fixed-Size Chunking
The simplest method. You split text into chunks of a predetermined number of tokens or characters (e.g., 512 tokens). Overlap is often added to maintain context across boundaries.

- **How it works:** `chunk 1: tokens 0-500, chunk 2: tokens 250-750, chunk 3: tokens 500-1000` (with 250-token overlap).
- **Pros:** Computationally cheap, predictable, works for any text.
- **Cons:** Ignores semantic boundaries; can split sentences, paragraphs, or ideas in half.
- **Best for:** Quick prototypes, homogeneous text (logs, code), or when using models with strict input limits.

### 2. Recursive Character Chunking
An improvement on fixed-size. It attempts to split text hierarchically by trying different separators in order (e.g., `\n\n` → `\n` → `.` → ` ` → `""`).

- **How it works:** It first tries to split by double newlines (paragraphs). If chunks are still too large, it falls back to single newlines, then sentences, then words.
- **Pros:** Preserves natural boundaries better than pure fixed-size; avoids mid-word cuts.
- **Cons:** Still not fully semantic; can still split a thought if a paragraph is too long.
- **Best for:** General-purpose text like articles, emails, or markdown files.

### 3. Semantic Chunking
Uses embedding similarity to find natural breakpoints. The model looks for sentences where the meaning changes significantly.

- **How it works:**
  1. Embed each sentence.
  2. Compute cosine similarity between consecutive sentences.
  3. When similarity drops below a threshold, start a new chunk.
- **Pros:** Highly coherent chunks; each chunk contains a single "idea" or topic.
- **Cons:** Computationally expensive (requires embedding every sentence); threshold tuning is critical.
- **Best for:** Complex documents, academic papers, legal contracts, or any text where topic shifts are important.

### 4. Document-Based Chunking (Structure-Aware)
Uses the inherent structure of the document (headers, sections, lists, code blocks) to define chunk boundaries.

- **How it works:** Parse markdown, HTML, or PDF. Each header becomes a new chunk; include the header text as metadata for all sub-chunks.
- **Pros:** Preserves hierarchical context; retrieves both section title and content together.
- **Cons:** Requires good document structure; fails on plain text.
- **Best for:** Markdown docs, wikis, technical manuals, legal briefs, and well-structured HTML.

### 5. Sentence-Level Chunking
Keeps each sentence as its own chunk. Usually combined with a sliding window of surrounding sentences for context.

- **How it works:** Split by punctuation (`.`, `!`, `?`). Each chunk = 1 sentence. During retrieval, you also fetch ±N neighboring sentences.
- **Pros:** Very precise retrieval; great for fact lookup.
- **Cons:** Lacks broader context; requires a re-ranking or context expansion step.
- **Best for:** QA over dense facts (e.g., "What year did X happen?"), legal or medical Q&A.

### 6. Proposition-Based Chunking
The most advanced. It uses an LLM to break text into atomic, self-contained statements called "propositions" (e.g., "The Eiffel Tower is in Paris. It was built in 1889." becomes two chunks).

- **How it works:** Prompt an LLM (e.g., GPT-4) to rewrite each clause as a minimal, standalone fact.
- **Pros:** Extremely precise retrieval; eliminates dependency on context; reduces hallucination.
- **Cons:** High latency and cost; LLM-dependent; can lose narrative flow.
- **Best for:** High-stakes domains (medical diagnosis, legal reasoning), knowledge graphs, or when you need traceable facts.

### 7. Metadata-Enriched Chunking
Not a pure splitting strategy, but a crucial augmentation. You attach metadata (source, date, author, section title, page number) to each chunk *before* embedding.

- **How it works:** Chunk via any method above, then prepend metadata: `[Source: Annual Report 2024, Section: Revenue, Page 3] Chunk text...`
- **Pros:** Allows filtered retrieval (e.g., "only from documents after 2023"); improves relevance.
- **Cons:** Increases token count; metadata must be extracted accurately.
- **Best for:** Multi-document RAG, time-sensitive queries, compliance-heavy applications.

### 8. Small-to-Big (Parent-Child) Chunking
A retrieval strategy that pairs small chunks for search with larger chunks for LLM context.

- **How it works:**
  - **Child chunks** (e.g., sentence or 128 tokens) are embedded and used for similarity search.
  - **Parent chunks** (e.g., paragraph or 512 tokens) containing the child are returned to the LLM.
- **Pros:** High retrieval precision (small chunks) + rich context (big chunks).
- **Cons:** More storage and logic overhead.
- **Best for:** Any RAG where you need both accuracy and context—very popular in production systems.

## How to choose?

| If you have...                         | Start with...                          |
| -------------------------------------- | -------------------------------------- |
| No time, just need it working          | Recursive character + overlap (20-50%) |
| Well-structured documents (headings)   | Document-based + metadata              |
| Academic papers / topic shifts         | Semantic or Proposition-based          |
| Many short facts (QA over a knowledge base) | Sentence-level + small-to-big     |
| Mission-critical accuracy + budget     | Small-to-big + metadata + re-ranking   |

**Key tuning parameters:**
- **Chunk size:** 128–512 tokens for facts; 512–2048 for narrative.
- **Overlap:** 10–25% of chunk size (helps when a question spans a boundary).
- **Embedding model:** Denser chunks (semantic/proposition) benefit from higher-dimension models (e.g., `text-embedding-3-large`).

Would you like a practical code example (e.g., in Python with LangChain or LlamaIndex) for any of these strategies?