# RAG Pipeline v4.0

**Complete Retrieval-Augmented Generation System with Semantic Search and Context-Aware Generation**

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Installation & Setup](#installation--setup)
5. [Input Specification](#input-specification)
6. [Processing Pipeline](#processing-pipeline)
7. [Output Specification](#output-specification)
8. [Usage Guide](#usage-guide)
9. [Configuration](#configuration)
10. [Search & Retrieval](#search--retrieval)
11. [Context Building](#context-building)
12. [Generation](#generation)
13. [API Reference](#api-reference)
14. [Examples](#examples)
15. [Integration with Ingestion](#integration-with-ingestion)
16. [Troubleshooting](#troubleshooting)

---

## Overview

The **RAG (Retrieval-Augmented Generation) Pipeline** is a production-grade system that combines semantic search with LLM generation to provide accurate, context-grounded answers to user queries.

### Key Features

✓ **Semantic Search**: FAISS-based vector similarity search  
✓ **FAISS ID Mapping**: Stable ID tracking across sessions  
✓ **Metadata Filtering**: Advanced filtering with extra_metadata support  
✓ **Source Deduplication**: Prevents duplicate sources in results  
✓ **Quality Ranking**: Chunks ranked by semantic similarity score  
✓ **Context Building**: Automatic context compilation for LLM consumption  
✓ **Two-Stage Retrieval**: Both FAISS and ChromaDB support  
✓ **Generation Integration**: Works with LFM2.5 and other LLMs  

### Workflow Overview

```
User Query
    ↓
Semantic Embedding
    ↓
Vector Similarity Search (FAISS)
    ↓
Metadata Filtering & Ranking
    ↓
Source Deduplication
    ↓
Context Building
    ↓
Prompt Assembly
    ↓
LLM Generation
    ↓
Structured Response with Sources
```

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────┐
│         USER QUERY (Natural Language)               │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │  QUERY EMBEDDING         │
        │ (BAAI/bge-large-en-v1.5) │
        │ Output: 1024-dim vector  │
        └────────────┬─────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  VECTOR SIMILARITY SEARCH     │
        │  (FAISS IndexFlatIP)          │
        │  - k nearest neighbors        │
        │  - Inner product scores       │
        │  - Returns top-k candidates   │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  FAISS ID → CHUNK ID MAPPING  │
        │  (from index_meta.json)       │
        │  - Resolves chunk identities  │
        │  - Retrieves from docstore    │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  METADATA FILTERING           │
        │  - Apply user filters         │
        │  - Check extra_metadata       │
        │  - Validate constraints       │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  SOURCE DEDUPLICATION        │
        │  - Group by source_document  │
        │  - Keep highest score        │
        │  - Reduce redundancy         │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  SCORE RANKING               │
        │  - Sort by similarity score  │
        │  - Top-k selection           │
        │  - Return RAGResult objects  │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  CONTEXT COMPILATION         │
        │  - Format retrieved chunks   │
        │  - Build context string      │
        │  - Calculate token count     │
        │  - Extract sources           │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  PROMPT BUILDING             │
        │  - Add system message        │
        │  - Insert context            │
        │  - Append user query         │
        │  - Format for LLM            │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────┐
        │  LLM GENERATION              │
        │  (LFM2.5-1.2B-Instruct)      │
        │  - Process prompt            │
        │  - Generate answer           │
        │  - Stream tokens (optional)  │
        └────────────┬──────────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │  STRUCTURED RESPONSE               │
        │  - Answer text                    │
        │  - Retrieved sources              │
        │  - Metadata (tokens, latency)     │
        │  - Confidence scores              │
        └────────────────────────────────────┘
```

### Component Interaction

```
VectorStoreRetriever
    ↓ (reads)
[index.faiss]
[docstore.json]
[index_meta.json]
    ↓
RAGPromptBuilder
    ↓ (formats)
[Context + Query]
    ↓
RAGGenerator
    ↓ (generates)
[LLM Response]
    ↓
RAGPipeline (orchestrates all above)
```

---

## Components

### 1. **RAGResult** (Data Structure)

Represents a single retrieved document/chunk.

**Fields:**
```python
id: str                   # Chunk identifier
content: str              # Chunk text (up to 300 chars in summaries)
score: float              # Similarity score (0.0-1.0)
metadata: Dict[str, Any]  # Full chunk metadata
source_file: str          # Source document path

# Helper method:
to_dict() -> Dict         # Convert to JSON-serializable format
```

**Example:**
```python
result = RAGResult(
    id="chunk_abc123",
    content="Python is a high-level programming language...",
    score=0.92,
    metadata={
        "source_document": "./docs/python.pdf",
        "quality_score": 0.85,
        "entropy_score": 0.79,
        # ... more metadata
    },
    source_file="./docs/python.pdf"
)
```

### 2. **RAGContext** (Data Structure)

Represents compiled context for generation.

**Fields:**
```python
query: str                          # Original user query
retrieved_results: List[RAGResult]  # All retrieved chunks
context_text: str                   # Formatted context for LLM
total_tokens: int                   # Estimated token count
sources: List[str]                  # Unique source file paths

# Helper method:
to_dict() -> Dict                   # Summary information
```

**Example:**
```python
context = RAGContext(
    query="How does Python handle memory?",
    retrieved_results=[result1, result2, result3],
    context_text="[Source 1: python.pdf]\nPython uses....",
    total_tokens=2450,
    sources=["./docs/python.pdf", "./docs/memory.pdf"]
)
```

### 3. **VectorStoreRetriever** (Search Engine)

Performs semantic search across FAISS index.

**Responsibilities:**
- Load FAISS index + docstore + metadata
- Embed user queries
- Search for similar vectors
- Map FAISS IDs to chunk IDs (using index_meta.json)
- Apply metadata filters
- Deduplicate by source
- Return ranked RAGResult objects

**Key Methods:**
- `__init__(store_path, embedding_model, index_type)` - Initialize
- `_load_index()` - Load FAISS or Chroma store
- `search(query, top_k, score_threshold, filters, dedupe_by_source)` - Retrieve chunks
- `_extract_source(metadata)` - Extract source from metadata
- `_matches_filters(metadata, filters)` - Check if metadata matches filters
- `_build_fallback_id_map()` - Rebuild ID mapping if missing
- `get_stats()` - Return retriever statistics

### 4. **RAGPromptBuilder** (Context Formatter)

Compiles retrieved context into LLM-ready prompts.

**Responsibilities:**
- Format retrieved chunks with source citations
- Build context string within token limits
- Compile chat messages
- Estimate token counts
- Create RAGContext objects

**Key Methods:**
- `__init__(max_context_tokens, context_template)` - Initialize
- `build_context(query, results, max_chunks)` - Create RAGContext
- `build_chat_messages(query, context, system_prompt)` - Format for LLM

**Default System Prompt:**
```
You are GAKR AI, a helpful assistant with access to a knowledge base.
Answer questions based on the provided context. If the answer is not in the context,
say "I don't have enough information in my knowledge base." Always cite your sources
using [Source X] notation.
```

### 5. **RAGGenerator** (LLM Interface)

Manages LLM model loading and response generation.

**Responsibilities:**
- Load LFM2.5 model and tokenizer
- Apply chat templates
- Generate responses (streaming or non-streaming)
- Manage GPU/CPU device allocation

**Key Methods:**
- `__init__(model_name, device, max_new_tokens)` - Initialize
- `generate(messages, temperature, top_p, stream)` - Generate response
- `_load_model()` - Load model from HuggingFace

### 6. **RAGPipeline** (Orchestrator)

Coordinates the entire RAG workflow.

**Responsibilities:**
- Initialize all components (retriever, builder, generator)
- Orchestrate query → retrieve → context → generate flow
- Provide high-level query interface

**Key Methods:**
- `__init__(store_path, embedding_model, llm_model, device)` - Initialize
- `query(query, top_k, max_context_chunks, temperature, filters, stream)` - Full pipeline
- `retrieve_context(query, top_k, max_context_chunks, filters)` - Retrieve only
- `query_stream(query, **kwargs)` - Stream results
- `get_stats()` - Return pipeline statistics

---

## Installation & Setup

### Prerequisites

Same as Ingestion Pipeline, plus LLM model:

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Download LLM Model

```bash
# Download LFM2.5 model (one-time)
from transformers import AutoModel, AutoTokenizer

model_name = "AshokGakr/tiny_thinking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Or download during first RAGPipeline initialization
# (automatic if model folder exists)
```

### Quick Setup

```bash
# Navigate to project
cd optimus-prime-rag

# Ensure ingestion completed
ls -la vector_store/
# Should show: index.faiss, docstore.json, index_meta.json

# Initialize RAG pipeline in Python
from rag_pipeline import RAGPipeline
rag = RAGPipeline(store_path="./vector_store")
```

---

## Input Specification

### Query Input

**Format:** Natural language text string

**Examples:**
```python
queries = [
    "What is machine learning?",
    "How does Python handle memory management?",
    "Explain quantum computing basics",
    "What are the best practices for REST API design?"
]
```

### Vector Store Input

**Source:** Output from Ingestion Pipeline

**Required Files:**
- `index.faiss` - FAISS index with embeddings
- `docstore.json` - Chunk content + metadata
- `index_meta.json` - FAISS ID mapping

**Automatic Detection:**
```python
# RAGPipeline auto-detects available index
rag = RAGPipeline(store_path="./vector_store")
# Uses FAISS if available, falls back to Chroma
```

### Configuration Inputs

```python
# Retrieval parameters
top_k = 5                    # Number of chunks to retrieve
score_threshold = 0.5        # Minimum similarity score
max_context_chunks = 3       # Max chunks in context

# Generation parameters
temperature = 0.1            # Sampling temperature (0.0-2.0)
max_new_tokens = 512         # Maximum generation length

# Filtering
filters = {
    "doc_type": "pdf",
    "quality_score": 0.7
}

# Deduplication
dedupe_by_source = True      # Remove duplicate sources
```

---

## Processing Pipeline

### Stage 1: Query Embedding

```
Input: User query string
Output: Query embedding (1024-dim vector)

Process:
1. Tokenize query text
2. Load SentenceTransformer (BAAI/bge-large-en-v1.5)
3. Compute embedding
4. Normalize to unit vector (L2 norm)
```

**Example:**
```python
query = "What is artificial intelligence?"
query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype('float32')
# Shape: (1, 1024)
```

### Stage 2: Vector Similarity Search

```
Input: Query embedding (1024-dim)
Output: Top-k similar chunks with scores

Process:
1. Load FAISS index from disk
2. Search: index.search(query_embedding, k=top_k*5)
3. Returns:
   - faiss_ids: Array of FAISS indices
   - distances: Array of inner-product scores (0.0-1.0)
```

**FAISS Details:**
- **Index Type:** IndexFlatIP (Inner Product)
- **Distance Metric:** Cosine similarity (after normalization)
- **Score Range:** 0.0 (dissimilar) to 1.0 (identical)
- **Top-k:** Retrieved: top_k*5, filtered to top_k

### Stage 3: FAISS ID Resolution

```
Input: FAISS IDs from search
Output: Chunk IDs (actual identifiers)

Process:
1. Load index_meta.json
   {
     "id_map": {
       "0": "chunk_abc123",
       "1": "chunk_def456",
       ...
     }
   }
2. Map each FAISS ID to chunk ID
   faiss_id=0 → chunk_abc123
3. Handle missing mappings (fallback mode)
```

**Importance:** Ensures stable chunk identification across sessions

### Stage 4: Metadata Lookup & Filtering

```
Input: Chunk IDs
Output: RAGResult objects (filtered)

Process:
1. Retrieve chunk data from docstore.json
2. Extract metadata
3. Apply user filters:
   - score_threshold: Skip if score < 0.5
   - filters: Check each constraint
     - Can filter on: quality_score, doc_type, language, etc.
     - Checks both direct metadata and extra_metadata
4. Return matching chunks
```

**Filter Examples:**
```python
# High-quality documents only
filters = {"quality_score": "> 0.8"}

# Specific file types
filters = {"doc_type": "code"}

# Language-specific
filters = {"language": "en"}
```

### Stage 5: Source Deduplication

```
Input: List[RAGResult] (potentially duplicate sources)
Output: List[RAGResult] (unique sources, best ranked)

Process:
1. If dedupe_by_source=False: Skip
2. If dedupe_by_source=True:
   - Group chunks by source_file
   - Within each group: Keep highest score
   - Return merged results sorted by score
```

**Example:**
```
Before dedup:
  - Source: file1.pdf, Score: 0.92
  - Source: file1.pdf, Score: 0.85  ← Remove (lower score)
  - Source: file2.pdf, Score: 0.88

After dedup:
  - Source: file1.pdf, Score: 0.92
  - Source: file2.pdf, Score: 0.88
```

### Stage 6: Result Ranking & Truncation

```
Input: Filtered results
Output: Top-k RAGResult objects

Process:
1. Sort by score (descending)
2. Keep top-k based on:
   - top_k parameter
   - Deduplication
   - Filtering constraints
3. Create RAGResult objects
```

### Stage 7: Context Building

```
Input: List[RAGResult], User query
Output: RAGContext with formatted text

Process:
1. Format each chunk:
   "[Source N: filename.pdf]
    Chunk content text here..."
   
2. Concatenate with limits:
   - Max chunks: max_context_chunks
   - Max tokens: max_context_tokens
   - Rough estimate: 1 token ≈ 4 chars
   
3. Build full context:
   "[Context compiled from retrieved sources]
    
    ---
    User Question: How does...?
    
    Based on the above context, provide..."
   
4. Estimate token count
5. Return RAGContext object
```

**Token Calculation:**
```
Estimated tokens = len(context_text) / 4
(Rough approximation; actual depends on tokenizer)
```

### Stage 8: Prompt Building

```
Input: RAGContext
Output: Chat messages for LLM

Process:
1. Create system message:
   "You are GAKR AI, a helpful assistant..."
   
2. Create user message:
   context.context_text
   
3. Format for LLM chat template:
   [
     {"role": "system", "content": "..."},
     {"role": "user", "content": "..."}
   ]
```

### Stage 9: LLM Generation

```
Input: Chat messages, generation parameters
Output: Generated response text

Process:
1. Apply chat template (ChatML format)
2. Tokenize (max_length=4096)
3. Generate tokens:
   - temperature: Sampling randomness (0.1 = deterministic)
   - top_p: Nucleus sampling (0.1 = narrow distribution)
   - max_new_tokens: Generation limit
   - repetition_penalty: 1.05 (prevent repetition)
   
4. Decode tokens to text
5. Return generation

Optional: Stream tokens in real-time
```

---

## Output Specification

### Query Response (Non-Streaming)

```python
response = {
    'query': 'What is machine learning?',
    'answer': 'Machine learning is a subset of artificial intelligence...',
    'sources': [
        {
            'id': 'chunk_abc123',
            'content': 'Machine learning is a...' (truncated),
            'score': 0.92,
            'source': './docs/ml_guide.pdf',
            'metadata': {...}
        },
        # ... more sources
    ],
    'context': {
        'query': 'What is machine learning?',
        'results_count': 3,
        'context_length': 2450,
        'total_tokens': 2800,
        'sources': ['./docs/ml_guide.pdf', './docs/ai_basics.pdf']
    },
    'metadata': {
        'retrieved_count': 3,
        'context_tokens': 2800,
        'generation_time_ms': 1250,
        'total_time_ms': 1450,
        'model': 'AshokGakr/tiny_thinking'
    }
}
```

### Retrieve-Only Response

```python
context, results = rag.retrieve_context(
    query="What is AI?",
    top_k=3
)

# context: RAGContext object
context.context_text  # Formatted context string
context.sources       # List of source files
context.total_tokens  # Token count

# results: List[RAGResult] objects
for result in results:
    print(result.source_file)
    print(result.score)
    print(result.metadata)
```

### Streaming Response

```python
response_dict = {
    'query': 'What is machine learning?',
    'streamer': <TextIteratorStreamer>,  # Token stream
    'context': {...},
    'sources': [...]
}

# Stream tokens
for token in response_dict['streamer']:
    print(token, end='', flush=True)
```

### Statistics

```python
stats = rag.get_stats()

{
    'retriever': {
        'total_documents': 1250,
        'embedding_dimension': 1024,
        'index_type': 'faiss',
        'embedding_model': 'BAAI/bge-large-en-v1.5'
    },
    'generator': {
        'model': 'AshokGakr/tiny_thinking',
        'device': 'cuda:0'
    }
}
```

---

## Usage Guide

### Basic Query

```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(store_path="./vector_store")

# Query
response = rag.query(
    query="What is machine learning?",
    top_k=3,
    max_context_chunks=3,
    temperature=0.1
)

# Access response
print(response['answer'])
print(f"Sources: {response['sources']}")
print(f"Time: {response['metadata']['total_time_ms']}ms")
```

### Retrieve-Only (No Generation)

```python
# Useful for testing, debugging, or custom generation

context, results = rag.retrieve_context(
    query="How does Python handle memory?",
    top_k=5,
    max_context_chunks=3
)

if context:
    print(context.context_text)
    print(f"Found {len(results)} relevant chunks")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.source_file} (score: {result.score:.2f})")
else:
    print("No relevant chunks found")
```

### Filtered Search

```python
# Search with quality filter
response = rag.query(
    query="Advanced machine learning algorithms",
    top_k=5,
    filters={
        "quality_score": 0.7,
        "doc_type": "pdf"
    },
    dedupe_by_source=True
)
```

### Streaming Response

```python
# Stream generation for real-time display
response = rag.query(
    query="Explain quantum computing",
    stream=True
)

# This returns immediately with a streamer
streamer = response['streamer']

# Stream tokens
for token in streamer:
    print(token, end='', flush=True)

print()
print(f"Sources: {response['sources']}")
```

### Custom Application Integration

```python
from rag_pipeline import RAGPipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
rag = RAGPipeline(store_path="./vector_store")

@app.route("/api/query", methods=["POST"])
def query_endpoint():
    data = request.json
    query = data.get("query")
    
    response = rag.query(query, top_k=5)
    
    return jsonify({
        "answer": response['answer'],
        "sources": response['sources'],
        "metadata": response['metadata']
    })

if __name__ == "__main__":
    app.run(debug=False)
```

---

## Configuration

### Environment Variables

```bash
# Vector store location
export RAG_VECTOR_STORE_PATH=./vector_store

# Embedding model
export RAG_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# LLM model
export RAG_LLM_MODEL=AshokGakr/tiny_thinking

# Device (cuda/cpu)
export RAG_DEVICE=cuda
```

### Program Configuration

```python
rag = RAGPipeline(
    store_path="./vector_store",           # Vector store location
    embedding_model="BAAI/bge-large-en-v1.5",  # Search embedder
    llm_model="AshokGakr/tiny_thinking",   # Generator model
    device="cuda"                           # cuda or cpu
)
```

### Retrieval Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `top_k` | 5 | 1-100 | Number of chunks to retrieve |
| `score_threshold` | 0.5 | 0.0-1.0 | Minimum similarity score |
| `max_context_tokens` | 2048 | 512-8192 | Max context for generation |
| `max_context_chunks` | 3 | 1-10 | Max chunks to include |
| `dedupe_by_source` | False | True/False | Remove duplicate sources |

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 0.1 | 0.0-2.0 | Sampling randomness |
| `top_p` | 0.1 | 0.0-1.0 | Nucleus sampling |
| `max_new_tokens` | 512 | 1-2048 | Max generation length |

---

## Search & Retrieval

### Search Algorithm

**FAISS IndexFlatIP (Inner Product):**
1. Query vector: 1024-dim (normalized)
2. Database vectors: N × 1024 (normalized)
3. Similarity = dot_product(query, db_vector)
4. Range: [0.0, 1.0] after normalization
5. Higher = more similar

### Ranking Strategy

```
1. Cosine similarity score (primary)
2. Quality score bonus (secondary)
3. Semantic coherence (tertiary)

Formula:
final_score = cosine_similarity * (1 + 0.1 * quality_score)
```

### Source Deduplication Logic

```python
def deduplicate_by_source(results: List[RAGResult]) -> List[RAGResult]:
    """Keep highest score from each source"""
    seen = {}
    for result in results:
        source = result.source_file
        if source not in seen:
            seen[source] = result
        elif result.score > seen[source].score:
            seen[source] = result
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)
```

### Filtering Examples

```python
# Filter 1: Quality threshold
filters = {"quality_score": 0.7}
# Only chunks with quality ≥ 0.7

# Filter 2: Document type
filters = {"doc_type": "code"}
# Only code chunks

# Filter 3: Multiple conditions
filters = {
    "quality_score": 0.6,
    "doc_type": "pdf",
    "language": "en"
}
# Must satisfy ALL conditions
```

---

## Context Building

### Context Format

```
[Source 1: docs/guide.pdf]
This is the content of the first retrieved chunk...

[Source 2: docs/reference.pdf]
This is the content of the second retrieved chunk...

---
User Question: How does machine learning work?

Based on the above context, provide a detailed answer:
```

### Token Limits

```python
# Default: 2048 tokens for context + generation
# Conservative estimate: 1 token = 4 characters

max_context_tokens = 2048
approx_chars_limit = 2048 * 4 = 8192 characters

# Chunks included until limit reached
max_context_chunks = 3  # Hard cap (regardless of tokens)
```

---

## Generation

### Chat Template (ChatML)

```
<|im_start|>system
You are GAKR AI, a helpful assistant...
<|im_end|>
<|im_start|>user
Here is the context...

User Question: What is...?
<|im_end|>
<|im_start|>assistant
```

### Generation Parameters

**Temperature = 0.1 (Default)**
- Very deterministic
- Repeatable results
- Best for factual QA

**Temperature = 0.7**
- More diverse
- Creative responses
- Less predictable

**Top-P = 0.1 (Default)**
- Narrow distribution
- Conservative sampling
- Reduces hallucinations

### Output Characteristics

- **Latency**: 1-5 seconds typical
- **Max tokens**: 512 default
- **Streaming**: Optional real-time tokens
- **Memory**: ~4-8 GB for model

---

## API Reference

### RAGPipeline

#### `__init__(store_path, embedding_model, llm_model, device)`

Initialize RAG pipeline.

**Parameters:**
- `store_path` (str): Vector store directory
- `embedding_model` (str): Search embedding model
- `llm_model` (str): Generation LLM model
- `device` (str): 'cuda' or 'cpu'

**Example:**
```python
rag = RAGPipeline(
    store_path="./vector_store",
    embedding_model="BAAI/bge-large-en-v1.5",
    llm_model="AshokGakr/tiny_thinking",
    device="cuda"
)
```

#### `query(query, top_k, max_context_chunks, temperature, filters, stream)`

Execute full RAG pipeline.

**Parameters:**
- `query` (str): User question
- `top_k` (int): Chunks to retrieve (default: 3)
- `max_context_chunks` (int): Max chunks in context (default: 3)
- `temperature` (float): Generation temperature (default: 0.1)
- `filters` (Dict): Metadata filters (default: None)
- `stream` (bool): Stream response (default: False)

**Returns:** Dict with answer, sources, context, metadata

**Example:**
```python
response = rag.query(
    "What is artificial intelligence?",
    top_k=5,
    max_context_chunks=3,
    temperature=0.1,
    filters={"quality_score": 0.7},
    stream=False
)
print(response['answer'])
```

#### `retrieve_context(query, top_k, max_context_chunks, filters)`

Retrieve context only (no generation).

**Parameters:**
- `query` (str): User question
- `top_k` (int): Chunks to retrieve
- `max_context_chunks` (int): Max chunks in context
- `filters` (Dict): Metadata filters

**Returns:** Tuple[RAGContext, List[RAGResult]]

**Example:**
```python
context, results = rag.retrieve_context(
    "How does Python work?",
    top_k=5
)
```

#### `query_stream(query, **kwargs)`

Stream RAG response.

**Parameters:** Same as query()

**Returns:** Generator yielding tokens

**Example:**
```python
for token in rag.query_stream("What is AI?"):
    print(token, end='', flush=True)
```

#### `get_stats()`

Get pipeline statistics.

**Returns:** Dict with component stats

**Example:**
```python
stats = rag.get_stats()
print(stats['retriever']['total_documents'])
```

### VectorStoreRetriever

#### `search(query, top_k, score_threshold, filters, dedupe_by_source)`

Search for relevant chunks.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results
- `score_threshold` (float): Min score (0.0-1.0)
- `filters` (Dict): Metadata filters
- `dedupe_by_source` (bool): Remove duplicate sources

**Returns:** List[RAGResult]

#### `get_stats()`

Get retriever statistics.

**Returns:** Dict with index info

### RAGPromptBuilder

#### `build_context(query, results, max_chunks)`

Build RAGContext from results.

**Parameters:**
- `query` (str): Original query
- `results` (List[RAGResult]): Retrieved chunks
- `max_chunks` (int): Max chunks to include

**Returns:** RAGContext

#### `build_chat_messages(query, context, system_prompt)`

Build messages for LLM.

**Parameters:**
- `query` (str): Original query
- `context` (RAGContext): Compiled context
- `system_prompt` (str): Optional system message

**Returns:** List[Dict] with role/content

### RAGGenerator

#### `generate(messages, temperature, top_p, stream)`

Generate LLM response.

**Parameters:**
- `messages` (List[Dict]): Chat messages
- `temperature` (float): Sampling temperature
- `top_p` (float): Nucleus sampling
- `stream` (bool): Stream output

**Returns:** str or TextIteratorStreamer

---

## Examples

### Example 1: Simple Q&A

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Query
response = rag.query("What is machine learning?")

# Access results
print("Question:", response['query'])
print("Answer:", response['answer'])
print("Sources:", response['metadata'].get('retrieved_count'), "chunks")
print("Time:", response['metadata']['total_time_ms'], "ms")

for source in response['sources']:
    print(f"  - {source['source']} (score: {source['score']:.2f})")
```

### Example 2: Retrieve-Only with Custom Processing

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Get context without generation
context, results = rag.retrieve_context(
    query="Explain quantum computing",
    top_k=5,
    max_context_chunks=3
)

if context:
    # Custom processing
    print(f"Found {len(results)} relevant chunks")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result.source_file}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Quality: {result.metadata.get('quality_score', 'N/A')}")
        print(f"   Preview: {result.content[:100]}...")
    
    # Use context for custom generation
    print("\nContext for LLM:")
    print(context.context_text[:500])
else:
    print("No relevant information found")
```

### Example 3: Filtered Search

```python
rag = RAGPipeline()

# High-quality code examples only
response = rag.query(
    query="Show me Python design patterns",
    top_k=10,
    filters={
        "doc_type": "code",
        "quality_score": 0.8,
        "language": "en"
    },
    dedupe_by_source=True,
    max_context_chunks=5
)

print(response['answer'])
```

### Example 4: Streaming Response

```python
import sys
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

response = rag.query(
    "Explain neural networks",
    stream=True
)

# Stream to stdout
sys.stdout.write("Assistant: ")
for token in response['streamer']:
    sys.stdout.write(token)
    sys.stdout.flush()

print()
print(f"\nSources: {len(response['sources'])}")
```

### Example 5: Integration with FastAPI (Like app.py)

```python
from fastapi import FastAPI, Form
from rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline(store_path="./vector_store")

@app.post("/api/query")
async def query(prompt: str = Form(...), top_k: int = Form(3)):
    # Include RAG context
    ctx, results = rag.retrieve_context(
        query=prompt,
        top_k=top_k,
        max_context_chunks=3
    )
    
    rag_context = ctx.context_text if ctx else "No context available"
    
    # Return JSON response
    return {
        "query": prompt,
        "context_chunks": len(results),
        "rag_ready": True,
        "sources": [r.source_file for r in results]
    }
```

---

## Integration with Ingestion

### Data Flow

```
ingestion_pipeline.py (output)
    ↓
    vector_store/
    ├── index.faiss (vectors)
    ├── docstore.json (chunks)
    └── index_meta.json (ID mapping)
    ↓
rag_pipeline.py (input)
    ↓
VectorStoreRetriever loads store
    ↓
Search and retrieve
    ↓
app.py integrates results
```

### Metadata Schema Alignment

**Ingestion Output (to_storage_dict):**
```python
metadata = {
    'source_document': './docs/file.pdf',
    'source_file': './docs/file.pdf',  # Alias
    'quality_score': 0.85,
    'entropy_score': 0.79,
    'semantic_coherence': 0.88,
    'signal_to_noise': 0.91,
    'extra_metadata': {...}
}
```

**Retrieval Input (_extract_source):**
```python
def _extract_source(metadata):
    return (
        metadata.get('source_document')
        or metadata.get('source_file')
        or 'unknown'
    )
```

**Both aliased for compatibility!**

### Index Metadata (index_meta.json)

Crucial for stable FAISS ID mapping:

```json
{
  "total_chunks": 1250,
  "next_id": 1251,
  "id_map": {
    "0": "chunk_abc123",
    "1": "chunk_def456",
    ...
  }
}
```

**Purpose:** Maps FAISS sequential IDs back to chunk IDs so:
- Queries return correct chunks
- Metadata retrieved reliably
- System survives index rebuilds

---

## Troubleshooting

### Issue: No results returned

**Symptom:** `retrieve_context()` returns (None, [])

**Causes:**
1. Vector store not initialized
2. Query mismatched to content
3. All results filtered out
4. Score threshold too high

**Solutions:**
```python
# Check vector store exists
import os
assert os.path.exists("./vector_store/index.faiss")
assert os.path.exists("./vector_store/docstore.json")

# Relax filters
context, results = rag.retrieve_context(
    query="your query",
    top_k=10,  # Get more candidates
    score_threshold=0.3,  # Lower threshold
    filters=None  # No filters
)

# Check what ingestion returned
print(f"Total chunks in store: {rag.retriever.get_stats()}")
```

### Issue: Wrong chunks returned

**Symptom:** Retrieved chunks don't match query

**Causes:**
1. Embedding model mismatch
2. Index corruption
3. Poor chunk quality
4. Query too vague

**Solutions:**
```python
# Verify embedding model
rag = RAGPipeline(
    embedding_model="BAAI/bge-large-en-v1.5"  # Match ingestion!
)

# Check retrieved scores
context, results = rag.retrieve_context(query)
for r in results:
    print(f"Score: {r.score:.3f}, Quality: {r.metadata.get('quality_score')}")

# Try specific query
response = rag.query("machine learning algorithms")  # More specific
```

### Issue: Index ID mapping missing

**Symptom:** Chunks appear in FAISS but not retrievable

**Causes:**
1. `index_meta.json` missing or corrupted
2. Ingestion didn't save metadata

**Solutions:**
```bash
# Check if index_meta.json exists
ls -la vector_store/index_meta.json

# Rebuild from scratch
rm vector_store/index_meta.json
python ingestion_pipeline.py --input /data/documents

# Or manually rebuild mapping
# (VectorStoreRetriever._build_fallback_id_map does this)
```

### Issue: Generation errors

**Symptom:** `generate()` fails or returns garbage

**Causes:**
1. LikedModel not loaded
2. CUDA out of memory
3. Prompt too long
4. Tokenizer encoding issues

**Solutions:**
```bash
# Use CPU instead
export RAG_DEVICE=cpu

# Reduce context size
max_context_tokens = 1024

# Rebuild vector store
rm -rf vector_store/
python ingestion_pipeline.py --input /data
```

### Issue: Slow retrieval

**Symptom:** Search takes > 5 seconds

**Causes:**
1. Large vector store
2. Many top-k results
3. Complex filtering
4. Disk I/O bottleneck

**Solutions:**
```python
# Reduce candidates retrieved
results = rag.retriever.search(
    query=query,
    top_k=3  # Instead of 10
)

# Simplify filters
filters = None  # No filtering

# Use SSD storage for vector_store
```

---

## Best Practices

1. **Match Embeddings:** Use same embedding model in ingestion and retrieval

2. **Quality Filtering:** Set `min_quality_score=0.6` in ingestion, `filters={"quality_score": 0.7}` in retrieval

3. **Monitor Latency:** Log generation times; typical: 1-5 seconds

4. **Cite Sources:** Always include source citations in response

5. **Handle Edge Cases:**
   ```python
   if not results:
       return "I don't have relevant information"
   ```

6. **Backup Index:** Regularly backup `./vector_store/` directory

7. **Test Queries:** Validate retrieval before deploying generation

---

## Performance Summary

| Operation | Time | Resources |
|-----------|------|-----------|
| Query embedding | 100-200ms | CPU/GPU |
| FAISS search (1000 docs) | 10-50ms | RAM |
| Context building | 50-100ms | CPU |
| LLM generation | 1-5 sec | GPU |
| **Total latency** | **1.5-5.5 sec** | **GPU/4GB+** |

**Optimization:** Use GPU for sub-second generation latencies.

---

**Last Updated:** 2026-02-15  
**Version:** 4.0  
**Status:** Production Ready ✓
