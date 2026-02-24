# Advanced Ingestion Pipeline v4.0

**Complete Production-Grade Document Processing and Vector Storage System**

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
10. [Parallel Processing](#parallel-processing)
11. [Quality Metrics](#quality-metrics)
12. [API Reference](#api-reference)
13. [Examples](#examples)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The **Advanced Ingestion Pipeline** is a comprehensive document processing system designed to:

- Load documents from multiple formats (PDF, DOCX, HTML, CSV, Code files, TXT)
- Build structured document graphs preserving hierarchical relationships
- Perform adaptive semantic chunking based on content type
- Generate dual embeddings (full + summary representations)
- Apply intelligent deduplication and quality filtering
- Store results in a high-performance FAISS vector index with persistent metadata
- Support parallel file processing for scalability

### Key Features

✓ **Multi-Format Support**: PDF, DOCX, HTML, CSV, Python, JavaScript, Java, JSON, TXT, Markdown  
✓ **Structure Preservation**: Document hierarchies maintained through graph structures  
✓ **Adaptive Chunking**: Content-aware chunk sizing (500-2000 tokens)  
✓ **Dual Embeddings**: Full chunk (1024-dim BGE-Large) + Summary (384-dim MiniLM)  
✓ **Quality Scoring**: Information density, entropy, coherence, signal-to-noise metrics  
✓ **Deduplication**: Semantic and content-based duplicate detection  
✓ **Parallel Processing**: Multi-threaded ingestion with thread-safe operations  
✓ **Persistent Storage**: FAISS index + JSON metadata with stable ID mapping  

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Document Files                        │
│  (PDF, DOCX, HTML, CSV, Code, TXT, MD - any combination)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────▼────────────────┐
         │  FILE TRACKING & STATUS    │ 
         │  (NEW/UPDATED/UNCHANGED)   │
         └───────────┬────────────────┘
                     │
     ┌───────────────┴─────────────────┐
     │   LOAD WITH STRUCTURE GRAPH     │
     │  (Format-specific loaders)      │
     │  - PDFLoader                    │
     │  - DOCXLoader                   │
     │  - HTMLLoader                   │
     │  - CSVLoader                    │
     │  - CodeLoader                   │
     └───────────────┬─────────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  ADAPTIVE SEMANTIC CHUNKING      │
     │  - Content-type detection        │
     │  - Dynamic chunk sizing          │
     │  - Boundary preservation         │
     └───────────────┬──────────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  CONTENT NORMALIZATION           │
     │  - Unicode normalization         │
     │  - HTML entity decoding          │
     │  - Whitespace cleaning           │
     └───────────────┬──────────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  DEDUPLICATION ENGINE            │
     │  - Semantic similarity check     │
     │  - Content hash comparison       │
     └───────────────┬──────────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  METADATA ENRICHMENT             │
     │  - Entity extraction             │
     │  - Keyword detection             │
     │  - Language identification       │
     └───────────────┬──────────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  QUALITY SCORING                 │
     │  - Information density (entropy) │
     │  - Semantic coherence            │
     │  - Signal-to-noise ratio         │
     │  - Readability metrics           │
     └───────────────┬──────────────────┘
                     │
     ┌───────────────▼──────────────────┐
     │  DUAL EMBEDDING GENERATION       │
     │  - Full embedding (BGE-Large)    │
     │  - Summary embedding (MiniLM)    │
     └───────────────┬──────────────────┘
                     │
     ┌───────────────▼─────────────────────┐
     │  VECTOR STORAGE (FAISS)             │
     │  ┌──────────────────────────────┐   │
     │  │ IndexFlatIP + IndexIDMap2     │   │
     │  │ Inner product similarity      │   │
     │  │ Stable FAISS ID mapping       │   │
     │  └──────────────────────────────┘   │
     │  ┌──────────────────────────────┐   │
     │  │ docstore.json                │   │
     │  │ Chunk content + metadata     │   │
     │  └──────────────────────────────┘   │
     │  ┌──────────────────────────────┐   │
     │  │ index_meta.json              │   │
     │  │ FAISS ID ↔ Chunk ID mapping  │   │
     │  └──────────────────────────────┘   │
     └─────────────────────────────────────┘
                     │
         ┌───────────▼────────────────┐
         │  OUTPUT: Vector Store      │
         │  Ready for RAG Retrieval    │
         └────────────────────────────┘
```

---

## Components

### 1. **ProcessedChunk** (Data Structure)

Represents a processed, embedding-ready chunk of content.

**Fields:**
```python
# Identity
id: str                              # Unique chunk identifier
content: str                         # Original content
cleaned_content: str                 # Normalized content

# Embeddings
embedding: np.ndarray               # Full chunk embedding (1024-dim)
summary_embedding: np.ndarray       # Summary embedding (384-dim)

# Source tracking
source_document: str                # Source file path
source_file_fingerprint: str        # File content hash
doc_type: str                       # 'pdf', 'docx', 'html', 'code', etc.
content_type: ContentType           # PROSE, CODE, MIXED, DATA, METADATA

# Structural context
section_title: str                  # Parent section name
heading_hierarchy: List[str]        # Full breadcrumb path
breadcrumb: List[str]               # Document hierarchy
parent_doc_id: str                  # Parent document ID

# Content metadata
language: str                       # Detected language (default: 'en')
keywords: List[str]                 # Extracted keywords
named_entities: List[Tuple]         # (entity, type) pairs
token_count: int                    # Token count of chunk

# Quality metrics (0.0-1.0 scale)
quality_score: float                # Overall quality
information_density: float          # Entropy-based density
entropy_score: float                # Shannon entropy
semantic_coherence: float           # Intra-chunk coherence
signal_to_noise: float              # Clean vs boilerplate ratio

# Flexible metadata
metadata: Dict[str, Any]            # Task-specific annotations

# Relationships
related_chunks: List[str]           # Semantically similar chunks
prev_chunk_id: str                  # Sequential previous
next_chunk_id: str                  # Sequential next
sibling_chunk_ids: List[str]        # Same parent section
```

### 2. **AdvancedIngestionPipeline** (Main Orchestrator)

Coordinates the entire ingestion workflow with parallel processing support.

**Key Methods:**
- `__init__(store_path, min_quality_score)` - Initialize pipeline
- `ingest(input_dirs, max_workers, model_name)` - Main entry point with parallel support
- `process_file(file_path, status, fingerprint)` - Process single file
- `get_stats()` - Retrieve ingestion statistics

### 3. **Format-Specific Loaders**

#### PDFLoader
- Extracts text with structure preservation
- Detects and extracts images (with optional OCR)
- Preserves page hierarchy and layout information
- Builds document graph with page → section → paragraph structure

#### DOCXLoader
- Preserves document hierarchy (body → section → paragraph)
- Extracts tables, lists, and special formatting
- Maintains style information for content-type classification

#### HTMLLoader
- Uses Trafilatura for content extraction
- Preserves semantic HTML structure
- Builds graph: document → section → heading → paragraph

#### CSVLoader
- Treats rows as chunks or groups rows by column header type
- Preserves columnar relationships
- Creates metadata for each row

#### CodeLoader
- Uses Tree-Sitter for syntax-aware parsing
- Preserves code structure: file → class → function → statement
- Maintains indentation and context

### 4. **AdaptiveSemanticChunker**

Intelligently segments documents with content-aware boundaries.

**Strategy:**
- **Min chunk**: 100 tokens
- **Target chunk**: 700 tokens
- **Max chunk**: 2000 tokens
- **Sentence-level boundaries**: Chunks respect sentence endings
- **Heading-aware**: Large headings trigger new chunks
- Dual embeddings guide coherence decisions

### 5. **DeduplicationEngine**

Removes redundant content using multiple strategies.

**Methods:**
- **Content-hash based**: Exact duplicate detection
- **Semantic similarity**: Detects ~90%+ similar chunks
- **Threshold**: 0.95 cosine similarity with fallback
- Preserves highest quality candidate

### 6. **MetadataEnricher**

Enriches chunks with contextual information.

**Extracts:**
- Keywords (TF-IDF weighted)
- Named entities (spaCy-based)
- Language detection
- Content type classification
- Quality metrics

### 7. **ProductionVectorStore**

Manages persistent FAISS storage with metadata.

**Storage Structure:**
```
./vector_store/
├── index.faiss              # FAISS vector index
├── docstore.json            # Chunk content + metadata
├── index_meta.json          # FAISS ID ↔ Chunk ID mapping (NEW)
├── fingerprints.json        # File tracking
└── chunks/                  # Individual chunk JSON files (optional)
```

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Required packages
pip install -r requirements.txt
```

### Requirements

Key dependencies (from `requirements.txt`):
- `sentence-transformers` - BGE-Large embeddings
- `faiss-cpu` - Vector indexing
- `pdfplumber` - PDF extraction
- `python-docx` - DOCX processing
- `trafilatura` - HTML extraction
- `tree-sitter` + `tree-sitter-languages` - Code parsing
- `langdetect` - Language detection
- `nltk` - Text segmentation
- `scikit-learn` - Deduplication similarity
- `tqdm` - Progress bars

### Optional Dependencies

```bash
# PDF OCR support
pip install pytesseract pillow
# Install system Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

# Advanced language support
pip install spacy
python -m spacy download en_core_web_sm
```

### Quick Setup

```bash
# Clone or navigate to project
cd optimus-prime-rag

# Install dependencies
pip install -r requirements.txt

# Initialize vector store (automatic on first ingest)
mkdir -p vector_store
```

---

## Input Specification

### Supported File Formats

| Format | Extension | Loader | Structure Preserved |
|--------|-----------|--------|---------------------|
| PDF | `.pdf` | PDFLoader | ✓ Pages, sections, layout |
| Word | `.docx` | DOCXLoader | ✓ Sections, paragraphs |
| HTML | `.html, .htm` | HTMLLoader | ✓ Semantic structure |
| CSV | `.csv` | CSVLoader | ✓ Row relationships |
| Python | `.py` | CodeLoader | ✓ Classes, functions |
| JavaScript | `.js` | CodeLoader | ✓ Functions, modules |
| Java | `.java` | CodeLoader | ✓ Classes, methods |
| JSON | `.json` | CodeLoader | ✓ Object hierarchy |
| Text | `.txt, .md` | TextLoader | ✓ Paragraph breaks |

### Input Requirements

**File Location:**
```
input_dirs = [
    "/path/to/documents",
    "/path/to/pdf/collection",
    "/path/to/single/file.pdf"
]
```

**File Size Limits:**
- **Recommended**: < 100 MB per file
- **Maximum practical**: 500 MB per file
- **Total collection**: Tested up to 10 GB

**File Encoding:**
- UTF-8 preferred
- Auto-detects common encodings (latin-1, cp1252, etc.)

---

## Processing Pipeline

### Step 1: File Discovery & Tracking

```
Input: Directory paths or file list
Output: Categorized files (NEW, UPDATED, UNCHANGED, DUPLICATE)

Process:
1. Collect all files matching pattern
2. Compute fingerprint (SHA-256 of content)
3. Compare with stored fingerprints
4. Categorize status
```

**File Status:**
- **NEW**: Never seen before
- **UPDATED**: Content changed since last ingestion
- **UNCHANGED**: Identical to previous ingestion (skipped)
- **DUPLICATE**: Identical to another file in current batch

### Step 2: Format Detection & Loading

```
Input: File path
Output: DocumentNode (hierarchical graph structure)

Process:
1. Detect format from extension
2. Select appropriate loader
3. Extract content with structure preservation
4. Build document graph (tree of nodes)
5. Attach metadata (source, format)
```

### Step 3: Adaptive Semantic Chunking

```
Input: DocumentNode tree
Output: List[ProcessedChunk]

Process:
1. Traverse document tree depth-first
2. Accumulate text until target size (~700 tokens)
3. Respect sentence boundaries (avoid mid-sentence breaks)
4. Trigger new chunk on:
   - Max token count reached
   - Heading encountered
   - Major structural boundary
5. Assign hierarchy metadata (breadcrumb, section_title)
6. Link sequential chunks (prev_chunk_id, next_chunk_id)
```

**Chunking Parameters:**
```python
MIN_TOKENS = 100         # Minimum chunk size
TARGET_TOKENS = 700      # Preferred chunk size
MAX_TOKENS = 2000        # Maximum before forcing break
OVERLAP = 100            # Token overlap for context
```

### Step 4: Content Normalization

```
Input: Raw chunk content
Output: Cleaned, normalized content

Process:
1. Unicode normalization (NFD form)
2. HTML entity decoding (&nbsp; → space, etc.)
3. Remove excess whitespace
4. Fix encoding artifacts
5. Decode base64 image references
6. Remove control characters
7. Normalize quotes and punctuation
```

### Step 5: Deduplication

```
Input: List[ProcessedChunk] (before dedup)
Output: List[ProcessedChunk] (after dedup)

Process:
1. Hash content (MD5) for exact matches
2. For potential duplicates:
   - Compute semantic similarity (cosine)
   - If similarity > 0.95:
     - Mark as duplicate
     - Keep highest quality_score candidate
     - Store mapping for lineage
3. Filter duplicates
```

### Step 6: Metadata Enrichment & Quality Scoring

```
Input: ProcessedChunk (basic)
Output: ProcessedChunk (enriched with metrics)

Process:
1. Extract keywords using TF-IDF
2. Identify named entities (spaCy)
3. Detect language (langdetect)
4. Classify content type (PROSE, CODE, etc.)
5. Calculate quality metrics:
   - Information density (entropy score)
   - Semantic coherence
   - Signal-to-noise ratio
   - Overall quality_score
6. Filter chunks (quality_score < min_quality → discarded)
```

**Quality Scoring Algorithm:**
```
quality_score = (
    0.3 * information_density +
    0.25 * semantic_coherence +
    0.25 * signal_to_noise +
    0.2 * (1 - language_inconsistency)
)
```

**Default Threshold:** 0.4 (40% quality minimum)

### Step 7: Dual Embedding Generation

```
Input: ProcessedChunk (with cleaned content)
Output: ProcessedChunk (with embeddings)

Process:
1. Generate full embedding:
   - Model: BAAI/bge-large-en-v1.5
   - Dimension: 1024
   - Input: cleaned_content
   - Normalize to unit vectors

2. Generate summary embedding:
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - Dimension: 384
   - Input: first 200 tokens of content
   - For quick similarity matching

3. Both embeddings normalized (L2 norm = 1.0)
```

### Step 8: Vector Storage

```
Input: List[ProcessedChunk] (with embeddings)
Output: FAISS index + docstore.json + index_meta.json

Process:
1. Retrieve embeddings from chunks
2. Stack into matrix (N × 1024)
3. Add to FAISS IndexIDMap2:
   - FAISS ID: Sequential from next_id
   - Chunk ID: chunk.id
4. Store chunk data in docstore.json
5. Persist ID mapping in index_meta.json
6. Save FAISS index to disk
7. Track file in fingerprints.json
```

**Storage Guarantees:**
- ✓ ID mapping persists across sessions
- ✓ FAISS search returns correct chunk IDs
- ✓ Metadata preserved for filtering/ranking
- ✓ Thread-safe concurrent writes

---

## Output Specification

### Primary Output: Vector Store

**Location:** `./vector_store/` (configurable)

**Contents:**

#### 1. `index.faiss`
- FAISS IndexIDMap2 object
- Stores full embeddings (1024-dim × N)
- Enables semantic similarity search
- Format: FAISS binary format

#### 2. `docstore.json`
```json
{
  "chunk_abc123": {
    "id": "chunk_abc123",
    "content": "Cleaned chunk text...",
    "embedding": [0.1, 0.2, ...],
    "summary_embedding": [0.05, 0.15, ...],
    "metadata": {
      "source_document": "./docs/file.pdf",
      "source_file": "./docs/file.pdf",
      "source_fingerprint": "sha256_hash",
      "doc_type": "pdf",
      "file_type": "pdf",
      "content_type": "prose",
      "section_title": "Introduction",
      "heading_hierarchy": ["Chapter 1", "Section 1.1"],
      "breadcrumb": ["Chapter 1", "Section 1.1", "Paragraph 3"],
      "language": "en",
      "keywords": ["keyword1", "keyword2"],
      "token_count": 650,
      "quality_score": 0.85,
      "information_density": 0.82,
      "entropy_score": 0.79,
      "semantic_coherence": 0.88,
      "signal_to_noise": 0.91,
      "pipeline_version": "4.0",
      "embedding_model": "BAAI/bge-large-en-v1.5",
      "created_at": "2026-02-15T10:30:00",
      "extra_metadata": { "custom": "values" },
      "relationships": {
        "related": ["chunk_xyz789"],
        "prev": "chunk_prev123",
        "next": "chunk_next456",
        "siblings": ["chunk_sibling1", "chunk_sibling2"]
      }
    }
  }
}
```

#### 3. `index_meta.json` (NEW - Stable ID Mapping)
```json
{
  "total_chunks": 1250,
  "next_id": 1251,
  "id_map": {
    "0": "chunk_abc123",
    "1": "chunk_def456",
    "2": "chunk_ghi789",
    ...
  }
}
```

**Purpose:** Maps FAISS sequential IDs to chunk IDs for reliable retrieval across sessions.

#### 4. `fingerprints.json`
```json
{
  "./docs/file.pdf": {
    "content_hash": "sha256_hash",
    "file_hash": "another_hash",
    "file_path": "./docs/file.pdf",
    "file_size": 2500000,
    "last_modified": 1676380200.0,
    "chunk_count": 45
  }
}
```

### Statistics Output

**Logged Summary:**
```
========================================
INGESTION COMPLETE
========================================
NEW files:          45
UPDATED files:      12
UNCHANGED files:    3 (skipped)
DUPLICATE files:    0
FAILED files:       0
----------------------------------------
Total chunks created:   1,250
Total chunks deduped:   45
Total chunks filtered:  12
Total chunks indexed:   1,193
----------------------------------------
Storage size:       125.4 MB
Processing time:    8m 45s
========================================
```

**Programmatic Stats:**
```python
stats = pipeline.stats  # Dict with all counts
```

---

## Usage Guide

### Basic Usage

#### 1. **Simple Ingestion**

```python
from ingestion_pipeline import AdvancedIngestionPipeline

# Initialize pipeline
pipeline = AdvancedIngestionPipeline(
    store_path="./vector_store",
    min_quality_score=0.4
)

# Ingest documents
stats = pipeline.ingest(
    input_dirs=[
        "/path/to/documents",
        "/path/to/pdfs"
    ],
    max_workers=4  # Use 4 parallel threads
)

print(f"Indexed {stats['chunks_indexed']} chunks")
```

#### 2. **Command-Line Usage**

```bash
# Basic ingestion
python ingestion_pipeline.py \
    --input /path/to/docs \
    --store ./vector_store \
    --model BAAI/bge-large-en-v1.5 \
    --quality 0.4 \
    --workers 8

# Multiple input directories
python ingestion_pipeline.py \
    --input /docs/pdfs /docs/word /docs/html \
    --store /mnt/vector_store \
    --workers auto
```

#### 3. **Advanced Configuration**

```python
from ingestion_pipeline import AdvancedIngestionPipeline

pipeline = AdvancedIngestionPipeline(
    store_path="./vector_store",
    min_quality_score=0.5  # Higher quality threshold
)

stats = pipeline.ingest(
    input_dirs=["/data/documents"],
    file_pattern="*.pdf",           # Only PDF files
    max_workers=8,                  # 8 parallel workers
    model_name="BAAI/bge-large-en-v1.5"  # Embedding model
)
```

### Advanced Scenarios

#### Single File Processing

```python
from pathlib import Path
from ingestion_pipeline import AdvancedIngestionPipeline

pipeline = AdvancedIngestionPipeline()

# Process single file
file_path = "/path/to/document.pdf"
stats = pipeline.ingest([file_path], max_workers=1)
```

#### Selective Reingestion

```python
# Pipeline tracks files automatically
# Already-ingested files with same content are skipped
stats = pipeline.ingest(["/data/documents"])

# Output shows:
# NEW files: 10
# UPDATED files: 2
# UNCHANGED files: 35 (skipped)
```

#### Quality Filtering

```python
# Strict quality threshold (filter poor-quality chunks)
pipeline = AdvancedIngestionPipeline(min_quality_score=0.7)
stats = pipeline.ingest(["/data/documents"])

# Output shows more chunks filtered:
# Chunks created: 2000
# Chunks filtered: 400  (quality < 0.7)
# Chunks indexed: 1600
```

---

## Configuration

### Environment Variables

```bash
# Vector store location
export RAG_VECTOR_STORE_PATH=/mnt/vectors

# Embedding model (default: BAAI/bge-large-en-v1.5)
export RAG_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Minimum quality score (default: 0.4)
export RAG_MIN_QUALITY=0.4

# Parallel workers (default: auto-detect)
export RAG_WORKERS=8
```

### Programmatic Configuration

```python
# Disable ORC for faster PDF processing
import ingestion_pipeline
ingestion_pipeline.USE_OCR = False

# Adjust chunking parameters
ingestion_pipeline.MIN_TOKENS = 50
ingestion_pipeline.TARGET_TOKENS = 500
ingestion_pipeline.MAX_TOKENS = 1500

# Create pipeline
pipeline = AdvancedIngestionPipeline()
```

### Pipeline Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `store_path` | `./vector_store` | any path | Output storage location |
| `min_quality_score` | `0.4` | 0.0-1.0 | Minimum chunk quality to keep |
| `embedding_model` | `BAAI/bge-large-en-v1.5` | model name | Which embedding model to use |
| `max_workers` | auto | 1-32 | Parallel file processing threads |

---

## Parallel Processing

### Thread Safety

✓ **Thread-safe operations:**
- Embedding model sharing (protected by lock)
- Vector store writes (protected by lock)
- File tracking (protected by lock)
- Statistics updates (protected by lock)

### Performance Tuning

**Auto Detection (Recommended):**
```bash
# Pipeline auto-detects CPU count
python ingestion_pipeline.py --input /data --workers auto
```

**Manual Configuration:**
```bash
# For 8-core CPU: use 6-8 workers
python ingestion_pipeline.py --input /data --workers 8

# For 16-core CPU: use 10-12 workers
python ingestion_pipeline.py --input /data --workers 12

# Single-threaded (debugging):
python ingestion_pipeline.py --input /data --workers 1
```

**Memory Considerations:**
- Each worker needs ~2-4 GB for embedding models
- 4 workers = 8-16 GB RAM recommended
- Adjust down if memory constrained

### Performance Metrics

**Typical Throughput (with 4 workers):**
- PDF: ~200 pages/hour
- DOCX: ~300 documents/hour
- HTML: ~500 documents/hour
- Code files: ~1000 files/hour
- Text: ~2000 files/hour

---

## Quality Metrics

### Information Density (Entropy Score)

```
Measures information content per token

High entropy = Diverse vocabulary = Valuable content
Low entropy = Repetitive text = Low quality

Algorithm: Shannon entropy of token distribution
Range: 0.0-1.0
Good: > 0.7
```

### Semantic Coherence

```
Measures how well sentences relate to each other

High coherence = Sentences flow logically
Low coherence = Disjointed content

Algorithm: Average cosine similarity between consecutive sentence embeddings
Range: 0.0-1.0
Good: > 0.8
```

### Signal-to-Noise Ratio

```
Measures useful content vs boilerplate

High ratio = Mostly useful content
Low ratio = Many headers, footers, repetition

Algorithm: Ratio of unique tokens to total tokens
Range: 0.0-1.0
Good: > 0.8
```

### Overall Quality Score

```
Composite metric combining all above

Formula:
quality_score = (
    0.30 * information_density +
    0.25 * semantic_coherence +
    0.25 * signal_to_noise +
    0.20 * language_consistency
)

Range: 0.0-1.0
Good: > 0.6
Excellent: > 0.8
```

---

## API Reference

### AdvancedIngestionPipeline

#### `__init__(store_path, min_quality_score)`

Initialize ingestion pipeline.

**Parameters:**
- `store_path` (str): Location for vector store (default: './vector_store')
- `min_quality_score` (float): Minimum quality to keep (default: 0.4)

**Returns:** AdvancedIngestionPipeline instance

**Example:**
```python
pipeline = AdvancedIngestionPipeline(
    store_path="/mnt/vectors",
    min_quality_score=0.5
)
```

#### `ingest(input_dirs, file_pattern, max_workers, model_name)`

Main ingestion method with parallel processing.

**Parameters:**
- `input_dirs` (List[str]): Input directories or files
- `file_pattern` (str): File pattern to match (default: '*')
- `max_workers` (Optional[int]): Parallel workers, None=auto (default: None)
- `model_name` (str): Embedding model (default: 'BAAI/bge-large-en-v1.5')

**Returns:** Dict with statistics

**Example:**
```python
stats = pipeline.ingest(
    input_dirs=["/data/docs"],
    max_workers=8,
    model_name="BAAI/bge-large-en-v1.5"
)
print(stats['chunks_indexed'])  # 1,234
```

#### `process_file(file_path, status, fingerprint)`

Process single file (called internally by ingest).

**Parameters:**
- `file_path` (str): Path to file
- `status` (str): 'NEW' or 'UPDATED'
- `fingerprint` (FileFingerprint): File fingerprint object

**Returns:** Number of chunks indexed

#### `get_stats()`

Get current ingestion statistics.

**Returns:** Dict with counts and metrics

**Example:**
```python
stats = pipeline.get_stats()
# {
#    'new_files': 45,
#    'updated_files': 12,
#    'unchanged_files': 3,
#    'duplicate_files': 0,
#    'failed_files': 0,
#    'chunks_created': 2500,
#    'chunks_deduped': 145,
#    'chunks_filtered': 250,
#    'chunks_indexed': 2105
# }
```

### ProcessedChunk

Data class representing a processed chunk.

**Key Attributes:**
```python
chunk.id                    # Unique chunk ID
chunk.content              # Original text
chunk.cleaned_content      # Normalized text
chunk.embedding            # Full embedding (1024-dim)
chunk.summary_embedding    # Summary embedding (384-dim)
chunk.quality_score        # 0.0-1.0 overall quality
chunk.metadata             # All stored metadata as dict
```

**Method:**
```python
storage_dict = chunk.to_storage_dict()  # Convert to JSON-serializable dict
```

---

## Examples

### Example 1: Basic Document Ingestion

```python
from ingestion_pipeline import AdvancedIngestionPipeline

# Initialize
pipeline = AdvancedIngestionPipeline()

# Ingest documents
stats = pipeline.ingest(
    input_dirs=[
        "./documents/pdfs",
        "./documents/word"
    ],
    max_workers=4
)

# Check results
print(f"✓ Ingested {stats['chunks_indexed']} chunks")
print(f"⊘ Skipped {stats['unchanged_files']} unchanged files")
print(f"✗ Failed {stats['failed_files']} files")
```

### Example 2: High-Quality Selective Ingestion

```python
# Keep only high-quality chunks
pipeline = AdvancedIngestionPipeline(
    store_path="./vector_store_hq",
    min_quality_score=0.75  # 75% minimum quality
)

stats = pipeline.ingest(["/data/knowledge_base"])

print(f"Created: {stats['chunks_created']} chunks")
print(f"Filtered: {stats['chunks_filtered']} (poor quality)")
print(f"Final: {stats['chunks_indexed']} chunks")
```

### Example 3: Multi-Format Document Collection

```python
# Mixed document types
pipeline = AdvancedIngestionPipeline()

stats = pipeline.ingest(
    input_dirs=[
        "/data/research_papers",    # PDFs
        "/data/documentation",      # DOCX + MARKDOWN
        "/data/code_repository",    # .py, .js, .java
        "/data/datasets",           # CSV files
        "/data/web_content"         # HTML files
    ],
    max_workers=8
)

# All formats handled automatically
print(f"Total chunks: {stats['chunks_indexed']}")
```

### Example 4: Incremental Updates

```python
pipeline = AdvancedIngestionPipeline()

# First ingestion
print("Initial ingestion...")
stats1 = pipeline.ingest(["/data/documents"])
print(f"Indexed: {stats1['chunks_indexed']}")

# Add new files later
print("\nSecond run (with new files)...")
stats2 = pipeline.ingest(["/data/documents"])
print(f"NEW: {stats2['new_files']}")
print(f"UPDATED: {stats2['updated_files']}")
print(f"UNCHANGED (skipped): {stats2['unchanged_files']}")
print(f"Total indexed: {stats2['chunks_indexed']}")
```

---

## Troubleshooting

### Issue: Out of Memory

**Symptom:** `MemoryError` during ingestion

**Solutions:**
```bash
# Reduce workers (fewer concurrent embeddings)
python ingestion_pipeline.py --input /data --workers 2

# Process smaller batches
python ingestion_pipeline.py --input /data/subset --workers 4
```

### Issue: Slow Processing

**Symptom:** Takes too long to ingest documents

**Solutions:**
```bash
# Increase workers (if CPU cores available)
python ingestion_pipeline.py --input /data --workers 12

# Disable OCR for faster PDF parsing
# Edit: USE_OCR = False in ingestion_pipeline.py

# Skip already-processed files
# Pipeline automatically skips UNCHANGED files
```

### Issue: Missing PDF Content

**Symptom:** PDF files ingest but chunks are empty

**Solutions:**
```python
# Check if PDF is image-based (needs OCR)
# Ensure Tesseract is installed for OCR support

# Use alternative PDF loader
# Edit PDFLoader to use pymupdf instead of pdfplumber
```

### Issue: Import Errors

**Symptom:** `ModuleNotFoundError`

**Solutions:**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# For PDF OCR support
pip install pytesseract pillow

# For code parsing
pip install tree-sitter tree-sitter-languages
```

### Issue: Vector Store Corruption

**Symptom:** FAISS search fails or returns wrong results

**Solutions:**
```bash
# Backup and rebuild
cp -r vector_store vector_store.backup
rm vector_store/index.faiss

# Re-ingest (will rebuild from docstore)
python ingestion_pipeline.py --input /data

# Check index_meta.json integrity
# If corrupted, delete it (will rebuild on next ingest)
rm vector_store/index_meta.json
```

### Issue: Metadata Not Persisting

**Symptom:** Source file information lost after restart

**Solutions:**
```bash
# Verify index_meta.json exists
ls -la vector_store/index_meta.json

# Check docstore.json for metadata
head -100 vector_store/docstore.json | grep metadata

# Re-run ingestion if needed
python ingestion_pipeline.py --input /data --workers 1
```

---

## Best Practices

1. **Quality Threshold:** Set `min_quality_score=0.5` for production (default 0.4 is lenient)

2. **Parallel Workers:** Use `max_workers = cpu_count - 2` for optimal performance

3. **Incremental Updates:** Always run pipeline on full document collection to track changes

4. **Monitor Storage:** Check vector_store size regularly; consider cleanup of old files

5. **Backup:** Backup `vector_store/` directory regularly (contains all indexed content)

6. **Testing:** Start with small batch, verify results before scaling

7. **Logging:** Enable logging to diagnose issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Performance Summary

| Operation | Time | Resources |
|-----------|------|-----------|
| Load 100 PDFs | 5-10 min | 4GB RAM, 4 workers |
| 1000 chunk dedup | 2-5 sec | Embedding comparison |
| Embedding gen (1000 chunks) | 10-20 sec | GPU recommended |
| FAISS index build | 1-2 sec | < 1GB |
| Full flow (1000 docs) | 15-30 min | 8GB RAM, 8 workers |

---

**Last Updated:** 2026-02-15  
**Version:** 4.0  
**Status:** Production Ready ✓
