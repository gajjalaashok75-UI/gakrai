# AutoBot - ReAct Agent with Local LLM

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/gajjalaashok75-UI/gakrai)](https://github.com/gajjalaashok75-UI/gakrai/issues)

**AutoBot** is a sophisticated **ReAct (Reason + Act) agent system** powered by local LLMs, designed to provide intelligent, context-aware responses through semantic search and tool integration. It combines reasoning, action execution, and memory management into a cohesive autonomous agent framework.

## 🚀 Key Features

- **🧠 ReAct Agent Architecture**: Implements Reason + Act pattern with iterative reasoning loops
- **🔍 Semantic Search**: FAISS-based vector similarity search with ChromaDB support
- **💾 3-Tier Memory System**: Short-term, working, and long-term memory with episodic storage
- **🛠️ Tool Integration**: Web search pipeline with extensible tool registry
- **📚 Document Processing**: Multi-format ingestion (PDF, DOCX, HTML, CSV, Code files)
- **🤖 Local LLM Support**: LFM2.5-1.2B-Instruct with GGUF quantization
- **⚡ Parallel Processing**: Multi-threaded document ingestion and processing
- **🎯 Quality Filtering**: Intelligent content scoring and deduplication

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Components](#components)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT (Natural Language)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────▼────────────────┐
         │  ReAct ORCHESTRATOR        │
         │  (Reason + Act Pattern)    │
         └───────────┬────────────────┘
                     │
     ┌───────────────▼─────────────────┐
     │  LLM INTERFACE                  │
     │  (LFM2.5-1.2B-Instruct)        │
     │  - GGUF Support                 │
     │  - Chat Templates               │
     └───────────────┬─────────────────┘
                     │
     ┌───────────────▼─────────────────┐
     │  TOOL REGISTRY                  │
     │  - Web Search                   │
     │  - Extensible Framework         │
     └───────────────┬─────────────────┘
                     │
     ┌───────────────▼─────────────────┐
     │  MEMORY MANAGER                 │
     │  - Short-term (Session)         │
     │  - Working (Shared State)       │
     │  - Long-term (Persistent)       │
     │  - Vector Store (FAISS/Chroma)  │
     └─────────────────────────────────┘
```

### ReAct Flow

1. **User Input** → ReAct Orchestrator
2. **First Pass**: LLM analyzes query and decides if tools are needed
3. **Tool Execution**: If needed, executes web search or other tools
4. **Second Pass**: LLM generates final answer grounded in tool results
5. **Memory Storage**: Conversation stored in multi-tier memory system

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM (recommended for local LLM)
- CUDA-compatible GPU (optional, for faster inference)

### Step 1: Clone Repository

```bash
git clone https://github.com/gajjalaashok75-UI/gakrai.git
cd gakrai
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model (Optional)

The model will be downloaded automatically on first run, or you can pre-download:

```bash
# The LFM2.5-1.2B-Instruct model will be downloaded to ./models/
# Approximately 1.2GB download
```

### Step 5: Initialize Configuration

```bash
# Configuration is automatically loaded from config/settings.yaml
# Customize settings if needed (see Configuration section)
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from core.react_orchestrator import ReActOrchestrator
import yaml

# Load configuration
with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize orchestrator
orchestrator = ReActOrchestrator(config)

async def main():
    # Initialize all components
    await orchestrator.initialize()
    
    # Ask a question
    response = await orchestrator.handle_input(
        "What are the latest developments in artificial intelligence?"
    )
    
    print("Response:", response.response)
    print("Steps taken:", response.total_steps)
    print("Execution time:", f"{response.execution_time:.2f}s")

# Run the example
asyncio.run(main())
```

### Interactive CLI

```bash
# Run the interactive command-line interface
python main.py

# Example interaction:
> What is machine learning?
[AutoBot analyzes the question, searches for current information, and provides a comprehensive answer]

> history
[Shows conversation history]

> clear
[Clears session memory]
```

### Command Line Options

```bash
# Run with ctransformers demo
python main.py --ctransformers-demo "Explain quantum computing"

# Custom configuration
python main.py --config custom_config.yaml
```

## 💡 Usage Examples

### Example 1: Web Search Integration

```python
# AutoBot automatically determines when to search the web
response = await orchestrator.handle_input(
    "What are the current Python 3.12 features?"
)
# AutoBot will:
# 1. Recognize this needs current information
# 2. Execute web search
# 3. Analyze results
# 4. Provide comprehensive answer with sources
```

### Example 2: Document Ingestion and RAG

```python
from memory.ingestion_pipeline import AdvancedIngestionPipeline

# Ingest documents into vector store
pipeline = AdvancedIngestionPipeline(
    store_path="./memory/vector_store",
    min_quality_score=0.4
)

# Process documents
stats = pipeline.ingest(
    input_dirs=["./documents/pdfs", "./documents/code"],
    max_workers=4
)

print(f"Indexed {stats['chunks_indexed']} chunks from {stats['new_files']} files")

# Now AutoBot can answer questions about your documents
response = await orchestrator.handle_input(
    "Based on my documents, explain the main concepts"
)
```

### Example 3: Memory and Context

```python
# AutoBot maintains conversation context
await orchestrator.handle_input("What is Python?")
await orchestrator.handle_input("What are its main advantages?")  # Refers to Python
await orchestrator.handle_input("Show me some code examples")     # Still about Python

# Access conversation history
history = await orchestrator.memory.get_recent_interactions(limit=10)
for interaction in history:
    print(f"Q: {interaction['user_input']}")
    print(f"A: {interaction['response'][:100]}...")
```

### Example 4: Tool Integration

```python
# AutoBot can be extended with custom tools
from tools.tool_registry import ToolRegistry

# The tool registry is extensible - add your own tools
# Tools are automatically detected and used by the ReAct agent
```

## ⚙️ Configuration

### Main Configuration (`config/settings.yaml`)

```yaml
assistant:
  name: "AutoBot"
  version: "0.2.0"

llm:
  intent_model:
    name: "LFM2.5-1.2B-Instruct-Q5_K_M"
    local_path: "./models/LFM2.5-1.2B-Instruct-Q5_K_M.gguf"
    max_tokens: 2048
    temperature: 0.3

memory:
  short_term_limit: 100
  long_term_db: "./memory/long_term.db"
  short_term_db: "./memory/short_term.db"
  vector_store: "./memory/vector_store"

tools:
  enabled:
    - "web_search"

agentic:
  react_max_steps: 8
  max_context_length: 32768
  max_tokens_hard_limit: 4096

debug:
  enabled: true
```

### Environment Variables

```bash
# Optional environment variables
export RAG_VECTOR_STORE_PATH="./memory/vector_store"
export RAG_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
export RAG_MIN_QUALITY="0.4"
export RAG_WORKERS="4"
```

### Performance Tuning

```yaml
# Adjust for your hardware
performance:
  cache_ttl: 300
  batch_size: 5

# Memory settings
memory:
  short_term_limit: 50    # Reduce for lower memory usage
  
# LLM settings
llm:
  intent_model:
    max_tokens: 1024      # Reduce for faster responses
    temperature: 0.1      # Lower for more deterministic responses
```

## 📁 Project Structure

```
autobot/
├── 📄 main.py                          # Entry point & CLI interface
├── 📁 core/                            # Core agent logic
│   ├── 🧠 react_orchestrator.py        # ReAct agent implementation
│   ├── 🤖 llm_interface.py             # Local LLM management
│   └── 📄 __init__.py
├── 📁 memory/                          # Memory & knowledge management
│   ├── 💾 memory_manager.py            # 3-tier memory system
│   ├── 📚 ingestion_pipeline.py        # Document processing
│   ├── 🔍 rag_pipeline.py              # Retrieval-augmented generation
│   ├── 📖 INGESTION_PIPELINE.md        # Detailed ingestion docs
│   ├── 📖 RAG_PIPELINE.md              # Detailed RAG docs
│   ├── 🗄️ long_term.db                 # Persistent memory
│   ├── 🗄️ short_term.db                # Session memory
│   └── 📁 vector_store/                # FAISS/ChromaDB index
├── 📁 tools/                           # Tool integrations
│   ├── 🔧 tool_registry.py             # Tool management
│   ├── 🔍 tool_detector.py             # Parse tool calls from LLM
│   └── 📁 web_search/                  # Web search implementation
│       ├── 🌐 search.py                # Main search pipeline
│       ├── ⚡ quick_scrape.py          # Search execution
│       └── 🧹 main_content_cleaner.py  # Content extraction
├── 📁 models/                          # LLM model management
│   ├── 📥 load-autobot-instruct.py     # Model loading utilities
│   ├── ⚙️ generate-autobot-instruct.py # Generation logic
│   └── 🤖 LFM2.5-1.2B-Instruct-Q5_K_M.gguf  # Model weights (downloaded)
├── 📁 config/                          # Configuration
│   └── ⚙️ settings.yaml                # Main configuration file
├── 📁 logs/                            # Application logs
│   └── 📄 autobot.log                  # Execution logs
└── 📄 requirements.txt                 # Python dependencies
```

## 🧩 Components

### 1. ReAct Orchestrator
- **Purpose**: Implements the Reason + Act pattern for intelligent decision making
- **Features**: Multi-step reasoning, tool integration, conversation management
- **Models**: Uses LFM2.5-1.2B-Instruct for reasoning and action decisions

### 2. LLM Interface
- **Purpose**: Manages local model loading and inference
- **Support**: GGUF via ctransformers (CPU) and transformers (GPU)
- **Features**: Chat templates, bfloat16 precision, streaming generation

### 3. Memory Manager
- **Short-term**: Session-level interactions (in-memory + SQLite)
- **Working**: Shared state for current reasoning loops
- **Long-term**: Persistent SQLite with semantic search via ChromaDB
- **Features**: Episodic memory, semantic memory, conversation history

### 4. Tool Registry
- **Current Tools**: Web search (DuckDuckGo-based)
- **Architecture**: Extensible framework for adding new tools
- **Features**: Automatic tool detection, parallel execution, retry logic

### 5. Document Ingestion Pipeline
- **Formats**: PDF, DOCX, HTML, CSV, Code files (Python, JS, Java), TXT, Markdown
- **Features**: Adaptive chunking, quality scoring, deduplication, parallel processing
- **Output**: FAISS vector index with metadata for RAG retrieval

### 6. RAG Pipeline
- **Search**: FAISS-based semantic similarity search
- **Features**: Metadata filtering, source deduplication, context building
- **Integration**: Works seamlessly with ReAct agent for knowledge retrieval

## 📚 API Reference

### ReActOrchestrator

```python
class ReActOrchestrator:
    async def initialize() -> bool
    async def handle_input(user_input: str) -> ReActResult
    async def get_conversation_history() -> List[Dict]
    async def clear_session_memory() -> bool
```

### Memory Manager

```python
class MemoryManager:
    async def store_interaction(user_input: str, response: str, intent: str)
    async def get_recent_interactions(limit: int = 10) -> List[Dict]
    async def search_memories(query: str, limit: int = 5) -> List[Dict]
    async def flush_short_to_long_term() -> Dict
```

### Tool Registry

```python
class ToolRegistry:
    async def execute_tool(tool_name: str, **kwargs) -> Dict
    def get_available_tools() -> List[Dict]
    def register_tool(name: str, function: Callable, schema: Dict)
```

### Document Ingestion

```python
class AdvancedIngestionPipeline:
    def ingest(input_dirs: List[str], max_workers: int = 4) -> Dict
    def process_file(file_path: str) -> int
    def get_stats() -> Dict
```

### RAG Pipeline

```python
class RAGPipeline:
    def query(query: str, top_k: int = 3, temperature: float = 0.1) -> Dict
    def retrieve_context(query: str, top_k: int = 5) -> Tuple[RAGContext, List[RAGResult]]
    def get_stats() -> Dict
```

## 🚀 Performance

### Typical Performance Metrics

- **Model Size**: 1.2B parameters (quantized to ~800MB)
- **Inference Speed**: 1-5 seconds per response
- **Memory Usage**: 4-8 GB (including model and search indices)
- **Search Latency**: 100-500ms for semantic search
- **Web Search**: 2-10 seconds depending on query complexity

### Optimization Tips

1. **GPU Acceleration**: Use CUDA for 3-5x faster inference
2. **Memory Management**: Adjust `short_term_limit` for memory constraints
3. **Parallel Processing**: Use `max_workers=4-8` for document ingestion
4. **Quality Filtering**: Set `min_quality_score=0.6+` for better results
5. **Context Length**: Reduce `max_context_length` for faster responses

### Hardware Requirements

**Minimum**:
- 8GB RAM
- 4-core CPU
- 2GB storage

**Recommended**:
- 16GB RAM
- 8-core CPU
- NVIDIA GPU with 4GB+ VRAM
- 10GB storage

**Optimal**:
- 32GB RAM
- 16-core CPU
- NVIDIA GPU with 8GB+ VRAM
- SSD storage

## 🔧 Development

### Adding Custom Tools

```python
# 1. Create tool function
async def my_custom_tool(param1: str, param2: int) -> Dict:
    # Your tool logic here
    return {"result": "success", "data": "..."}

# 2. Register tool
tool_schema = {
    "name": "my_custom_tool",
    "description": "Description of what the tool does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter 1"},
            "param2": {"type": "integer", "description": "Parameter 2"}
        },
        "required": ["param1", "param2"]
    }
}

# 3. Add to tool registry
tool_registry.register_tool("my_custom_tool", my_custom_tool, tool_schema)
```

### Extending Memory System

```python
# Custom memory backend
class CustomMemoryBackend:
    async def store(self, key: str, value: Any):
        # Custom storage logic
        pass
    
    async def retrieve(self, key: str) -> Any:
        # Custom retrieval logic
        pass

# Integrate with memory manager
memory_manager.add_backend("custom", CustomMemoryBackend())
```

### Model Integration

```python
# Support for custom models
class CustomLLMInterface(LLMInterface):
    def _load_model(self):
        # Load your custom model
        pass
    
    async def generate(self, messages: List[Dict]) -> str:
        # Custom generation logic
        pass
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/gajjalaashok75-UI/gakrai.git
cd gakrai

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
```

### Contribution Areas

- 🛠️ **Tool Development**: Add new tools (email, calendar, databases, APIs)
- 🧠 **Model Integration**: Support for new LLM models and providers
- 📚 **Document Formats**: Add support for new file formats
- 🔍 **Search Improvements**: Enhanced semantic search and ranking
- 🎨 **UI/UX**: Web interface, mobile app, desktop GUI
- 📊 **Analytics**: Usage metrics, performance monitoring
- 🔒 **Security**: Authentication, authorization, data privacy

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LFM2.5**: Local language model for reasoning and generation
- **FAISS**: Efficient similarity search and clustering
- **ChromaDB**: Vector database for semantic search
- **LangChain/LangGraph**: Agent orchestration framework
- **Transformers**: HuggingFace model integration
- **DuckDuckGo**: Privacy-focused web search

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/gajjalaashok75-UI/gakrai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gajjalaashok75-UI/gakrai/discussions)
- **Documentation**: See `/memory/` folder for detailed component docs

## 🗺️ Roadmap

- [ ] **Web Interface**: React-based web UI
- [ ] **API Server**: REST API for external integrations
- [ ] **Plugin System**: Dynamic tool loading
- [ ] **Multi-Modal**: Image and audio processing
- [ ] **Distributed**: Multi-agent collaboration
- [ ] **Cloud Integration**: AWS/Azure/GCP deployment
- [ ] **Mobile App**: iOS/Android applications

---

**AutoBot** - Intelligent automation through local AI reasoning and action. Built with ❤️ for developers and researchers. by Gakr team
