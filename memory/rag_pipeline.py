#!/usr/bin/env python3
"""
RAG Pipeline for GAKR tiny_thinking Model (RAG-ready)
=====================================
Integrates with ingestion_pipeline.py storage
Provides: Search → Retrieve → Context → Generate

Storage Structure Used:
./vector_store/
├── index.faiss              # FAISS vector index
├── docstore.json            # Document contents & metadata
├── fingerprints.json        # File fingerprints
└── chunks/                  # Individual chunk files
"""

import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from fastapi import HTTPException

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import torch

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_SUPPORT = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMA_SUPPORT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Single retrieval result."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_file: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content[:300] + "..." if len(self.content) > 300 else self.content,
            'score': float(self.score),
            'metadata': self.metadata,
            'source': self.source_file
        }


@dataclass
class RAGContext:
    """Compiled context for generation."""
    query: str
    retrieved_results: List[RAGResult]
    context_text: str
    total_tokens: int
    sources: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'results_count': len(self.retrieved_results),
            'context_length': len(self.context_text),
            'total_tokens': self.total_tokens,
            'sources': list(set(self.sources))
        }


class VectorStoreRetriever:
    """
    Retrieves documents from FAISS index created by ingestion_pipeline.py
    """
    
    def __init__(
        self,
        store_path: str = "./vector_store",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        index_type: Optional[str] = None  # "faiss" or "chroma"
    ):
        self.store_path = Path(store_path)
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.faiss_id_to_doc_id: Dict[int, str] = {}
        self.meta: Dict[str, Any] = {}
        
        # Load embedding model (same as ingestion)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Load index and metadata
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and document store."""
        faiss_path = self.store_path / "index.faiss"
        docstore_path = self.store_path / "docstore.json"
        meta_path = self.store_path / "index_meta.json"

        # Auto-detect index type
        if not self.index_type:
            if faiss_path.exists() and docstore_path.exists():
                self.index_type = "faiss"
            elif CHROMA_SUPPORT:
                self.index_type = "chroma"
            else:
                raise FileNotFoundError(
                    f"No FAISS store at {faiss_path} / {docstore_path}, and Chroma is unavailable."
                )

        if self.index_type == "faiss":
            if not faiss_path.exists():
                raise FileNotFoundError(f"No FAISS index found at {faiss_path}. Run ingestion first!")
            if not docstore_path.exists():
                raise FileNotFoundError(f"No docstore found at {docstore_path}. Run ingestion first!")

            logger.info(f"Loading FAISS index from {faiss_path}")
            self.index = faiss.read_index(str(faiss_path))

            logger.info(f"Loading docstore from {docstore_path}")
            with open(docstore_path, 'r', encoding='utf-8') as f:
                self.docstore = json.load(f)

            self.id_list = []
            self.faiss_id_to_doc_id = {}
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.meta = json.load(f)
                    logger.info(f"Index loaded: {self.meta.get('total_chunks', 0)} chunks")
                    raw_id_map = self.meta.get("id_map", {})
                    if raw_id_map:
                        self.faiss_id_to_doc_id = {
                            int(faiss_id): chunk_id
                            for faiss_id, chunk_id in raw_id_map.items()
                        }

            if not self.faiss_id_to_doc_id:
                self.faiss_id_to_doc_id = self._build_fallback_id_map()
            
            self.id_list = [
                self.faiss_id_to_doc_id[faiss_id]
                for faiss_id in sorted(self.faiss_id_to_doc_id.keys())
            ]

            logger.info(
                f"Retriever ready with {len(self.docstore)} documents (FAISS), "
                f"{len(self.faiss_id_to_doc_id)} mapped vectors"
            )
            return

        # Chroma persistent
        if not CHROMA_SUPPORT:
            raise ImportError(
                "ChromaDB is not installed. Install chromadb or switch to FAISS index."
            )
        
        logger.info(f"Loading Chroma store from {self.store_path}")
        settings = Settings(anonymized_telemetry=False)
        self.chroma_client = chromadb.PersistentClient(path=str(self.store_path), settings=settings)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name="rag_chunks")
        # docstore is optional in chroma mode
        self.docstore = {}
        self.id_list = []
        logger.info("Retriever ready (Chroma)")
    
    def _build_fallback_id_map(self) -> Dict[int, str]:
        """
        Build FAISS-ID to doc-id mapping when metadata file is missing.
        Falls back to docstore insertion order.
        """
        doc_ids = list(self.docstore.keys())
        if not doc_ids:
            return {}
        
        try:
            if hasattr(self.index, "id_map"):
                index_ids = faiss.vector_to_array(self.index.id_map).tolist()
                if index_ids:
                    mapped: Dict[int, str] = {}
                    for position, faiss_id in enumerate(index_ids):
                        if position < len(doc_ids):
                            mapped[int(faiss_id)] = doc_ids[position]
                    if mapped:
                        logger.warning("Using fallback FAISS id mapping based on index/docstore order")
                        return mapped
        except Exception as e:
            logger.warning(f"Could not read FAISS id_map directly ({e}); falling back to sequential ids")
        
        return {i: doc_id for i, doc_id in enumerate(doc_ids)}
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filters: Optional[Dict] = None,
        dedupe_by_source: bool = False
    ) -> List[RAGResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filters: Optional metadata filters
        
        Returns:
            List of RAGResult objects
        """
        if top_k <= 0:
            return []
        
        candidate_k = max(top_k * 5, top_k)
        
        # Embed query
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype('float32')
        
        # Chroma search
        if self.index_type == "chroma":
            query_res = self.chroma_collection.query(
                query_embeddings=[query_embedding[0].tolist()],
                n_results=candidate_k
            )
            ids = query_res.get("ids", [[]])[0]
            docs = query_res.get("documents", [[]])[0]
            metas = query_res.get("metadatas", [[]])[0]
            dists = query_res.get("distances", [[]])[0]
            
            results_list: List[RAGResult] = []
            seen_sources = set()
            
            for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
                score = float(1.0 - dist)
                if score < score_threshold:
                    continue
                metadata = meta or {}
                if filters and not self._matches_filters(metadata, filters):
                    continue
                
                source = self._extract_source(metadata)
                if dedupe_by_source and source in seen_sources:
                    continue
                seen_sources.add(source)
                
                results_list.append(RAGResult(
                    id=doc_id,
                    content=doc_text,
                    score=score,
                    metadata=metadata,
                    source_file=source
                ))
                if len(results_list) >= top_k:
                    break
            
            results_list.sort(key=lambda x: x.score, reverse=True)
            logger.info(f"Search '{query[:50]}...' found {len(results_list)} results (Chroma)")
            return results_list[:top_k]

        # Search FAISS
        if getattr(self.index, "ntotal", 0) == 0:
            return []
        
        faiss_k = min(candidate_k, int(self.index.ntotal))
        distances, faiss_ids = self.index.search(query_embedding, k=faiss_k)
        
        results = []
        seen_sources = set()
        
        for faiss_id, distance in zip(faiss_ids[0], distances[0]):
            if faiss_id == -1:
                continue
            
            doc_id = self.faiss_id_to_doc_id.get(int(faiss_id))
            if not doc_id:
                continue
            doc_data = self.docstore.get(doc_id)
            if not doc_data:
                continue
            
            # Convert distance to similarity score (cosine similarity)
            # FAISS returns inner product for normalized vectors = cosine similarity
            score = float(distance)
            
            # Apply threshold
            if score < score_threshold:
                continue
            
            # Apply filters if specified
            if filters:
                metadata = doc_data.get('metadata', {})
                if not self._matches_filters(metadata, filters):
                    continue
            
            source = self._extract_source(doc_data.get('metadata', {}))
            if dedupe_by_source and source in seen_sources:
                continue
            seen_sources.add(source)
            
            result = RAGResult(
                id=doc_id,
                content=doc_data['content'],
                score=score,
                metadata=doc_data.get('metadata', {}),
                source_file=source
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Search '{query[:50]}...' found {len(results)} results")
        return results[:top_k]
    
    def _extract_source(self, metadata: Dict[str, Any]) -> str:
        """Resolve source file path from ingestion metadata schema."""
        return (
            metadata.get('source_document')
            or metadata.get('source_file')
            or 'unknown'
        )
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters (supports extra_metadata fallback)."""
        extra = metadata.get('extra_metadata', {})
        if not isinstance(extra, dict):
            extra = {}
        
        for key, value in filters.items():
            actual = metadata.get(key, extra.get(key))
            if actual != value:
                return False
        return True
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'total_documents': len(self.docstore),
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'embedding_model': self.embedding_model_name
        }


class RAGPromptBuilder:
    """
    Builds optimized prompts for tiny_thinking model with retrieved context.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 2048,
        context_template: Optional[str] = None
    ):
        self.max_context_tokens = max_context_tokens
        
        # Default context template
        self.context_template = context_template or """Use the following retrieved information to answer the user's question. If the answer is not in the context, say you don't have enough information.

{context}

---
User Question: {query}

Based on the above context, provide a detailed answer:"""
    
    def build_context(
        self,
        query: str,
        results: List[RAGResult],
        max_chunks: int = 3
    ) -> RAGContext:
        """
        Build context string from retrieved results.
        
        Args:
            query: Original query
            results: Retrieved documents
            max_chunks: Maximum chunks to include
        
        Returns:
            RAGContext object
        """
        if not results:
            return RAGContext(
                query=query,
                retrieved_results=[],
                context_text="No relevant information found.",
                total_tokens=0,
                sources=[]
            )
        
        # Build context parts
        context_parts = []
        sources = []
        total_length = 0
        
        for i, result in enumerate(results[:max_chunks], 1):
            # Format chunk
            chunk_text = f"[Source {i}: {Path(result.source_file).name}]\n{result.content}\n"
            
            # Check length limit (rough estimate: 1 token ≈ 4 chars)
            if total_length + len(chunk_text) > self.max_context_tokens * 4:
                break
            
            context_parts.append(chunk_text)
            sources.append(result.source_file)
            total_length += len(chunk_text)
        
        context_text = "\n".join(context_parts)
        full_prompt = self.context_template.format(
            context=context_text,
            query=query
        )
        
        # Estimate tokens (rough)
        estimated_tokens = len(full_prompt) // 4
        
        return RAGContext(
            query=query,
            retrieved_results=results,
            context_text=full_prompt,
            total_tokens=estimated_tokens,
            sources=sources
        )
    
    def build_chat_messages(
        self,
        query: str,
        context: RAGContext,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for tiny_thinking model.
        
        Args:
            query: User query
            context: RAG context
            system_prompt: Optional custom system prompt
        
        Returns:
            List of message dictionaries
        """
        default_system = """You are GAKR AI, a helpful assistant with access to a knowledge base. 
Answer questions based on the provided context. If the answer is not in the context, 
say "I don't have enough information in my knowledge base." Always cite your sources 
using [Source X] notation."""
        
        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": context.context_text}
        ]
        
        return messages


class RAGGenerator:
    """
    Generates responses using tiny_thinking model with retrieved context (LFM2.5-Thinking).
    """
    
    def __init__(
        self,
        model_name: str = "AshokGakr/tiny_thinking",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load tiny_thinking model and tokenizer (LFM2.5-Thinking)."""
        logger.info(f"Loading tiny_thinking model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=False,
            padding_side="left"
        )
        
        # Load model with optimizations
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": False,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        if self.device != "cuda" or load_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        top_p: float = 0.1,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Generate response from messages.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stream: Whether to stream output
        
        Returns:
            Generated text or streamer
        """
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except TypeError:
            # Fallback for models without chat template
            prompt = self._manual_format(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p if temperature > 0 else None,
            "temperature": temperature if temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
        }
        
        if stream:
            # Streaming generation
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer
            
            thread = Thread(target=self.model.generate, kwargs={
                **inputs,
                **gen_kwargs
            })
            thread.start()
            
            return streamer
        
        # Non-streaming generation
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _manual_format(self, messages: List[Dict[str, str]]) -> str:
        """Manual formatting for models without chat template."""
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            else:
                formatted.append(f"Assistant: {content}")
        return "\n\n".join(formatted) + "\n\nAssistant:"


class RAGPipeline:
    """
    Complete RAG pipeline: Retrieve → Build Context → Generate
    Integrates with ingestion_pipeline.py storage
    """
    
    def __init__(
        self,
        store_path: str = "./vector_store",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_model: str = "AshokGakr/tiny_thinking",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        logger.info("=" * 60)
        logger.info("INITIALIZING RAG PIPELINE")
        logger.info("=" * 60)
        
        # Initialize components
        self.retriever = VectorStoreRetriever(
            store_path=store_path,
            embedding_model=embedding_model
        )
        
        self.prompt_builder = RAGPromptBuilder()
        
        self.generator = RAGGenerator(
            model_name=llm_model,
            device=device
        )
        
        logger.info("RAG Pipeline ready")
        logger.info("=" * 60)
    
    def query(
        self,
        query: str,
        top_k: int = 3,
        max_context_chunks: int = 3,
        temperature: float = 0.1,
        filters: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            max_context_chunks: Max chunks to include in context
            temperature: Generation temperature
            filters: Optional metadata filters
            stream: Whether to stream response
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = datetime.now()
        
        # Step 1: Retrieve relevant documents
        logger.info(f"🔍 Retrieving documents for: {query[:50]}...")
        results = self.retriever.search(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        if not results:
            return {
                'query': query,
                'answer': "I don't have any relevant information in my knowledge base to answer this question.",
                'sources': [],
                'context': None,
                'metadata': {
                    'retrieved_count': 0,
                    'generation_time_ms': 0,
                    'total_time_ms': 0
                }
            }
        
        # Step 2: Build context
        logger.info(f"📚 Building context from {len(results)} results...")
        context = self.prompt_builder.build_context(
            query=query,
            results=results,
            max_chunks=max_context_chunks
        )
        
        # Step 3: Build messages
        messages = self.prompt_builder.build_chat_messages(query, context)
        
        # Step 4: Generate response
        logger.info(f"🤖 Generating response...")
        gen_start = datetime.now()
        
        if stream:
            # Return streamer for streaming
            streamer = self.generator.generate(
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            return {
                'query': query,
                'streamer': streamer,
                'context': context.to_dict(),
                'sources': [r.to_dict() for r in results]
            }
        
        # Non-streaming generation
        answer = self.generator.generate(
            messages=messages,
            temperature=temperature,
            stream=False
        )
        
        gen_time = (datetime.now() - gen_start).total_seconds() * 1000
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Compile response
        response = {
            'query': query,
            'answer': answer,
            'sources': [r.to_dict() for r in results],
            'context': context.to_dict(),
            'metadata': {
                'retrieved_count': len(results),
                'context_tokens': context.total_tokens,
                'generation_time_ms': round(gen_time, 2),
                'total_time_ms': round(total_time, 2),
                'model': self.generator.model_name
            }
        }
        
        logger.info(f"✅ Response generated in {total_time:.2f}ms")
        return response

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        max_context_chunks: int = 3,
        filters: Optional[Dict] = None
    ) -> Tuple[Optional[RAGContext], List[RAGResult]]:
        """Retrieve context only (no generation)."""
        results = self.retriever.search(
            query=query,
            top_k=top_k,
            filters=filters
        )
        if not results:
            return None, []
        context = self.prompt_builder.build_context(
            query=query,
            results=results,
            max_chunks=max_context_chunks
        )
        return context, results
    
    def query_stream(self, query: str, **kwargs):
        """Stream RAG response."""
        result = self.query(query, stream=True, **kwargs)
        
        # Stream tokens
        streamer = result['streamer']
        for token in streamer:
            yield token
        
        # Final metadata
        yield json.dumps({
            'type': 'metadata',
            'data': result['context']
        })
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'retriever': self.retriever.get_stats(),
            'generator': {
                'model': self.generator.model_name,
                'device': str(self.generator.device)
            }
        }


# ================== FASTAPI INTEGRATION =================

def create_rag_endpoints(app, store_path: str = "./vector_store"):
    """
    Add RAG endpoints to existing FastAPI app (from app.py)
    
    Usage in app.py:
        from rag_pipeline import create_rag_endpoints
        create_rag_endpoints(app, store_path="./vector_store")
    """
    
    # Initialize pipeline once
    rag_pipeline = RAGPipeline(store_path=store_path)
    
    @app.post("/api/rag/query")
    async def rag_query(
        query: str,
        top_k: int = 3,
        temperature: float = 0.1,
        filters: Optional[str] = None  # JSON string
    ):
        """
        RAG query endpoint - retrieves from knowledge base and generates answer
        """
        if not query:
            raise HTTPException(400, "Query is required")
        
        # Parse filters if provided
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except:
                pass
        
        try:
            result = rag_pipeline.query(
                query=query,
                top_k=top_k,
                temperature=temperature,
                filters=filter_dict,
                stream=False
            )
            return result
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise HTTPException(500, f"RAG processing failed: {str(e)}")
    
    @app.post("/api/rag/search")
    async def rag_search(
        query: str,
        top_k: int = 5,
        filters: Optional[str] = None
    ):
        """
        Search only endpoint - returns retrieved documents without generation
        """
        if not query:
            raise HTTPException(400, "Query is required")
        
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except:
                pass
        
        try:
            results = rag_pipeline.retriever.search(
                query=query,
                top_k=top_k,
                filters=filter_dict
            )
            return {
                'query': query,
                'results': [r.to_dict() for r in results],
                'count': len(results)
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(500, f"Search failed: {str(e)}")
    
    @app.get("/api/rag/stats")
    async def rag_stats():
        """Get RAG pipeline statistics"""
        return rag_pipeline.get_stats()
    
    @app.get("/api/rag/sources")
    async def rag_sources():
        """List all indexed sources"""
        sources = {}
        for doc_id, doc_data in rag_pipeline.retriever.docstore.items():
            source = doc_data.get('metadata', {}).get('source_file', 'unknown')
            if source not in sources:
                sources[source] = {
                    'chunks': 0,
                    'file_type': doc_data.get('metadata', {}).get('file_type', 'unknown')
                }
            sources[source]['chunks'] += 1
        
        return {
            'total_sources': len(sources),
            'total_chunks': len(rag_pipeline.retriever.docstore),
            'sources': sources
        }
    
    logger.info("RAG endpoints registered")


# ================== STANDALONE USAGE =================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Pipeline CLI')
    parser.add_argument('--store', '-s', default='./vector_store',
                       help='Path to vector store')
    parser.add_argument('--query', '-q', help='Query to process')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--search-only', action='store_true',
                       help='Only search, no generation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(store_path=args.store)
    
    if args.query:
        # Single query
        if args.search_only:
            results = pipeline.retriever.search(args.query)
            print(f"\nSearch results for: '{args.query}'")
            print("=" * 60)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. Score: {r.score:.4f} | Source: {r.source_file}")
                print(f"   {r.content[:200]}...")
        else:
            result = pipeline.query(args.query)
            print(f"\nQuery: {result['query']}")
            print(f"Answer: {result['answer']}")
            print(f"\nSources: {[s['source'] for s in result['sources']]}")
            print(f"Time: {result['metadata']['total_time_ms']}ms")
    
    elif args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("RAG PIPELINE - INTERACTIVE MODE")
        print("Commands: /search <query> | /stats | /sources | quit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'quit':
                    break
                
                # Commands
                if user_input.startswith('/search '):
                    query = user_input[8:]
                    results = pipeline.retriever.search(query)
                    print(f"\nFound {len(results)} results:")
                    for r in results:
                        print(f"  - {r.source_file} (score: {r.score:.3f})")
                
                elif user_input == '/stats':
                    stats = pipeline.get_stats()
                    print(json.dumps(stats, indent=2))
                
                elif user_input == '/sources':
                    sources = set()
                    for doc_data in pipeline.retriever.docstore.values():
                        sources.add(doc_data.get('metadata', {}).get('source_file', 'unknown'))
                    print(f"\nIndexed sources ({len(sources)}):")
                    for s in sorted(sources):
                        print(f"  - {s}")
                
                else:
                    # Regular RAG query
                    result = pipeline.query(user_input)
                    print(f"\n🤖 {result['answer']}")
                    print(f"\n📚 Sources: {', '.join([s['source'] for s in result['sources']])}")
                    print(f"⏱️  {result['metadata']['total_time_ms']}ms")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")
    
    else:
        parser.print_help()
