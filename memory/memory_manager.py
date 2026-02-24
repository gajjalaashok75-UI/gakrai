"""
Memory Manager for AutoBot
Handles short-term, working, and long-term memory.
"""

import asyncio
import json
import logging
import sqlite3
import time
from typing import Dict, Any, List, Optional

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False

class MemoryManager:
    """Manages all memory operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.short_term: List[str] = []
        self.working_memory: Dict[str, Any] = {}

        self.db_path = config['memory'].get('long_term_db', './memory/long_term.db')
        self.short_term_db_path = config['memory'].get('short_term_db', './memory/short_term.db')
        self.vector_path = config['memory'].get('vector_store', './memory/vector_store')

        self.db_conn = None
        self.short_db_conn = None
        self.vector_client = None
        self.collection = None
        self._working_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize memory stores."""
        # Long-term DB
        self.db_conn = sqlite3.connect(self.db_path)
        self._create_tables()

        # Vector store
        if CHROMADB_AVAILABLE:
            try:
                self.vector_client = chromadb.PersistentClient(path=self.vector_path)
                self.collection = self.vector_client.get_or_create_collection("memories")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")
                self.vector_client = None
        else:
            self.logger.warning("ChromaDB not available, vector memory disabled")

        # Short-term DB (for session-level interactions)
        try:
            self.short_db_conn = sqlite3.connect(self.short_term_db_path)
            self._create_short_term_tables()
        except Exception as exc:
            self.logger.warning("Failed to initialize short-term DB: %s", exc)

    def _create_tables(self):
        """Create database tables."""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                timestamp REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                user_input TEXT,
                response TEXT,
                intent TEXT,
                timestamp REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id INTEGER PRIMARY KEY,
                summary TEXT,
                source_user_input TEXT,
                source_response TEXT,
                intent TEXT,
                metadata_json TEXT,
                timestamp REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id INTEGER PRIMARY KEY,
                fact_key TEXT,
                fact_value TEXT,
                source TEXT,
                confidence REAL,
                metadata_json TEXT,
                created_at REAL,
                updated_at REAL,
                UNIQUE(fact_key, fact_value)
            )
        ''')
        self.db_conn.commit()

    def _create_short_term_tables(self):
        """Create tables for short-term DB."""
        cursor = self.short_db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                user_input TEXT,
                response TEXT,
                intent TEXT,
                timestamp REAL
            )
        ''')
        self.short_db_conn.commit()

    async def add_short_term(self, item: str):
        """Add to short-term memory."""
        # Keep in-memory list for quick access
        self.short_term.append(item)
        if len(self.short_term) > self.config['memory']['short_term_limit']:
            self.short_term.pop(0)

    async def add_short_term_interaction(self, user_input: str, response: str, intent: str = ""):
        """Store a short-term interaction (user query + final answer).

        This persists into `short_term_db` and keeps in-memory list.
        """
        ts = time.time()
        # Append to in-memory list as a dict for quick access
        self.short_term.append({'user_input': user_input, 'response': response, 'timestamp': ts})

        try:
            if self.short_db_conn:
                cursor = self.short_db_conn.cursor()
                cursor.execute(
                    "INSERT INTO interactions (user_input, response, intent, timestamp) VALUES (?, ?, ?, ?)",
                    (user_input, response, intent, ts),
                )
                self.short_db_conn.commit()
        except Exception as exc:
            self.logger.warning("Failed to write short-term interaction: %s", exc)

    async def add_interaction(self, user_input: str, response: str, intent: str):
        """Store interaction in long-term memory."""
        timestamp = time.time()

        cursor = self.db_conn.cursor()
        cursor.execute(
            "INSERT INTO interactions (user_input, response, intent, timestamp) VALUES (?, ?, ?, ?)",
            (user_input, response, intent, timestamp)
        )
        self.db_conn.commit()

        # Add to vector store for semantic search
        if self.collection:
            self.collection.add(
                documents=[f"User: {user_input}\nAutoBot: {response}"],
                metadatas=[{"intent": intent, "timestamp": timestamp, "memory_type": "interaction"}],
                ids=[f"interaction_{timestamp}"]
            )

    async def recall(self, query: str, limit: int = 5) -> List[str]:
        """Recall relevant memories."""
        if self.collection:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            return results['documents'][0] if results['documents'] else []
        else:
            return []

    async def get_short_term_interactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent short-term interactions from short-term DB (most recent first)."""
        if not self.short_db_conn:
            return []
        cursor = self.short_db_conn.cursor()
        cursor.execute(
            "SELECT user_input, response, intent, timestamp FROM interactions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        return [
            {"user_input": r[0], "response": r[1], "intent": r[2], "timestamp": r[3]} for r in rows
        ]

    async def flush_short_to_long_term(self):
        """Append all short-term interactions to the long-term DB and clear the short-term DB."""
        if not self.short_db_conn:
            return {"flushed": False, "reason": "no_short_db"}

        try:
            short_cursor = self.short_db_conn.cursor()
            short_cursor.execute("SELECT user_input, response, intent, timestamp FROM interactions ORDER BY timestamp ASC")
            rows = short_cursor.fetchall()
            if not rows:
                return {"flushed": True, "count": 0}

            cursor = self.db_conn.cursor()
            for r in rows:
                cursor.execute(
                    "INSERT INTO interactions (user_input, response, intent, timestamp) VALUES (?, ?, ?, ?)",
                    (r[0], r[1], r[2] or "", r[3]),
                )
            self.db_conn.commit()

            # Clear short-term DB
            short_cursor.execute("DELETE FROM interactions")
            self.short_db_conn.commit()

            # Clear in-memory short_term list
            self.short_term = []
            return {"flushed": True, "count": len(rows)}
        except Exception as exc:
            self.logger.exception("Failed flushing short-term to long-term: %s", exc)
            return {"flushed": False, "error": str(exc)}

    async def get_short_term(self, limit: int = 10) -> List[str]:
        """Return the most recent short-term items."""
        if limit <= 0:
            return []
        return self.short_term[-limit:]

    async def set_working_memory(self, key: str, value: Any):
        """Set a key in shared working memory."""
        async with self._working_lock:
            self.working_memory[key] = value

    async def get_working_memory(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a key from shared working memory."""
        return self.working_memory.get(key, default)

    async def append_working_memory(self, key: str, value: Any, max_items: int = 100) -> List[Any]:
        """Append a value to a list field in working memory."""
        async with self._working_lock:
            existing = self.working_memory.get(key)
            if not isinstance(existing, list):
                existing = []
            existing.append(value)
            if max_items > 0 and len(existing) > max_items:
                existing = existing[-max_items:]
            self.working_memory[key] = existing
            return existing

    async def clear_working_memory(self, key: Optional[str] = None):
        """Clear one key or the full working memory."""
        async with self._working_lock:
            if key is None:
                self.working_memory.clear()
            else:
                self.working_memory.pop(key, None)

    async def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions from long-term storage."""
        if not self.db_conn or limit <= 0:
            return []

        cursor = self.db_conn.cursor()
        cursor.execute(
            """
            SELECT user_input, response, intent, timestamp
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        rows.reverse()
        return [
            {
                "user_input": row[0],
                "response": row[1],
                "intent": row[2],
                "timestamp": row[3],
            }
            for row in rows
        ]

    async def add_episodic_memory(
        self,
        summary: str,
        source_user_input: str,
        source_response: str,
        intent: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store episodic memory summary (chronological conversation memory).
        Inspired by arc-files/08_episodic_with_semantic.ipynb.
        """
        if not summary:
            return {"stored": False, "reason": "empty_summary"}

        ts = time.time()
        metadata = metadata or {}

        cursor = self.db_conn.cursor()
        cursor.execute(
            """
            INSERT INTO episodic_memories
            (summary, source_user_input, source_response, intent, metadata_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                summary,
                source_user_input,
                source_response,
                intent,
                json.dumps(metadata, default=str),
                ts,
            ),
        )
        episodic_id = cursor.lastrowid
        self.db_conn.commit()

        if self.collection:
            try:
                self.collection.add(
                    documents=[summary],
                    metadatas=[
                        {
                            "intent": intent,
                            "timestamp": ts,
                            "memory_type": "episodic",
                            "episodic_id": episodic_id,
                        }
                    ],
                    ids=[f"episodic_{episodic_id}_{int(ts * 1000)}"],
                )
            except Exception as exc:
                self.logger.warning(f"Failed to add episodic memory to vector store: {exc}")

        return {"stored": True, "episodic_id": episodic_id, "timestamp": ts}

    async def add_semantic_memories(
        self,
        facts: List[Dict[str, Any]],
        source: str = "conversation",
    ) -> Dict[str, Any]:
        """
        Store semantic memory facts as normalized key-value records.
        Upserts on (fact_key, fact_value).
        """
        if not facts:
            return {"stored": 0, "updated": 0}

        stored = 0
        updated = 0
        now = time.time()
        cursor = self.db_conn.cursor()

        for fact in facts:
            if not isinstance(fact, dict):
                continue

            fact_key = str(fact.get("key", "")).strip()
            fact_value = str(fact.get("value", "")).strip()
            try:
                confidence = float(fact.get("confidence", 0.7))
            except Exception:
                confidence = 0.7
            metadata = fact.get("metadata", {})
            if not fact_key or not fact_value:
                continue

            cursor.execute(
                """
                SELECT id FROM semantic_memories
                WHERE fact_key = ? AND fact_value = ?
                """,
                (fact_key, fact_value),
            )
            row = cursor.fetchone()

            if row:
                cursor.execute(
                    """
                    UPDATE semantic_memories
                    SET source = ?, confidence = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        source,
                        confidence,
                        json.dumps(metadata, default=str),
                        now,
                        row[0],
                    ),
                )
                updated += 1
            else:
                cursor.execute(
                    """
                    INSERT INTO semantic_memories
                    (fact_key, fact_value, source, confidence, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact_key,
                        fact_value,
                        source,
                        confidence,
                        json.dumps(metadata, default=str),
                        now,
                        now,
                    ),
                )
                stored += 1

                if self.collection:
                    try:
                        semantic_doc = f"{fact_key}: {fact_value}"
                        self.collection.add(
                            documents=[semantic_doc],
                            metadatas=[
                                {
                                    "memory_type": "semantic",
                                    "fact_key": fact_key[:120],
                                    "timestamp": now,
                                    "confidence": confidence,
                                }
                            ],
                            ids=[f"semantic_{cursor.lastrowid}_{int(now * 1000)}"],
                        )
                    except Exception as exc:
                        self.logger.warning(f"Failed to add semantic memory to vector store: {exc}")

        self.db_conn.commit()
        return {"stored": stored, "updated": updated}

    async def recall_episodic(self, query: str, limit: int = 5) -> List[str]:
        """Recall episodic memory from vector store first, then SQL fallback."""
        if limit <= 0:
            return []

        if self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=max(1, limit * 2),
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                episodic_docs: List[str] = []
                for doc, meta in zip(docs, metas):
                    if isinstance(meta, dict) and meta.get("memory_type") == "episodic":
                        episodic_docs.append(doc)
                    if len(episodic_docs) >= limit:
                        break
                if episodic_docs:
                    return episodic_docs
            except Exception as exc:
                self.logger.warning(f"Episodic recall vector query failed: {exc}")

        cursor = self.db_conn.cursor()
        cursor.execute(
            """
            SELECT summary
            FROM episodic_memories
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [row[0] for row in cursor.fetchall()]

    async def recall_semantic(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Recall semantic facts using token-based LIKE matching."""
        if limit <= 0:
            return []

        tokens = [tok for tok in query.lower().split() if len(tok) > 2][:6]
        if not tokens:
            tokens = [query.lower()[:40]]

        where_parts = []
        params: List[Any] = []
        for token in tokens:
            where_parts.append("(LOWER(fact_key) LIKE ? OR LOWER(fact_value) LIKE ?)")
            like_token = f"%{token}%"
            params.extend([like_token, like_token])

        where_sql = " OR ".join(where_parts)
        sql = f"""
            SELECT fact_key, fact_value, source, confidence, metadata_json, updated_at
            FROM semantic_memories
            WHERE {where_sql}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = self.db_conn.cursor()
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            metadata = {}
            if row[4]:
                try:
                    metadata = json.loads(row[4])
                except Exception:
                    metadata = {"raw_metadata": row[4]}
            results.append(
                {
                    "key": row[0],
                    "value": row[1],
                    "source": row[2],
                    "confidence": row[3],
                    "metadata": metadata,
                    "updated_at": row[5],
                }
            )
        return results

    async def recall_episodic_with_semantic(
        self, query: str, episodic_limit: int = 3, semantic_limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve combined episodic + semantic memory context.
        """
        episodic = await self.recall_episodic(query, episodic_limit)
        semantic = await self.recall_semantic(query, semantic_limit)
        return {
            "query": query,
            "episodic": episodic,
            "semantic": semantic,
            "episodic_count": len(episodic),
            "semantic_count": len(semantic),
        }

    async def shutdown(self):
        """Clean shutdown."""
        if self.db_conn:
            self.db_conn.close()
