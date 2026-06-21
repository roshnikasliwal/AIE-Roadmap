---
title: "Choosing Your Vector Database: FAISS vs Qdrant vs ChromaDB vs pgvector"
date: 2026-03-10 09:00:00 +0530
categories: [AI, RAG]
tags: [vector-database, faiss, qdrant, chromadb, pgvector, python, rag-series]
mermaid: true
---

Every RAG system needs a vector store. The choice you make here has real implications for development speed, query performance, filtering capabilities, and operational complexity.

After using FAISS, Qdrant, and ChromaDB in production across different project scales, here's the honest breakdown — what each is actually good for, and when to use it.

## What You're Actually Choosing Between

The four most common options span a spectrum from "embedded library" to "full vector database":

```
FAISS          ChromaDB         pgvector         Qdrant
  |                |                |               |
Library        Embedded DB      SQL Extension    Dedicated DB
No persistence  Local/server     Postgres         Server/Cloud
No filtering    Basic filtering  SQL filtering    Rich filtering
Fastest ANN     Easy setup       SQL familiarity  Best production
```

## FAISS (Facebook AI Similarity Search)

FAISS is a C++ library for efficient similarity search. It's not a database — it's an index. No persistence, no filtering, no server. Just extremely fast nearest-neighbor search.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Your documents
documents = [
    "FastAPI is a modern, fast web framework for building APIs with Python",
    "Qdrant is a vector database for high-performance similarity search",
    "Docker containers package code and its dependencies together",
]

# Encode
embeddings = encoder.encode(documents, normalize_embeddings=True)
dimension = embeddings.shape[1]  # 384 for MiniLM

# Build index — IndexFlatIP = exact inner product search (cosine on normalized vectors)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype(np.float32))

# For large datasets, use approximate search (much faster, slightly less accurate)
# index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, n_clusters=100)
# index.train(embeddings.astype(np.float32))

# Search
query = "how to build REST APIs in Python"
query_embedding = encoder.encode([query], normalize_embeddings=True)
scores, indices = index.search(query_embedding.astype(np.float32), k=3)

for score, idx in zip(scores[0], indices[0]):
    print(f"Score: {score:.3f} | {documents[idx]}")

# Persist (FAISS-native format)
faiss.write_index(index, "my_index.faiss")
np.save("documents.npy", np.array(documents))

# Load
index = faiss.read_index("my_index.faiss")
documents = np.load("documents.npy")
```

**FAISS index types cheat sheet:**

| Index Type | Best For | Speed | Accuracy |
|---|---|---|---|
| `IndexFlatL2` / `IndexFlatIP` | < 100K vectors, exact search | Slow | Perfect |
| `IndexIVFFlat` | 100K–10M vectors | Fast | ~97% |
| `IndexHNSWFlat` | Low latency, in-memory | Very fast | ~99% |
| `IndexIVFPQ` | > 10M vectors, memory-constrained | Fast | ~90% |

**Use FAISS when:**
- You need maximum raw speed for similarity search
- Your dataset fits in memory (< ~10M vectors)
- You don't need metadata filtering
- You're building a feature that runs entirely in-process (e.g., a CLI tool, a batch job)

**Avoid FAISS when:**
- You need persistence across restarts without custom serialization
- You need to filter by metadata (category, date, author)
- You need to update/delete individual vectors (FAISS doesn't support this cleanly)

---

## ChromaDB

ChromaDB is the easiest path to a persisted, filterable vector store. It runs embedded (no server) or as a server. It's the right choice when you want to get something working quickly.

```python
import chromadb
from chromadb.utils import embedding_functions

# Embedded mode (no server needed — great for development)
client = chromadb.PersistentClient(path="./chroma_db")

# Or client mode (server)
# client = chromadb.HttpClient(host="localhost", port=8000)

# Use a built-in embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# Add documents with metadata
collection.add(
    documents=[
        "FastAPI is a modern web framework for building APIs",
        "Authentication tokens expire after 3600 seconds by default",
        "Docker containers include all dependencies for reproducible deployments",
    ],
    metadatas=[
        {"source": "docs", "category": "frameworks", "version": "0.110"},
        {"source": "docs", "category": "security", "version": "2.1"},
        {"source": "docs", "category": "devops", "version": "24.0"},
    ],
    ids=["doc_001", "doc_002", "doc_003"]
)

# Query with metadata filtering
results = collection.query(
    query_texts=["how long do auth tokens last"],
    n_results=3,
    where={"category": "security"},  # Filter by metadata
    where_document={"$contains": "token"},  # Filter by content
    include=["documents", "metadatas", "distances"]
)

# Update a document
collection.update(
    ids=["doc_002"],
    documents=["Authentication tokens expire after 3600 seconds (configurable via AUTH_TIMEOUT)"],
    metadatas=[{"source": "docs", "category": "security", "version": "2.2"}]
)

# Delete
collection.delete(ids=["doc_001"])

# Get collection stats
print(f"Total documents: {collection.count()}")
```

**Use ChromaDB when:**
- You're prototyping and want zero infrastructure overhead
- You need basic metadata filtering but not complex queries
- Your dataset is < 1M vectors
- You want embedded mode (no Docker, no server, just a local directory)

**Avoid ChromaDB when:**
- You need multi-tenancy with strict isolation
- You need to serve millions of queries per day (performance ceiling exists)
- You need advanced filtering with complex AND/OR/range conditions

---

## Qdrant

Qdrant is purpose-built for production vector search. It runs as a server (Docker or cloud), supports rich payload filtering, has built-in sparse vector support for hybrid search, and is designed to handle tens of millions of vectors.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SearchRequest, ScoredPoint
)
from sentence_transformers import SentenceTransformer
import uuid

encoder = SentenceTransformer("intfloat/e5-base-v2")
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 768  # e5-base-v2 dimension

# Create collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    # Optional: on-disk storage for large collections
    # on_disk_payload=True,
)

# Create payload index for fast filtering
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema="keyword",
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="version",
    field_schema="float",
)

# Ingest documents
documents = [
    {
        "id": str(uuid.uuid4()),
        "text": "Authentication tokens expire after 3600 seconds",
        "payload": {
            "source": "docs/auth.md",
            "category": "security",
            "version": 2.1,
            "language": "en",
            "last_updated": "2026-01-15",
        }
    },
    # ... more documents
]

# E5 models require a prefix for queries vs passages
def encode_passage(text: str) -> list:
    return encoder.encode(f"passage: {text}", normalize_embeddings=True).tolist()

def encode_query(text: str) -> list:
    return encoder.encode(f"query: {text}", normalize_embeddings=True).tolist()

points = [
    PointStruct(
        id=doc["id"],
        vector=encode_passage(doc["text"]),
        payload={**doc["payload"], "text": doc["text"]}
    )
    for doc in documents
]

client.upsert(collection_name=COLLECTION_NAME, points=points)

# --- Querying ---

# Basic semantic search
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=encode_query("how long do auth sessions last"),
    limit=5,
    with_payload=True,
)

# Search with metadata filter (AND condition)
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=encode_query("authentication configuration"),
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="security")),
            FieldCondition(key="version", range=Range(gte=2.0)),  # version >= 2.0
        ]
    ),
    limit=5,
    with_payload=True,
    score_threshold=0.6,  # Minimum similarity score
)

for result in results:
    print(f"Score: {result.score:.3f} | {result.payload['text'][:100]}")

# Delete by filter (e.g., remove outdated documents)
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=Filter(
        must=[FieldCondition(key="version", range=Range(lt=1.5))]
    )
)
```

### Hybrid Search (Sparse + Dense)

Qdrant supports hybrid search out of the box — combining dense semantic search with sparse keyword matching (BM25). This is critical for precise technical queries:

```python
from qdrant_client.models import SparseVector, NamedSparseVector, NamedVector

# Create collection with both dense and sparse vectors
client.recreate_collection(
    collection_name="hybrid_kb",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    }
)

# At query time, combine both scores with RRF (Reciprocal Rank Fusion)
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = client.query_points(
    collection_name="hybrid_kb",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(query=SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
)
```

**Use Qdrant when:**
- You're building for production (not a prototype)
- You need metadata filtering with complex conditions (range queries, multi-field)
- Your dataset exceeds 1M vectors
- You need multi-tenancy (separate collections per customer)
- You want hybrid search (sparse + dense)

---

## pgvector

If you're already running PostgreSQL, `pgvector` adds vector similarity search as a native extension. You get SQL queries, ACID transactions, and your existing database tooling — all in one place.

```python
from sqlalchemy import create_engine, Column, String, text
from sqlalchemy.orm import DeclarativeBase, Session
from pgvector.sqlalchemy import Vector

engine = create_engine("postgresql://user:pass@localhost/mydb")

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    content = Column(String)
    category = Column(String)
    version = Column(Float)
    embedding = Column(Vector(384))  # Dimension matches your model

Base.metadata.create_all(engine)

# Create IVFFlat index for fast approximate search
with engine.connect() as conn:
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """))
    conn.execute(text("SET ivfflat.probes = 10"))  # Higher = more accurate, slower

# Insert
with Session(engine) as session:
    session.add(Document(
        id="doc_001",
        content="Authentication tokens expire after 3600 seconds",
        category="security",
        version=2.1,
        embedding=encoder.encode("Authentication tokens expire after 3600 seconds").tolist()
    ))
    session.commit()

# Query with SQL filtering — this is pgvector's killer feature
with Session(engine) as session:
    query_embedding = encoder.encode("how long do auth tokens last").tolist()
    
    results = session.execute(text("""
        SELECT content, category, version,
               1 - (embedding <=> :query_embedding) AS similarity
        FROM documents
        WHERE category = 'security'
          AND version >= 2.0
          AND 1 - (embedding <=> :query_embedding) > 0.6
        ORDER BY embedding <=> :query_embedding
        LIMIT 5
    """), {"query_embedding": str(query_embedding)}).fetchall()
```

**Use pgvector when:**
- You're already on PostgreSQL and want to avoid another service
- Your queries naturally combine vector search with SQL conditions
- You need ACID guarantees (e.g., atomic updates of content + embedding together)
- Vector search is a secondary feature, not the primary data access pattern

---

## Side-by-Side Comparison

| | FAISS | ChromaDB | Qdrant | pgvector |
|---|---|---|---|---|
| **Setup** | pip install | pip install | Docker/Cloud | Postgres extension |
| **Persistence** | Manual (serialize/deserialize) | Built-in | Built-in | PostgreSQL |
| **Metadata filtering** | None | Basic | Rich (range, geo, nested) | Full SQL |
| **Max scale** | ~100M vectors (memory) | ~1M practical | 100M+ | Postgres limits |
| **Performance** | Fastest (in-memory) | Good | Excellent | Good (with index) |
| **Updates/deletes** | Rebuild index | Yes | Yes | Yes |
| **Hybrid search** | No | No | Yes (native) | Via tsvector |
| **Multi-tenancy** | No | Collections | Collections | Schemas/tables |
| **Best for** | Batch jobs, in-process search | Prototyping | Production RAG | Existing Postgres stack |

## The Decision Path

```
Do you need to run without a server? 
  Yes → FAISS (if no filtering) or ChromaDB (if you need filtering/persistence)
  
Are you prototyping? 
  Yes → ChromaDB

Are you already on PostgreSQL? 
  Yes → pgvector (if scale < 5M vectors)

Are you building for production with > 100K documents? 
  Yes → Qdrant

Do you need hybrid search (keyword + semantic)? 
  Yes → Qdrant (native sparse vectors)
```

## Key Takeaways

1. **ChromaDB for prototyping, Qdrant for production** — this covers 80% of use cases
2. **FAISS when you need maximum speed in-process** — taxonomy matching, batch similarity jobs
3. **pgvector when you're already on Postgres** — one less service to operate
4. **All vector stores need payload indexes** — create indexes on metadata fields you filter by, or filtering is a full scan
5. **Hybrid search matters for technical content** — keyword matching catches exact terms that semantic search misses

---

*Part of the [RAG Systems That Actually Work series]({{ site.baseurl }}/tags/rag-series/) — production lessons from building RAG pipelines on proprietary knowledge bases.*


## ## Vector Database Comparison

```mermaid
flowchart LR
    A[Choose Vector DB] --> B[FAISS
Facebook AI]
    A --> C[Qdrant
Rust-native]
    A --> D[ChromaDB
Embedded]
    A --> E[pgvector
PostgreSQL]
    B --> F[In-memory speed
Research/batch
No built-in filter]
    C --> G[Payload filtering
GRPC+REST
Production-grade]
    D --> H[Dev-friendly
Local or server
Python-first]
    E --> I[SQL + vectors
Existing PG infra
ACID transactions]
```
