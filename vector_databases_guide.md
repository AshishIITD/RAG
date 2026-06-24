# Vector Databases Guide

This document explains how vector databases calculate similarity and provides an overview of the most famous options available.

## Distance Metrics in ChromaDB

By default, **ChromaDB uses L2 (Squared Euclidean Distance)**. However, it is not limited to L2! 

ChromaDB supports three mathematical metrics to measure distance:
1. **L2 (Squared Euclidean)**: The default. Good for general distance measuring.
2. **Cosine (`cosine`)**: Excellent for text and documents, as it measures the angle between vectors (how "directionally" similar the topics are) rather than their strict distance.
3. **Inner Product (`ip`)**: Useful when your embeddings are normalized (which just means the math is simplified to make searches slightly faster).

### How to use Cosine Similarity in ChromaDB
To force Chroma to use Cosine Similarity instead of L2, simply add the `collection_metadata` parameter when creating your database:

```python
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings_model,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"} # Explicitly tells Chroma to use Cosine
)
```

---

## Famous Vector Databases

In the AI world, vector databases are generally split into three categories based on how they are hosted and used:

### 1. Local & Lightweight (Great for Beginners and Prototyping)
These run directly on your machine without needing the internet, just like ChromaDB.
*   **FAISS (by Meta/Facebook):** This isn't technically a full "database", but rather an extremely fast math library for searching vectors. It lacks database features like storing metadata or easy updating, but it is incredibly fast.
*   **LanceDB:** Think of this as the "SQLite" of vector databases. It saves data directly to files on your hard drive and is extremely fast for local projects.
*   **Qdrant (Local mode):** Built in the Rust programming language, it is blazing fast and has great features. You can run it locally or host it in the cloud.

### 2. Cloud Managed (Great for Production and Enterprise)
These are servers hosted by a company. You don't manage the hard drives; you just send them your data over the internet.
*   **Pinecone:** Probably the most famous cloud vector database. It is fully managed, meaning zero setup, but it operates purely in the cloud (no local offline version). It's widely used in enterprise AI.
*   **Weaviate:** A very powerful open-source database that you can self-host or use via their cloud. It has advanced features built-in specifically for AI.
*   **Milvus:** Built for absolutely massive scale (think billions of vectors). It is very powerful but can be complex to set up.

### 3. Traditional Databases with "Vector Upgrades"
*   **PostgreSQL (with `pgvector`):** This is becoming incredibly popular. If a company is already using Postgres for their normal website data (like usernames and passwords), they can just install the `pgvector` plugin and use their existing database for AI vectors too!
*   **MongoDB Atlas:** Similar to Postgres, MongoDB recently added vector search so companies don't have to buy a separate database like Pinecone.
