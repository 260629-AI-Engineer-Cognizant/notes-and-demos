# Vector Database Fundamentals

## Big Idea

A vector database stores embeddings and lets us search by similarity.

Traditional search asks:

```text
Which documents contain this exact word?
```

Semantic search asks:

```text
Which documents are closest in meaning to this question?
```

This is the foundation of Retrieval-Augmented Generation, or RAG.

---

## What Is a Vector?

A vector is a list of numbers.

Example:

```python
[0.12, -0.45, 0.88, 0.03]
```

In machine learning, vectors often represent objects such as:

- words
- sentences
- paragraphs
- images
- users
- products
- documents

A vector's length is called its dimension.

Example:

```python
[0.12, -0.45, 0.88, 0.03]
```

This is a 4-dimensional vector.

Embedding models often produce vectors with hundreds or thousands of dimensions.

---

## What Is an Embedding?

An embedding is a vector representation of something.

For text, an embedding model turns text into numbers.

Example:

```text
"CNNs are used for image classification."
```

becomes something like:

```python
[0.03, -0.18, 0.42, ..., 0.09]
```

The important idea is that similar meanings should create similar vectors.

Example:

```text
"CNNs are used for image classification."
"Convolutional neural networks work well for pictures."
```

These should be close together in vector space, even though they do not use the exact same words.

---

## Keyword Search vs Semantic Search

### Keyword Search

Keyword search finds exact or near-exact word matches.

Query:

```text
image model
```

Likely matches documents containing the words `image` and `model`.

Problem: it may miss documents that say:

```text
Convolutional neural networks classify pictures.
```

because the wording is different.

---

### Semantic Search

Semantic search compares meaning.

Query:

```text
What neural network should I use for pictures?
```

A semantic search system can retrieve:

```text
CNNs are commonly used for image classification.
```

because the query and document have similar meaning.

---

## How Similarity Search Works

Basic workflow:

```text
1. Store documents.
2. Convert each document into an embedding vector.
3. Store each vector with the document text and metadata.
4. Convert the user's query into an embedding vector.
5. Compare the query vector to stored vectors.
6. Return the closest matches.
```

---

## Distance Metrics

A distance metric tells us how close two vectors are.

### Cosine Similarity

Cosine similarity compares direction.

Useful when we care about meaning more than vector magnitude.

Simple interpretation:

```text
same direction = very similar
opposite direction = very different
```

Common for text embeddings.

---

### Euclidean Distance

Euclidean distance is straight-line distance.

Useful when actual distance in space matters.

Simple interpretation:

```text
smaller distance = more similar
larger distance = less similar
```

---

### Dot Product

Dot product compares both direction and magnitude.

It is common in some high-performance embedding search systems.

---

## What Does a Vector Database Store?

A vector database record usually has:

| Part | Example |
|---|---|
| ID | `doc-001` |
| Vector | `[0.1, -0.2, 0.4, ...]` |
| Document text | `CNNs are useful for image classification.` |
| Metadata | `{ "topic": "tensorflow", "week": 2 }` |

---

## Important Vector DB Terms

### Collection / Index

A group of vectors.

Chroma usually uses the term `collection`.

Pinecone usually uses the term `index`.

---

### Upsert

Upsert means:

```text
insert if it does not exist
update if it already exists
```

This is common when loading vectors into a vector database.

---

### Metadata

Metadata is extra information stored with a vector.

Example:

```python
{
    "topic": "vector databases",
    "week": 3,
    "difficulty": "intro"
}
```

Metadata allows filtered search.

Example:

```text
Find documents similar to this query, but only where topic = "tensorflow".
```

---

### Namespace / Partition

A namespace or partition separates groups of vectors.

Examples:

- one namespace per customer
- one namespace per project
- one namespace per course
- one namespace per environment, such as dev/test/prod

---

## Why Not Just Use SQL?

SQL databases are great for exact structured queries.

Example:

```sql
SELECT * FROM documents WHERE topic = 'tensorflow';
```

But SQL does not naturally answer:

```text
Which document is closest in meaning to this question?
```

You can manually store vectors in SQL and calculate similarity yourself, but vector databases are built to make this easier and faster.

---

## Common Vector DB Use Cases

| Use Case | Description |
|---|---|
| Semantic search | Search by meaning instead of exact words |
| RAG | Retrieve useful context before sending a prompt to an LLM |
| Recommendation systems | Find similar products, articles, or users |
| Duplicate detection | Find near-duplicate records |
| Image search | Search by visual similarity |
| Support chatbot | Retrieve relevant help docs |

---

## How This Leads Into RAG

RAG stands for Retrieval-Augmented Generation.

Basic RAG flow:

```text
User question
→ embed the question
→ retrieve similar documents from vector DB
→ send question + retrieved context to LLM
→ generate answer grounded in retrieved context
```

Week 3 focuses on the retrieval part. Week 4 builds on this with LangChain.

---

## Bite-Sized Version

A vector database stores embeddings, which are numeric representations of text, images, or other data. Instead of searching for exact keyword matches, vector databases search for similar meaning. They store IDs, vectors, documents, and metadata. Similarity is measured with metrics like cosine similarity, Euclidean distance, or dot product. This is the core retrieval system behind RAG.

---

## Interview Version

A vector database is designed to store and query high-dimensional embeddings. An embedding model converts data, such as text, into a numeric vector where similar meanings are close together. During semantic search, the query is embedded with the same model and compared against stored vectors using a distance metric like cosine similarity. Vector databases also store metadata so results can be filtered by source, topic, user, date, or other fields. This makes vector databases especially useful for RAG systems, semantic search, recommendations, and similarity-based retrieval.
