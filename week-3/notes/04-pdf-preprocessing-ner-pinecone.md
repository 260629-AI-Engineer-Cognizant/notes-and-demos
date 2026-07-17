# PDF Preprocessing, NER, and Pinecone Semantic Search

## The Complete Retrieval Pipeline

A vector database cannot directly search a raw PDF. The document must first be transformed into records that the database understands.

```text
PDF
  -> extracted page text
  -> cleaned text
  -> overlapping chunks
  -> NER and source metadata
  -> embedding vectors
  -> Pinecone records
  -> semantic search results
```

Each transformation has a different responsibility:

| Stage | Responsibility |
|---|---|
| PDF extraction | Read the text contained in each PDF page |
| Cleaning | Remove formatting noise and normalize text |
| Chunking | Create smaller searchable units |
| NER | Identify named people, organizations, and locations |
| Embedding | Represent the meaning of each chunk numerically |
| Pinecone upsert | Store vectors, IDs, and metadata |
| Semantic query | Retrieve chunks whose meanings resemble the question |

---

## Data Preprocessing for a PDF

### Why preprocessing is necessary

PDF files are designed to preserve page layout, not to provide clean application data. Extracted text can contain:

- words split across line breaks;
- repeated spaces and newlines;
- page headers and footers;
- empty pages;
- unexpected reading order;
- tables represented as disconnected text; and
- image-only pages with no extractable text.

A PDF parser such as `pypdf` can extract an existing text layer. It does not automatically perform Optical Character Recognition (OCR). A scanned document may need an OCR service or library before it can be chunked.

### Preserve source information early

Do not discard the page number while extracting text.

```python
pages.append({
    "page_number": page_number,
    "raw_text": page.extract_text() or "",
})
```

Page numbers later help us:

- display where a result came from;
- verify whether retrieval is correct;
- filter searches;
- create citations for a RAG answer; and
- find the original source passage.

### Cleaning example

```python
def clean_pdf_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

Cleaning rules should be based on the actual source. A rule that helps one PDF can damage another PDF, so inspect samples before processing the entire dataset.

---

## Chunking

### What is a chunk?

A chunk is a smaller passage created from a longer document. Each chunk becomes an independently searchable vector record.

Instead of storing one vector for a 300-page handbook, we store many vectors:

```text
Handbook
  -> page 1, chunk 1
  -> page 1, chunk 2
  -> page 2, chunk 1
  -> ...
```

### Why not embed the entire PDF once?

A single vector would compress all topics in the handbook into one general representation. A search could identify the handbook as relevant, but it would not identify the passage that answers the question.

Embedding models also have input-length limits. Text beyond a model's limit may be truncated or rejected.

### Chunk-size tradeoff

| Chunk size | Advantage | Disadvantage |
|---|---|---|
| Too small | Precise matches | May lose surrounding explanation |
| Too large | Preserves more context | Can mix unrelated topics and weaken relevance |
| Balanced | Contains one coherent passage | Requires testing for the specific dataset |

A useful starting point for this lesson is:

```python
chunk_size = 900
chunk_overlap = 150
```

These are character counts, not token counts.

### Use a text-splitting library

We need to understand the chunking decisions, but we do not need to write the boundary algorithm from scratch. The notebook uses LangChain's standalone text-splitter package:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    add_start_index=True,
)

page_chunks = text_splitter.create_documents(
    texts=[page["text"]],
    metadatas=[{"page_number": page["page_number"]}],
)
```

`RecursiveCharacterTextSplitter` tries the separators in order. It first attempts to preserve larger units and moves toward smaller boundaries only when the text is still too large.

The library abstracts the loop, but we still own the important decisions:

- chunk size;
- overlap;
- separator order;
- source metadata;
- inspection of the resulting chunks; and
- evaluation of search quality.

### Why use overlap?

Without overlap, a sentence or explanation near a boundary can be separated from important context.

```text
Chunk 1: ...requirements are documented and reviewed during
Chunk 2: validation to confirm they represent stakeholder needs...
```

Overlap repeats part of the previous chunk:

```text
Chunk 1: ...requirements are documented and reviewed during validation...
Chunk 2: ...reviewed during validation to confirm they represent stakeholder needs...
```

The repeated text gives both chunks enough surrounding information to remain meaningful.

### Stable chunk IDs

An ID should identify the document, page, and chunk position:

```text
nasa-se-p0042-c003
```

Stable IDs are important because Pinecone upsert replaces a record when the same ID is used again.

### Chunk record before embedding

```python
{
    "id": "nasa-se-p0042-c003",
    "document_id": "nasa-systems-engineering-handbook-rev2",
    "source": "NASA Systems Engineering Handbook, Revision 2",
    "page_number": 42,
    "chunk_number": 3,
    "text": "The passage contained in this chunk..."
}
```

---

## Named Entity Recognition (NER)

### What NER does

Named Entity Recognition identifies spans of text that represent named items.

Example:

```text
NASA conducted testing at Johnson Space Center.
```

Possible entities:

| Text | Entity label |
|---|---|
| NASA | Organization (`ORG`) |
| Johnson Space Center | Facility or organization (`FAC` or `ORG`) |

Common spaCy entity labels include:

| Label | Meaning |
|---|---|
| `PERSON` | People |
| `ORG` | Companies, agencies, and institutions |
| `GPE` | Countries, states, and cities |
| `LOC` | Non-governmental locations |
| `FAC` | Facilities, buildings, and infrastructure |
| `PRODUCT` | Named products |
| `EVENT` | Named events |
| `DATE` | Dates and time periods |

### NER is not semantic search

NER and embeddings solve different problems:

| NER | Embeddings |
|---|---|
| Extracts named, structured items | Represents overall semantic meaning |
| Produces labels and text values | Produces numeric vectors |
| Useful for metadata and filtering | Useful for similarity ranking |
| Can identify `NASA` as an organization | Can connect “space agency” with a NASA passage |

### NER helper function

The notebook's helper:

1. receives a processed spaCy document;
2. maps selected spaCy labels to application metadata fields;
3. removes surrounding punctuation and whitespace;
4. deduplicates repeated entities;
5. limits the number of stored values; and
6. omits empty fields.

```python
NER_FIELD_MAP = {
    "PERSON": "people",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "FAC": "locations",
    "PRODUCT": "named_items",
    "EVENT": "named_items",
    "WORK_OF_ART": "named_items",
}
```

Example output:

```python
{
    "organizations": ["NASA"],
    "locations": ["Johnson Space Center"],
    "named_items": ["Apollo Program"]
}
```

### NER limitations

NER is model-based and imperfect.

- An entity can receive the wrong label.
- A domain-specific term may not be recognized.
- The same entity can appear under different names.
- Generic models may not understand specialized engineering concepts.

NER output should be inspected and cleaned before being treated as reliable metadata.

---

## Embedding Models

### What an embedding is

An embedding is a fixed-length vector that represents learned properties of an item.

```python
[0.018, -0.224, 0.091, ..., 0.037]
```

For text embeddings, semantically similar passages should have vectors that are close together.

```text
"requirements validation checks stakeholder needs"

should be closer to

"confirm that requirements represent what stakeholders expect"

than to

"the spacecraft uses a thermal protection system"
```

### Embedding-model approaches

| Approach | Example | Advantages | Tradeoffs |
|---|---|---|---|
| Small local sentence model | `all-MiniLM-L6-v2` | Fast, lightweight, no inference API | Less powerful on difficult or specialized retrieval |
| Larger local sentence model | `all-mpnet-base-v2` | Strong general semantic similarity | Slower, larger, produces 768-dimensional vectors |
| Hosted embedding API | Provider-hosted embedding model | Managed infrastructure and scaling | Credentials, network requests, and usage management |
| Pinecone integrated embedding | Pinecone-hosted embedding model | Can upsert and search using source text | Hides the explicit vector-generation stage |

### Why use `all-MiniLM-L6-v2` here?

It provides a practical classroom balance:

- available through Sentence Transformers;
- runs in Colab without a paid inference service;
- fast enough for hundreds or thousands of chunks;
- produces 384-dimensional embeddings; and
- works well for a foundational semantic-search demonstration.

### Dimension must match

If the embedding model produces 384 numbers, the Pinecone index must have dimension 384.

```python
VECTOR_DIMENSION = embeddings.shape[1]
```

A 384-dimensional vector cannot be inserted into an index created for 768-dimensional vectors.

Changing embedding models usually requires creating a compatible index and re-embedding all stored text.

### Use the same model for documents and queries

Documents and queries must be represented in the same vector space.

```python
document_vectors = model.encode(document_chunks)
query_vector = model.encode(question)
```

Using different unrelated embedding models would make the coordinates incomparable.

### Normalized embeddings

```python
model.encode(texts, normalize_embeddings=True)
```

Normalization scales each vector to length 1. This works naturally with cosine similarity and makes cosine comparison closely related to a dot product.

---

## Batch Embedding

Embedding one chunk at a time repeats model and processing overhead.

```python
for text in chunk_texts:
    vector = model.encode(text)
```

Batch embedding processes several texts together:

```python
embeddings = model.encode(
    chunk_texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

Batch size is a resource tradeoff:

- larger batches can improve throughput;
- excessively large batches can run out of memory; and
- the best value depends on model size and available hardware.

---

## Pinecone Overview

Pinecone is a managed vector database. It stores vector records and provides similarity search, metadata filtering, namespaces, and database-management operations without requiring us to operate the underlying database servers.

### Chroma and Pinecone comparison

| Topic | Chroma | Pinecone |
|---|---|---|
| Deployment | Commonly local or self-hosted | Managed cloud service |
| Main grouping | Collection | Index and namespace |
| Connection | Local client/process | API key and hosted endpoint |
| Scaling | Depends on local deployment | Managed serverless infrastructure |
| Classroom role | Local vector-database fundamentals | Hosted production-style vector search |

### Important Pinecone terms

| Term | Meaning |
|---|---|
| API key | Secret credential used to access Pinecone |
| Index | Vector search structure with a dimension and metric |
| Dimension | Number of values in every dense vector |
| Metric | Method used to compare vectors, such as cosine |
| Record | An ID, vector, and metadata |
| Namespace | Logical partition of records inside an index |
| Metadata | Structured fields stored with a record |
| Upsert | Insert a record or replace the record with the same ID |

### Pinecone on AWS from Colab

```python
pc.create_index(
    name=INDEX_NAME,
    dimension=VECTOR_DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

`cloud="aws"` means Pinecone hosts the serverless index on AWS infrastructure.

The Colab notebook connects directly to Pinecone using the Pinecone SDK and API key. This exercise does not require:

- an AWS access key;
- an AWS secret key;
- `boto3`; or
- manually created AWS database infrastructure.

### Protect API keys

Never place a secret directly in a notebook cell.

Use one of these approaches:

- Colab Secrets;
- an environment variable; or
- `getpass()` for a temporary hidden entry.

Do not commit keys to Git or submit a notebook containing a visible key.

---

## Preparing Records for Upsert

A dense Pinecone record contains an ID, vector values, and metadata.

```python
{
    "id": "nasa-se-p0042-c003",
    "values": [0.018, -0.224, 0.091, ...],
    "metadata": {
        "text": "The passage contained in this chunk...",
        "document_id": "nasa-systems-engineering-handbook-rev2",
        "source": "NASA Systems Engineering Handbook, Revision 2",
        "page_number": 42,
        "chunk_number": 3,
        "organizations": ["NASA"]
    }
}
```

### Metadata rules used in this lesson

- Keep the object flat; do not store nested dictionaries.
- Use clear, consistent field names.
- Remove null and empty values.
- Store numbers as numbers when they will be compared numerically.
- Limit large entity lists.
- Store chunk text so it can be displayed after retrieval.

### What belongs in vectors versus metadata?

| Vector | Metadata |
|---|---|
| Overall semantic meaning | Exact page number |
| Similarity relationships | Source title |
| Learned language patterns | Document ID |
| Conceptual relevance | Named organizations or locations |

The vector is used for ranking by meaning. Metadata is used for exact display, organization, and filtering.

---

## Batch Upsert

Upsert records in groups rather than sending one request per vector.

```python
for batch in batched(records, batch_size=100):
    index.upsert(vectors=batch, namespace=NAMESPACE)
```

If a namespace does not exist, Pinecone creates it during the upsert.

Pinecone is eventually consistent. Immediately querying after a write may briefly return older results, so applications may need a short delay or retry strategy.

---

## Semantic Search

A semantic query follows the same vectorization process as the documents.

```text
Question
  -> query embedding
  -> Pinecone nearest-neighbor search
  -> matching record IDs, scores, and metadata
```

```python
query_vector = embedding_model.encode(
    query,
    normalize_embeddings=True,
).tolist()

result = index.query(
    namespace=NAMESPACE,
    vector=query_vector,
    top_k=5,
    include_metadata=True,
)
```

### What does `top_k` mean?

`top_k=5` asks Pinecone for the five closest eligible records.

It does not guarantee that all five results are useful. Production systems may also use:

- a similarity threshold;
- reranking;
- hybrid semantic and keyword search;
- metadata filters; or
- evaluation datasets to measure retrieval quality.

### Understanding scores

For cosine similarity, a higher score represents greater vector similarity.

A score is not a universal percentage of correctness. Its meaning depends on:

- the embedding model;
- the data;
- chunking decisions;
- the query; and
- the similarity metric.

---

## Metadata Filtering

A metadata filter restricts which records can participate in the vector search.

### Page-range filter

```python
filter={
    "page_number": {
        "$gte": 30,
        "$lte": 80,
    }
}
```

### NER-derived organization filter

```python
filter={
    "organizations": {
        "$in": ["NASA"]
    }
}
```

Filtering and semantic ranking work together:

```text
Metadata filter: Which records are eligible?
Vector similarity: How should eligible records be ranked?
```

### Namespace versus metadata filter

| Namespace | Metadata filter |
|---|---|
| Partitions records inside an index | Selects records by stored fields |
| Every operation targets one namespace | Applied to a particular query or operation |
| Useful for strong data separation | Useful for attributes such as source or page |
| Example: one namespace per tenant or dataset | Example: `page_number >= 30` |

---

## CRUD with Vector Records

### Create

Upsert a record with a new ID.

### Read

- Fetch exact records by ID.
- Query for records by similarity.

### Update or replace

Upserting the same ID replaces the complete existing record.

```python
index.upsert(vectors=[revised_record], namespace=NAMESPACE)
```

Pinecone also provides update operations for changing selected values or metadata. Upsert is appropriate when replacing the complete record.

### Delete

```python
index.delete(ids=[record_id], namespace=NAMESPACE)
```

Deleting an index is different from deleting a record. Deleting an index removes every namespace and record it contains.

---

## Retrieval-Augmented Generation (RAG)

### What retrieval does

```text
question -> query embedding -> vector search -> relevant chunks
```

The result is supporting context, not a final answer.

### What RAG adds

```text
question
  -> retrieve relevant chunks
  -> place chunks into an LLM prompt
  -> LLM generates an answer grounded in the retrieved context
```

RAG combines:

1. **Retrieval:** find relevant external information.
2. **Augmentation:** add that information to the model's prompt.
3. **Generation:** have an LLM create the response.

### Why use RAG?

RAG can help an LLM answer using:

- private organizational knowledge;
- recently updated documents;
- domain-specific sources; and
- passages that can be shown or cited.

RAG does not guarantee accuracy. Results still depend on extraction quality, chunking, embeddings, retrieval, prompt construction, and model behavior.

### Transition to LangChain

This lesson implements the retrieval pipeline directly except for chunk-boundary logic, which uses LangChain's focused text-splitter package.

LangChain can later provide abstractions for:

- document loaders;
- document loaders and additional text-splitting strategies;
- embedding-model wrappers;
- vector-store integrations;
- retrievers; and
- prompt and LLM pipelines.

Using only the splitter today previews LangChain without hiding embeddings, Pinecone records, queries, or retrieval. Next week can add the rest of LangChain's abstractions.

---

## Common Mistakes

### Embedding the entire document once

This prevents passage-level retrieval and can exceed model input limits.

### Using different models for chunks and queries

The vectors would not share a meaningful coordinate system.

### Creating the wrong index dimension

The index dimension must equal the embedding vector length.

### Losing source metadata

Without page and source information, results are difficult to validate or cite.

### Treating NER results as perfectly accurate

Generic NER output must be inspected, especially for specialized domains.

### Storing nested or null metadata

Prepare flat metadata and omit empty fields before upsert.

### Querying immediately after upsert

Eventual consistency can cause a brief delay before new records appear.

### Treating the Pinecone score as confidence

Similarity is not the same thing as answer correctness.

### Exposing the API key

Use secrets or a hidden prompt. Never hard-code credentials.

---

## Bite-Sized Version

A raw PDF must be extracted, cleaned, and split into chunks before semantic search. Each chunk is embedded into a vector and stored in Pinecone with a stable ID and metadata such as source, page number, and named entities. A user question is embedded with the same model, and Pinecone returns the closest chunk vectors. Metadata filters can restrict eligible records. This retrieval process becomes RAG when the retrieved chunks are added to an LLM prompt and the model generates an answer.
