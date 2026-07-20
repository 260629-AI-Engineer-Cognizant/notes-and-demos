# RAG Pipeline Components

# Documents

LangChain uses `Document` objects to represent text and metadata.

```python
Document(
    page_content="Badge B-17 opened the east gallery door.",
    metadata={
        "source": "security_log.md",
        "category": "security"
    }
)
```

## `page_content`

The text that may be split, embedded, retrieved, and sent to the model.

## `metadata`

Information describing the text.

Examples:

- Filename
- Page number
- Category
- Date
- Department
- Access group
- Document type

Metadata helps with:

- Source display
- Filtering
- Debugging
- Access control
- Provenance

# Document loaders

A loader converts a source into `Document` objects.

Common loaders support:

- PDF
- Markdown and text
- CSV
- HTML
- Websites
- Word documents
- S3
- Databases


# Document transformation

Documents may need to be cleaned before indexing.

Examples:

- Normalize whitespace
- Remove repeated headers
- Remove navigation text
- Add metadata
- Redact sensitive information
- Split long text

# Chunking

Chunking divides large documents into smaller retrieval units.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=220,
    chunk_overlap=40
)

chunks = splitter.split_documents(documents)
```

## Why split?

Retrieving an entire large document may:

- Include too much irrelevant text
- Use unnecessary tokens
- Reduce retrieval precision
- Exceed a model's context limit

## Chunk size

Controls how much content is placed in each chunk.

### Too small

- Facts lose surrounding context
- Definitions may be separated from details
- More chunks must be stored

### Too large

- Several topics are mixed together
- Retrieval becomes less precise
- More irrelevant context reaches the model

## Chunk overlap

Repeats some text between neighboring chunks.

Overlap helps preserve ideas that cross a chunk boundary.

There is no universal perfect chunk size. Test chunking using real questions.

# Embeddings

An embedding model converts text into a numeric vector.

Semantically related text should receive nearby vectors.

```text
Question:
Who entered after closing?

Evidence:
Badge B-17 opened the gallery door at 9:12 PM.
```

The wording differs, but the meanings are related.

Today's notebook uses a local sentence-transformer model for embeddings.

# Vector stores

A vector store saves:

- Embeddings
- Original chunk text
- Metadata

It also supports similarity search.

Today's vector store is Chroma:

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

This performs the equivalent of:

1. Embed each chunk.
2. Store the vector.
3. Store the text.
4. Store the metadata.

# Retrievers

A retriever accepts a question and returns relevant `Document` objects.

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

documents = retriever.invoke(question)
```

## Vector store versus retriever

```text
Vector store:
Stores and searches vectors.

Retriever:
Application-facing interface that returns relevant Documents.
```

A retriever may use:

- Vector similarity
- Keyword search
- Metadata filters
- Hybrid search
- Query rewriting
- Re-ranking
- Access rules

# Context formatting

The retriever returns Python objects. The model needs readable text.

```python
context = "\n\n".join(
    f"Source: {doc.metadata['source']}\n{doc.page_content}"
    for doc in retrieved_documents
)
```

After the operation is understood, it can become a helper:

```python
def format_documents(documents):
    ...
```

# Prompt templates

A prompt template creates consistent instructions with variables.

```python
prompt = ChatPromptTemplate.from_template(
    '''
    Answer only from the supplied evidence.

    Evidence:
    {context}

    Question:
    {question}
    '''
)
```

Benefits include:

- Reusable instructions
- Less string concatenation
- Easier testing
- Clear variables
- Easier versioning

# 10Chat model

Today's model is Gemini:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)
```

LangChain gives different chat providers a similar interface:

```python
llm.invoke(messages)
```

The provider can change while most of the RAG pipeline remains unchanged.

# Output parser

A chat model normally returns an `AIMessage`.

`StrOutputParser` extracts the text:

```python
parser = StrOutputParser()
```

Later, structured parsers can request:

- JSON
- Lists
- Dictionaries
- Pydantic models


# LangChain composition

After the manual flow is understood:

```python
rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Read it as:

```text
Question
├── retrieve and format → context
└── keep original question

context + question
→ prompt
→ model
→ output text
```

The chain is a reusable version of the same manual operations.
