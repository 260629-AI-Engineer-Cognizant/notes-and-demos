# LangChain and RAG Basics

# Large language models

A **large language model**, or LLM, is trained on large amounts of text and can generate,
summarize, classify, extract, and transform language.

Examples of tasks include:

- Answering questions
- Summarizing documents
- Extracting structured information
- Classifying requests
- Drafting text
- Explaining code

## Important limitation

An LLM does not automatically know:

- Private company documents
- A customer's current account information
- Documents created after training
- Whether remembered information is still current
- Whether a confident answer is actually supported

An LLM generates likely language. It is not automatically a database or a system of record.

# What LangChain is

LangChain is a framework for connecting language models to application components.

LangChain provides common interfaces for:

- Chat models
- Prompt templates
- Documents
- Document loaders
- Text splitters
- Embedding models
- Vector stores
- Retrievers
- Output parsers
- Tools
- Workflows and agents

LangChain is **not** the model itself.

In today's notebook:

```text
Gemini = the language model

LangChain = the framework connecting:
documents → retrieval → prompt → Gemini → response
```

# Why use LangChain?

Without a framework, an application still needs to:

1. Load data.
2. Split it.
3. generate embeddings.
4. store and retrieve vectors.
5. format a prompt.
6. call a model.
7. process the result.

LangChain provides reusable interfaces so these parts can be composed and replaced more easily.

For example, the application can switch:

```text
Gemini → Amazon Bedrock
```

without rewriting its document splitting, Chroma storage, or retrieval logic.

# Common LangChain use cases

## Document question answering

Retrieve relevant passages from policies, manuals, reports, or case files and use them to answer
questions.

## Semantic search

Find text with similar meaning even when the wording differs.

Example:

```text
Question:
Who entered after the museum closed?

Matching evidence:
Badge B-17 opened the gallery door at 9:12 PM.
```

## Summarization

Summarize:

- One document
- Several retrieved documents
- A conversation
- A collection of reports

## Information extraction

Convert unstructured language into fields such as:

```json
{
  "person": "Marcus Bell",
  "badge": "B-17",
  "time": "9:12 PM"
}
```

## Classification and routing

Determine whether a request is about:

- Billing
- Technical support
- Company policy
- A general question

Then route it to the appropriate workflow.

## Conversational applications

Maintain message history so follow-up questions can use previous context.

## Tool-using applications

Allow a model-controlled workflow to call approved tools such as:

- A calculator
- A database lookup
- A search function
- A vector retriever
- An internal API

# What RAG is

**Retrieval-Augmented Generation**, or RAG, combines retrieval with generation.

Instead of asking the model to answer using only its training, the application:

1. Retrieves relevant external information.
2. Supplies that information to the model.
3. Asks the model to answer from the supplied evidence.

```text
Question
   ↓
Retrieve evidence
   ↓
Prompt containing evidence
   ↓
LLM
   ↓
Grounded answer
```

# Why use RAG?

RAG is useful when the model needs:

- Private knowledge
- Current documents
- Frequently changing information
- Source references
- Domain-specific facts
- Evidence for its answer

Typical applications include:

- Company policy assistants
- Product documentation chatbots
- Legal-document search
- Medical-literature search
- Customer-support knowledge bases
- Research assistants
- The museum case-file assistant

# Indexing versus query time

A RAG system has two larger paths.

## Indexing path

Usually performed when documents are added or updated:

```text
source
→ load
→ clean
→ split
→ embed
→ store
```

## Query path

Performed each time a user asks a question:

```text
question
→ retrieve
→ format context
→ prompt model
→ return answer
```

The full knowledge base should not be re-embedded for every question.

# What two-step RAG means

**Two-step RAG** describes the query-time answering pattern:

```text
Step 1: Retrieve
Step 2: Generate
```

Retrieval always happens before generation.

The model does not decide whether it should search. This makes the process:

- Simple
- Predictable
- Easy to inspect
- Easier to test
- Appropriate for documentation and question-answering systems
