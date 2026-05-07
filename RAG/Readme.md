# RAG (Retrieval-Augmented Generation) — Learning Notes

> Notes compiled while learning from YouTube, articles, and docs. A brain-dump of everything I've picked up about RAG.

---

## What is RAG?

RAG = Retrieval-Augmented Generation. The idea is simple: instead of relying solely on what a language model "knows" from training, you **fetch relevant context** from an external knowledge base at query time and feed it to the LLM.

Think of it like giving an open-book exam to the model — instead of memorizing everything, it can look things up.

**Why bother?**
- LLMs have a knowledge cutoff — they don't know recent events
- LLMs hallucinate facts, especially on niche topics
- You can't fine-tune a model on every private document
- RAG lets you plug in *your* data without retraining

---

## The Core RAG Pipeline

```
User Query
    |
    v
[Embed the query]
    |
    v
[Vector Search] ---- (external knowledge base / vector DB)
    |
    v
[Retrieve top-k chunks]
    |
    v
[Stuff chunks into prompt + original query]
    |
    v
[LLM generates answer]
    |
    v
Final Response
```

### Step-by-step breakdown

1. **Ingestion** — load documents, split into chunks, embed each chunk, store in a vector DB
2. **Retrieval** — at query time, embed the query, run similarity search to get top-k relevant chunks
3. **Augmentation** — inject retrieved chunks into the LLM prompt as context
4. **Generation** — LLM reads the context + question, produces a grounded answer

---

## Chunking — More Important Than It Seems

Chunking is how you split your documents before embedding. Get this wrong and retrieval suffers badly.

### Chunking strategies

| Strategy | How it works | Good for |
|---|---|---|
| Fixed-size | Split every N tokens/chars | Quick and dirty baseline |
| Sentence splitter | Split on sentence boundaries | General text |
| Recursive character | Split on `\n\n`, `\n`, ` `, etc. in order | Most common default |
| Semantic chunking | Embed sentences, split where meaning shifts | High-quality retrieval |
| Document-aware | Respect headers, sections (Markdown, HTML) | Structured docs |

**Key parameters:**
- `chunk_size` — how big each chunk is (e.g., 512 tokens)
- `chunk_overlap` — how much consecutive chunks share (e.g., 50–100 tokens), prevents losing context at boundaries

**Rule of thumb:** chunk size should match the "unit of meaning" in your data. A 2-page PDF section shouldn't be one chunk.

---

## Embeddings

An embedding is a vector (list of floats) that represents the semantic meaning of a piece of text. Similar texts have vectors that are close together in vector space.

### Popular embedding models

- `text-embedding-3-small` / `text-embedding-3-large` — OpenAI, widely used
- `all-MiniLM-L6-v2` — Sentence Transformers, free, fast, great baseline
- `bge-large-en-v1.5` — BAAI, strong open-source option
- `nomic-embed-text` — good open-source alternative
- Cohere Embed v3 — solid commercial option

**Dimensionality matters:** higher dims = more expressive but slower search and more storage.

**Important:** Use the **same embedding model** for ingestion and retrieval. Mixing models = garbage results.

---

## Vector Databases

Where you store your embeddings. They support fast approximate nearest-neighbor (ANN) search.

| DB | Notes |
|---|---|
| **FAISS** | Facebook's library, in-memory, great for local/small scale |
| **Chroma** | Easy to use locally, popular with LangChain/LlamaIndex tutorials |
| **Pinecone** | Managed cloud, production-ready |
| **Weaviate** | Open-source, hybrid search built-in |
| **Qdrant** | Rust-based, fast, open-source |
| **pgvector** | Postgres extension — if you already use Postgres, easy win |
| **Milvus** | Scales to billions of vectors |

**Similarity metrics:**
- Cosine similarity — most common for text
- Dot product — faster, works if vectors are normalized
- Euclidean (L2) — less common for text

---

## Retrieval Strategies

Basic vector search is just the start. There are smarter ways to retrieve.

### Dense retrieval
Standard vector similarity search. Fast, semantic-aware.

### Sparse retrieval (BM25, TF-IDF)
Keyword-based. Good for exact matches, proper nouns, codes. Old-school but still useful.

### Hybrid retrieval
Combine dense + sparse. Use a weighted sum or Reciprocal Rank Fusion (RRF). Often outperforms either alone.

```
final_score = alpha * dense_score + (1 - alpha) * sparse_score
```

### MMR (Maximal Marginal Relevance)
Avoids returning 5 near-identical chunks. Balances relevance vs. diversity in the retrieved set.

### Multi-query retrieval
Generate multiple versions of the user query → retrieve for each → merge results. Helps when the original query is ambiguous.

### HyDE (Hypothetical Document Embeddings)
Ask the LLM to generate a hypothetical answer first → embed that → use it to search. Weird but works surprisingly well.

---

## Prompt Construction

Once you have your chunks, you need to build the prompt. Classic structure:

```
System: You are a helpful assistant. Answer based on the provided context only.
        If the answer is not in the context, say you don't know.

Context:
---
[chunk 1 text]
---
[chunk 2 text]
---

Question: {user_query}

Answer:
```

**Tips:**
- Tell the model to say "I don't know" if the answer isn't in context — reduces hallucination
- Order retrieved chunks by relevance (most relevant first or last — "lost in the middle" problem means models pay more attention to edges)
- Don't stuff too many chunks — diminishing returns and you'll hit the context window

---

## The "Lost in the Middle" Problem

Research finding: LLMs pay more attention to content at the **beginning and end** of long contexts. Stuff buried in the middle gets ignored.

**Fix:** put the most important chunk first or last. Experiment with ordering.

---

## Reranking

After retrieving top-k candidates, use a cross-encoder to **rerank** them. Cross-encoders are slower but much more accurate than bi-encoders for scoring relevance.

Popular rerankers:
- Cohere Rerank API
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (Sentence Transformers)
- BGE Reranker

Typical flow: retrieve top-20 with vector search → rerank → keep top-5 for the prompt. This is a big quality boost with manageable latency.

---

## Advanced RAG Patterns

### Parent-Child Chunking
- Store **small chunks** for precise retrieval
- But retrieve the **parent (larger) chunk** to give the LLM more context
- Best of both worlds

### Sentence Window Retrieval
- Embed individual sentences
- But when retrieved, return a window of surrounding sentences as context

### Self-Query / Metadata Filtering
- Use an LLM to extract filters from the user query (e.g., "only docs from 2024")
- Apply these as metadata filters in vector search before doing similarity search

### RAG Fusion
- Generate multiple queries → retrieve for each → merge and deduplicate using RRF → pass to LLM

### Agentic RAG
- LLM decides *when* to retrieve and *what* to search for
- Can do multiple rounds of retrieval to answer complex questions
- Tools like LangGraph, LlamaIndex Agents support this

---

## Evaluation

Hard to know if your RAG is actually working without proper evals.

### Key metrics

| Metric | What it measures |
|---|---|
| **Context Relevance** | Are retrieved chunks relevant to the query? |
| **Faithfulness** | Does the answer stick to the retrieved context? (no hallucination) |
| **Answer Relevance** | Does the answer actually address the question? |
| **Context Recall** | Did retrieval find all the needed info? |

### Eval frameworks
- **RAGAS** — popular open-source RAG eval framework, measures all 4 above
- **TruLens** — tracks and evaluates LLM app runs
- **LangSmith** — LangChain's tracing + eval platform

**Tip:** Build a golden Q&A set from your documents. Run retrieval, check if the right chunks come back, then check if the answer is correct. Do this before and after any change.

---

## Common Pitfalls

1. **Bad chunking** — chunks too large, too small, or cutting through sentences badly
2. **Wrong embedding model** — using a general model for domain-specific text (legal, medical) → consider fine-tuning or domain-specific models
3. **Not enough overlap** — information at chunk boundaries gets lost
4. **No reranking** — top-k from vector search isn't always the most relevant
5. **Ignoring metadata** — not filtering by date, source, or doc type when you should
6. **Context window overflow** — stuffing too many chunks and truncating important content
7. **Hallucination on "I don't know"** — model makes up an answer when it's not in context; fix with explicit prompting
8. **Not evaluating** — shipping without knowing if retrieval actually works

---

## Frameworks & Tools

| Tool | Role |
|---|---|
| **LangChain** | Chains, loaders, splitters, retrievers — full ecosystem |
| **LlamaIndex** | Data framework for LLM apps, strong RAG primitives |
| **Haystack** | Production-ready NLP pipelines |
| **DSPy** | Programming (not prompting) LLM pipelines |
| **LangGraph** | Stateful agents with LangChain |

---

## Quick Mental Model

```
Documents → Chunks → Embeddings → Vector DB
                                        |
Query → Embed → Search → Top-K Chunks → Prompt → LLM → Answer
```

The whole game is: **make sure the right chunk is in the prompt when the LLM needs to answer**.

---

## Resources That Helped

- "RAG from scratch" series on YouTube (LangChain channel)
- LlamaIndex documentation — especially the "building RAG" guides
- RAGAS paper — good grounding on how to evaluate
- "Lost in the Middle" paper — explains context ordering effects
- Pinecone Learning Center — practical walkthroughs

---

*Last updated: May 2026*
