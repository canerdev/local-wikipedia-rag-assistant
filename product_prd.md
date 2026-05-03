# Product Requirements Document
## Local Wikipedia RAG Assistant

**Version:** 1.0
**Status:** Draft

---

## 1. Overview

Build a local question-answering system that answers questions about famous people and places using Wikipedia as the data source. The system runs entirely on localhost — no external APIs. It ingests Wikipedia pages, chunks and embeds them, stores them in a local vector database, and uses a local language model to generate answers from retrieved context.

---

## 2. Functional Requirements

### 2.1 Ingest

| # | Requirement |
|---|---|
| I-1 | Fetch Wikipedia pages for at least 20 people and 20 places. |
| I-2 | The following entities must be included at minimum. |

**Required people:** Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo

**Required places:** Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest

| # | Requirement |
|---|---|
| I-3 | Each stored chunk must carry metadata: entity name, type (`person` or `place`), and source URL. |
| I-4 | Use language-native HTTP functionality to fetch Wikipedia pages. No high-level scraping libraries. |

### 2.2 Chunk

| # | Requirement |
|---|---|
| C-1 | Split each Wikipedia page into smaller chunks before embedding. |
| C-2 | Chunk size and strategy must be defined and documented. Design around the assumption that documents can be large. |

### 2.3 Embed and Store

| # | Requirement |
|---|---|
| E-1 | Generate embeddings locally using either `nomic-embed-text` via Ollama or `sentence-transformers`. No external embedding API is allowed. |
| E-2 | Store all chunks in a single local Chroma vector store with metadata (Option B). |
| E-3 | The vector store must persist to disk so ingestion does not need to be re-run on every startup. |

### 2.4 Retrieve

| # | Requirement |
|---|---|
| R-1 | Given a user query, determine whether it is about a person, a place, or both. |
| R-2 | If the query is about a person, filter the vector store to `type = person` before retrieval. If about a place, filter to `type = place`. If both or unclear, retrieve from all chunks. |
| R-3 | This routing logic may be simple — keyword-based or rule-based is acceptable. |
| R-4 | Return the top relevant chunks to pass to the generation step. |

### 2.5 Generate

| # | Requirement |
|---|---|
| G-1 | Use a local language model via Ollama (`llama3.2`, `Phi3`, or `mistral`) to generate answers. |
| G-2 | The answer must be grounded in the retrieved chunks. |
| G-3 | If the retrieved context does not contain enough information to answer, the system must return "I don't know" rather than hallucinate. |

### 2.6 Chat Interface

| # | Requirement |
|---|---|
| U-1 | Provide a Streamlit chat UI or CLI. |
| U-2 | The user can ask a question and receive a generated answer. |
| U-3 | The user can optionally view the retrieved chunks used to generate the answer. |
| U-4 | The user can clear or reset the conversation. |

---

## 3. Design Decision: Single Vector Store with Metadata (Option B)

One Chroma collection stores all chunks — both people and places. Each chunk carries a `type` field (`person` or `place`) in its metadata. At query time, a metadata filter is applied based on query routing. This avoids maintaining two separate stores and two separate ingestion pipelines, at the cost of slightly more query logic.

---

## 4. Technical Constraints

- Runs fully on localhost
- Language: Python
- Local model: Ollama
- Embeddings: `nomic-embed-text` via Ollama or `sentence-transformers`
- Vector store: Chroma (persisted to disk)
- UI: Streamlit or CLI
- No external LLM or embedding API

---

## 5. Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-1 | All required people and places are ingested and retrievable. |
| AC-2 | Queries about a specific person return answers grounded in that person's Wikipedia content. |
| AC-3 | Queries about a specific place return answers grounded in that place's Wikipedia content. |
| AC-4 | Mixed queries (e.g. "which famous place is in Turkey") return relevant results from both types. |
| AC-5 | Queries about unknown entities (e.g. "president of Mars") return "I don't know". |
| AC-6 | The vector store persists to disk — restarting the app does not require re-ingestion. |
| AC-7 | The chat interface supports asking questions, viewing retrieved context, and resetting the conversation. |
| AC-8 | No external API is used for the model or embeddings. |
| AC-9 | Wikipedia pages are fetched using language-native HTTP — no high-level scraping library. |

---

## 6. Required Deliverables

| File | Description |
|---|---|
| `product_prd.md` | This document |
| `readme.md` | Setup instructions, how to run, example queries |
| `recommendation.md` | 1–2 paragraphs on production deployment |
| GitHub repository | Working codebase |
| Demo video | 5 minute Loom or unlisted YouTube |

---

*End of PRD*
