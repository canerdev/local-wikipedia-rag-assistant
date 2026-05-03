# Local Wikipedia RAG Assistant

**Demo video:** [Screen recording on Google Drive](https://drive.google.com/file/d/1kdJzzIXRs0H_CH8anQ1BXlkQLBZUFvAk/view?usp=sharing)

Python app that answers questions about **people and places** using **locally embedded Wikipedia chunks**, **Chroma** on disk, and **Ollama** for embeddings + chat. Runs on **localhost** only (no paid LLM/embedding APIs).

**Use these steps from the project root** (`local-wikipedia-rag-assistant`) so `./chroma_db` and imports resolve correctly.

## How it works

The app is **retrieval-augmented generation (RAG)**: the model is only asked to answer from **short passages** retrieved from a **local vector index**, not from “the whole internet” at answer time.

### 1. Offline: build the vector index

You run **`python -m ingest.run_ingest`** once (or again after changing the article list or embedder).

1. **Fetch** each configured article from **Wikipedia** over HTTPS (MediaWiki API; see `ingest/fetcher.py`).
2. **Chunk** the article text so long pages become many overlapping segments. Splitting is guided by **wiki section headings** where possible, then by size limits, so each chunk stays topically coherent (`ingest/chunker.py`).
3. **Embed** every chunk with **`nomic-embed-text`** via **Ollama** — the same setup must be used when you ask questions later (`ingest/embedder.py`).
4. **Store** embeddings and text in **Chroma** under **`./chroma_db`**. Every chunk carries **metadata**: at least **entity name**, **`person` or `place`**, **source URL**, plus fields like section title for transparency (`ingest/store.py`).

All people and places live in **one Chroma collection**. That avoids maintaining two separate indexes; filters at query time narrow by type when appropriate.

### 2. Online: answer a user question

The **Streamlit** UI calls **`ask()`** in `app.py`. That path is always the same:

1. **Classify** the question with a small **keyword / rule-based router**: is it mainly about a **person**, a **place**, or **ambiguous / both** (`core/router.py`)?
2. **Embed** the question with the **same embedding model** as ingest.
3. **Retrieve** the **top‑k** most similar chunks from Chroma using **cosine-style** vector search. If the router picked person or place, Chroma queries include a **`entity_type`** filter; if ambiguous, search runs **without** that filter so mixed questions can pull from either kind of article (`core/retriever.py`).
4. If retrieval returns **fewer than k hits** while a narrow filter was used, **`ask()`** **retries once** with **no type filter** so borderline wording does not silently miss relevant text.
5. **Generate**: the **`llama3.2`** (or configured) chat model receives a **structured prompt**: system rules + serialized context chunks + user question. It is instructed to stay grounded and to reply **`I don't know`** when context is insufficient (`core/generator.py`).

Nothing in this path calls remote LLM or embedding APIs; only **Wikipedia** is contacted during **ingest**, and **Ollama** during **embed + generate**.

### Where to go deeper

Chunk sizes, metadata field list, routing rules, and prompt wording are specified in [`docs/architecture.md`](docs/architecture.md). The product goals and constraints are in [`product_prd.md`](product_prd.md).

---

## 1. Install dependencies

**Requirements:** Python **3.10+**

Create a virtual environment, activate it, then install packages:

```bash
python3 -m venv .venv
```

| OS / shell | Activate |
|------------|-----------|
| macOS / Linux (bash/zsh) | `source .venv/bin/activate` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Run the local model (Ollama)

1. Install **Ollama**: https://ollama.com/download  
2. Ensure it is running (on macOS the menu-bar app usually starts the API at `http://127.0.0.1:11434`). Only run `ollama serve` in a terminal if nothing is listening and you prefer the CLI daemon.  
3. Pull the two models used by defaults in this repo:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

- **`llama3.2`** — generates answers at query time.  
- **`nomic-embed-text`** — embeds text at **ingest** time and embeds questions at **query** time (must stay the same for sensible retrieval).

Check:

```bash
ollama list
```

You should see both models listed.

**Note:** If `ollama serve` prints “address already in use”, Ollama is already running—you can skip that command.

---

## 3. Ingest data

This step downloads Wikipedia articles over HTTPS, chunks them, embeds with **`nomic-embed-text`** via Ollama, and writes a persistent store under **`./chroma_db`** *(local only — not checked into git; rerun ingest after clone)*. Needs **internet** and a working Ollama embed model.

With the venv **activated**, from the project root:

```bash
python -m ingest.run_ingest
```

Wait until it finishes. Success looks like **40 entities succeeded**, **`chunks_upserted` > 0**, and **`chunk_count`** matching that number in the log line `Store: {...}`.

Re-running **upserts** into the same collection (safe to repeat after failures).

**If ingest fails on HTTPS (certificate errors):** `certifi` is in `requirements.txt`; ensure `pip install -r requirements.txt` completed. On some macOS Python installs you may need to run **Install Certificates.command** from the Python folder, then retry.

**If you see HTTP 429 from Wikipedia:** wait and run ingest again; the script spaces requests and retries. You can increase `FETCH_DELAY_S` in `ingest/run_ingest.py` if needed.

---

## 4. Start the application

With the venv **activated**, Ollama **running**, and ingest **done** (non-empty `./chroma_db`), from the project root:

```bash
streamlit run app_ui.py
```

Streamlit opens a browser tab. Ask a question, optionally expand **retrieved sources**, and use **reset** to clear the chat.

**Optional (no UI):** from the project root, with the same venv:

```bash
python -c "from app import ask; r = ask('Who was Albert Einstein?'); print(r.answer_text)"
```

---

## 5. Example queries

Try questions that match ingested entities (mix of **people** and **places**):

- Who was Marie Curie, and what is she known for?
- Where is the Taj Mahal located?
- When was the Eiffel Tower built?
- What sport is Lionel Messi associated with?
- Tell me something about Machu Picchu.
- Compare in one sentence: Grand Canyon versus Niagara Falls geography (fact-based).

The router may classify as person vs place vs both; mixed questions still retrieve from all chunk types when needed.

---

## Verification (optional)

```bash
python -m unittest discover -s tests -v
```

---

## Optional configuration

Overrides (defaults are fine for grading/running as-is):

| Variable | Default | Role |
|---------|---------|------|
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Vector store folder |
| `CHROMA_COLLECTION_NAME` | `wikipedia_rag` | Collection name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Chat model (`phi3`, `mistral` possible if pulled + set) |
| `RAG_EMBEDDER_BACKEND` | `ollama_nomic` | Use `sentence_transformers` only if you change code paths consistently |
| `RAG_EMBEDDER_MODEL` | `nomic-embed-text` | Must match what you used at ingest |

**Rule:** whichever embedder/backend you used for ingest must match **`ask()`** at runtime (`RAG_EMBEDDER_*`), or retrieval quality will suffer.
