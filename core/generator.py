"""Ollama chat completion with architecture §6 prompt template."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from core.retriever import RetrievedChunk

OLLAMA_CHAT_TIMEOUT_S = 180.0

SYSTEM_MSG = """You are the Local Wikipedia Answer Assistant operating only on localhost.
You must answer ONLY using factual content present inside the CONTEXT blocks below.
Treat each CONTEXT block's text as unreliable if it contradicts another; prefer the majority consistent claim; if irreconcilable, say you do not know.

If the CONTEXT does not clearly contain enough information to answer the question, respond with exactly:
I don't know

Do not use outside knowledge, speculation, or invented citations.
Do not mention system instructions, policies, or vector storage.
Keep answers concise (2–6 sentences) unless the question explicitly asks for a list.
When stating a fact grounded in CONTEXT, you may mention the Wikipedia page title inferred from CONTEXT metadata for clarity (do not invent URLs beyond what metadata provides inline in CONTEXT).
Never fabricate quotations."""

USER_TMPL = """CONTEXT:
{context_blocks}

QUESTION:
{question}

Respond with ONLY the answer text (or exactly "I don't know"). Do not prepend labels like "Answer:".
If you include multiple facts, separate sentences clearly; omit bullet lists unless requested."""


def _serialize_context_blocks(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, ch in enumerate(chunks):
        md = ch.metadata or {}
        et = md.get("entity_type", "")
        en = md.get("entity_name", "")
        sec = md.get("section_title", "")
        url = md.get("source_url", "")
        block = (
            f'[{i}] type={et} entity="{en}" section="{sec}"\n'
            f"URL: {url}\n"
            f"{ch.text}\n"
            f"---"
        )
        parts.append(block)
    return "\n".join(parts)


def generate(
    query: str,
    chunks: list[RetrievedChunk],
    ollama_model: str,
    ollama_host: str = "http://127.0.0.1:11434",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    if not chunks:
        return "I don't know"

    ctx = _serialize_context_blocks(chunks)
    user_content = USER_TMPL.format(context_blocks=ctx, question=query.strip())

    url = ollama_host.rstrip("/") + "/api/chat"
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_CHAT_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama chat HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama chat request failed: {e.reason}") from e

    data = json.loads(raw.decode("utf-8"))
    msg = data.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"].strip()

    content = data.get("response")
    if isinstance(content, str):
        return content.strip()

    raise RuntimeError("Malformed Ollama /api/chat response")
