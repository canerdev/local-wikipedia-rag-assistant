"""Streamlit chat UI for the local Wikipedia RAG assistant (PRD U-1 … U-4)."""

from __future__ import annotations

import html

import streamlit as st

from app import GenerationResult, ask

st.set_page_config(page_title="Local Wikipedia RAG", layout="wide")

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def _reset_conversation() -> None:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Rendering (user → right, assistant → left)
# ---------------------------------------------------------------------------


def _render_user_turn(text: str) -> None:
    left, right = st.columns((1, 2), gap="medium")
    with left:
        st.empty()
    with right:
        safe = html.escape(text).replace("\n", "<br/>")
        st.markdown(
            f'<div class="rag-user-row"><div class="rag-bubble rag-bubble-user">{safe}</div></div>',
            unsafe_allow_html=True,
        )


def _render_assistant_turn(result: dict) -> None:
    left, right = st.columns((2, 1), gap="medium")
    with left:
        body = html.escape(result.get("content", "")).replace("\n", "<br/>")
        st.markdown(
            f'<div class="rag-bubble rag-bubble-assistant">{body}</div>',
            unsafe_allow_html=True,
        )

        chunks = result.get("chunks") or []
        if chunks:
            with st.expander("Retrieved chunks used for this answer", expanded=False):
                route = result.get("route")
                if route:
                    st.caption(f"Query route: **{route}**")
                for i, ch in enumerate(chunks, start=1):
                    md = ch.metadata or {}
                    title = md.get("entity_name", "—")
                    et = md.get("entity_type", "")
                    sec = md.get("section_title", "")
                    url = md.get("source_url", "")
                    st.markdown(f"**{i}.** `{title}` ({et}) — _{sec}_")
                    st.caption(f"distance `{ch.distance:.5f}` · {url}")
                    preview = ch.text.strip()
                    if len(preview) > 1200:
                        preview = preview[:1200] + "…"
                    st.text(preview)
        elif not result.get("error"):
            st.caption("No source chunks returned for this answer.")

    with right:
        st.empty()


def _replay_history() -> None:
    for m in st.session_state.messages:
        if m["role"] == "user":
            _render_user_turn(m["content"])
        else:
            _render_assistant_turn(m)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

# Bubbles used fixed light backgrounds; in dark theme Streamlit still applies light text →
# invisible text. Tie bubble foreground/background to Streamlit theme variables + forced contrast.
st.markdown(
    """
    <style>
    /* Chat bubbles: always readable in light and dark Streamlit themes */
    .rag-bubble {
        padding: 0.85rem 1.1rem;
        border-radius: 1rem;
        text-align: left;
        line-height: 1.55;
        border: 1px solid rgba(127, 127, 127, 0.35);
    }
    .rag-user-row {
        display: flex;
        justify-content: flex-end;
        width: 100%;
    }
    .rag-bubble,
    .rag-bubble * {
        color: var(--text-color) !important;
    }
    .rag-bubble-user {
        max-width: min(92%, 40rem);
        background: var(--secondary-background-color);
        background: color-mix(in srgb, var(--primary-color) 24%, var(--secondary-background-color));
        border-left: 4px solid var(--primary-color);
    }
    .rag-bubble-assistant {
        background: var(--secondary-background-color);
    }
    /* Older browsers: keep user hint via left border only */
    @supports not (background: color-mix(in srgb, red, red)) {
        .rag-bubble-user {
            background: var(--secondary-background-color);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Local Wikipedia RAG Assistant")
st.caption("Answers use your ingested Wikipedia corpus and local Ollama. No retrieval or generation logic in this file.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=20, value=5, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    st.divider()
    if st.button("Reset conversation", type="primary", use_container_width=True):
        _reset_conversation()
        st.rerun()

prompt = st.chat_input("Ask about a person or place from your corpus…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

_replay_history()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_q = st.session_state.messages[-1]["content"]
    try:
        with st.spinner("Generating answer…"):
            gr: GenerationResult = ask(
                last_q,
                k=k,
                return_sources=True,
                temperature=temperature,
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": gr.answer_text,
                "chunks": gr.retrieved_chunks,
                "route": gr.route_label,
            },
        )
    except Exception as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Something went wrong: {e}",
                "chunks": [],
                "route": None,
                "error": True,
            },
        )
    st.rerun()
