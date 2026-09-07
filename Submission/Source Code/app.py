"""
app.py
======
IntelliAssist AI: Smart Document AI Assistant
Streamlit entry point - wires document_loader -> vector_store -> rag_chain
-> analysis into the UI described in the project brief:

  Chat tab       AI Chatbot with RAG + Source Citation Display
  Search tab     Semantic Search (embeddings) vs TF-IDF, side by side
  Summarize tab  Text Summarisation (LLM, with extractive fallback)
  Analytics tab  Sentiment & Intent Analysis over the conversation
  History tab    Conversation History (view + download)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd
from dotenv import load_dotenv
import streamlit as st

from src import analysis, document_loader, llm_provider, rag_chain, vector_store

load_dotenv()
st.set_page_config(page_title="IntelliAssist AI", page_icon="🧠", layout="wide")


# --------------------------------------------------------------- session state
def init_state():
    defaults = {
        "vector_store": None,
        "chunks": [],
        "raw_texts": {},           # filename -> full text (for summarisation)
        "chain": None,
        "llm": None,
        "llm_provider": None,
        "processed_files": [],
        "chat_history": [],        # list of dicts, see ask handler below
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧠 IntelliAssist AI")
    st.caption("Smart Document AI Assistant")

    st.subheader("1. Choose an LLM provider")
    provider = st.selectbox("Provider", llm_provider.PROVIDERS, help=(
        "Needed for generated chat answers and LLM summaries. "
        "Everything else (search, sentiment, intent) works without a key."
    ))
    env_key_name = "GOOGLE_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY"
    secret_key = ""
    try:
        secret_key = str(st.secrets.get(env_key_name, ""))
    except Exception:
        secret_key = ""
    api_key = st.text_input(
        f"{provider} API key",
        value=os.getenv(env_key_name, "") or secret_key,
        type="password",
        help=f"You can also store this key in .env as {env_key_name}.",
    )

    if st.button("Connect LLM", use_container_width=True):
        try:
            st.session_state.llm = llm_provider.get_llm(provider, api_key)
            st.session_state.llm_provider = provider
            if st.session_state.llm:
                st.success(
                    f"Connected to {provider} ({llm_provider.provider_model_name(provider)})."
                )
                # rebuild the chat chain with the new LLM if docs are already loaded
                if st.session_state.vector_store is not None:
                    st.session_state.chain = rag_chain.build_chain(
                        st.session_state.vector_store, st.session_state.llm
                    )
            else:
                st.warning("No key entered - running in retrieval-only mode.")
        except Exception as e:
            st.error(f"Couldn't connect: {e}")

    st.divider()

    st.subheader("2. Upload documents")
    uploads = st.file_uploader(
        "PDF, DOCX, or TXT - multiple allowed",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Extracting text, chunking, and building the embedding index…"):
                loaded = [document_loader.extract_text(f) for f in uploads]
                for doc in loaded:
                    st.session_state.raw_texts[doc.filename] = doc.text
                chunks = document_loader.chunk_documents(loaded)
                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store.build_vector_store(chunks)
                st.session_state.processed_files = [d.filename for d in loaded]
                st.session_state.chain = rag_chain.build_chain(
                    st.session_state.vector_store, st.session_state.llm
                )
            st.success(f"Processed {len(loaded)} file(s) into {len(chunks)} chunks.")

    if st.session_state.processed_files:
        st.caption("Loaded documents:")
        for name in st.session_state.processed_files:
            st.text(f"• {name}")

    st.divider()
    llm_status = "🟢 connected" if st.session_state.llm else "⚪ not connected (retrieval-only)"
    st.caption(f"LLM: {llm_status}")
    st.caption(f"Chunks indexed: {len(st.session_state.chunks)}")


# --------------------------------------------------------------------- header
st.title("IntelliAssist AI")
st.caption("Upload documents, then chat, search, summarise, and track conversation analytics.")

if not st.session_state.processed_files:
    st.info("👈 Upload one or more documents in the sidebar and click **Process documents** to get started.")

tab_chat, tab_search, tab_summary, tab_analytics, tab_history = st.tabs(
    ["💬 Chat", "🔍 Semantic Search", "📝 Summarise", "📊 Analytics", "🕘 History"]
)


# ------------------------------------------------------------------ Chat tab
with tab_chat:
    st.subheader("Ask questions about your documents")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander(f"Sources ({len(turn['sources'])})"):
                    for src in turn["sources"]:
                        st.markdown(f"**{src['source']}** · chunk {src['chunk_id']}")
                        st.caption(src["preview"])

    query = st.chat_input("Ask a question about your documents…")
    if query:
        if st.session_state.vector_store is None:
            st.warning("Process at least one document first.")
        else:
            with st.spinner("Thinking…"):
                intent = analysis.detect_intent(query)
                try:
                    result = rag_chain.ask(
                        st.session_state.chain,
                        st.session_state.vector_store,
                        query,
                        st.session_state.chat_history,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()
                sentiment = analysis.analyze_sentiment(query)

                sources = [
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "?"),
                        "preview": doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""),
                    }
                    for doc in result["sources"]
                ]

                st.session_state.chat_history.append({
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "query": query,
                    "answer": result["answer"],
                    "sources": sources,
                    "intent": intent,
                    "sentiment": sentiment["label"],
                    "generated": result["generated"],
                })
            st.rerun()


# -------------------------------------------------------------- Search tab
with tab_search:
    st.subheader("Compare semantic (embeddings) vs. keyword (TF-IDF) search")
    search_query = st.text_input("Search query", key="search_query")
    top_k = st.slider("Results to show", 1, 10, 4)

    if search_query:
        if st.session_state.vector_store is None:
            st.warning("Process at least one document first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Semantic search** (sentence-transformer embeddings)")
                for doc, score in vector_store.semantic_search(st.session_state.vector_store, search_query, k=top_k):
                    st.markdown(f"`{doc.metadata.get('source')}` · score {score:.3f}")
                    st.caption(doc.page_content[:350])
                    st.divider()
            with col2:
                st.markdown("**TF-IDF search** (keyword overlap)")
                results = analysis.tfidf_search(st.session_state.chunks, search_query, k=top_k)
                if not results:
                    st.caption("No keyword overlap found.")
                for doc, score in results:
                    st.markdown(f"`{doc.metadata.get('source')}` · score {score:.3f}")
                    st.caption(doc.page_content[:350])
                    st.divider()


# ------------------------------------------------------------- Summarise tab
with tab_summary:
    st.subheader("Summarise a document")
    if not st.session_state.raw_texts:
        st.info("Process at least one document first.")
    else:
        selected = st.selectbox("Document", list(st.session_state.raw_texts.keys()))
        length = st.select_slider("Extractive summary length (fallback mode)", options=[3, 5, 8, 12], value=5)

        if st.button("Generate summary"):
            text = st.session_state.raw_texts[selected]
            with st.spinner("Summarising…"):
                if st.session_state.llm is not None:
                    try:
                        summary = rag_chain.llm_summary(st.session_state.llm, text)
                        st.success("Generated with LLM")
                    except Exception as exc:
                        st.error(f"Summary generation failed: {exc}")
                        summary = analysis.extractive_summary(text, num_sentences=length)
                        st.info("Showing the local extractive summary instead.")
                else:
                    summary = analysis.extractive_summary(text, num_sentences=length)
                    st.info("No LLM connected - showing an extractive summary instead.")
            st.markdown(summary)


# ------------------------------------------------------------- Analytics tab
with tab_analytics:
    st.subheader("Conversation analytics")
    history = st.session_state.chat_history
    if not history:
        st.info("Ask a few questions in the Chat tab to see analytics here.")
    else:
        df = pd.DataFrame(history)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sentiment of your questions**")
            st.bar_chart(df["sentiment"].value_counts())
        with col2:
            st.markdown("**Intent distribution**")
            st.bar_chart(df["intent"].value_counts())

        generated_count = df["generated"].sum()
        st.caption(f"{generated_count}/{len(df)} answers were LLM-generated; the rest were retrieval-only.")


# --------------------------------------------------------------- History tab
with tab_history:
    st.subheader("Full conversation history")
    history = st.session_state.chat_history
    if not history:
        st.info("No conversation yet.")
    else:
        for turn in reversed(history):
            with st.container(border=True):
                st.caption(f"{turn['timestamp']} · intent: {turn['intent']} · sentiment: {turn['sentiment']}")
                st.markdown(f"**Q:** {turn['query']}")
                st.markdown(f"**A:** {turn['answer']}")

        st.download_button(
            "Download history as JSON",
            data=json.dumps(history, indent=2),
            file_name="intelliassist_history.json",
            mime="application/json",
        )
        if st.button("Clear history"):
            st.session_state.chat_history = []
            st.rerun()
