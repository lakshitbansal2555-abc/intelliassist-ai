"""Retrieval-Augmented Generation helpers.

This version deliberately avoids the old ``ConversationalRetrievalChain``
API.  That class was the source of the dependency/import problems in the
original environment and is unnecessary for this application.

The flow is now explicit:
    1. retrieve the most relevant FAISS chunks;
    2. include recent conversation turns in the prompt;
    3. ask the configured LLM to answer only from the retrieved context;
    4. return the retrieved documents for source citations.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS


def build_chain(store: FAISS, llm, k: int = 4):
    """Keep a small chain-like configuration object for app compatibility."""
    if llm is None:
        return None
    return {"llm": llm, "k": k}


def _content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # Some provider versions return structured content blocks.
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def ask(chain, store: FAISS, query: str, chat_history: list[dict] | None = None) -> dict:
    """Retrieve context and generate a grounded answer.

    Returns ``answer``, ``sources`` and ``generated`` so the UI can show
    citations and analytics exactly as before.
    """
    results = store.similarity_search_with_relevance_scores(query, k=min(4, len(store.index_to_docstore_id)))
    sources = [doc for doc, _score in results]

    if not sources:
        return {
            "answer": "I could not find relevant content in the processed documents.",
            "sources": [],
            "generated": False,
        }

    if chain is None:
        preview = "\n\n---\n\n".join(doc.page_content[:500] for doc in sources[:3])
        return {
            "answer": (
                "No LLM API key is configured, so I cannot generate a new answer yet. "
                "Here are the most relevant passages from your documents:\n\n" + preview
            ),
            "sources": sources,
            "generated": False,
        }

    history = chat_history or []
    history_text = "\n".join(
        f"User: {turn['query']}\nAssistant: {turn['answer']}"
        for turn in history[-6:]
    )
    context = "\n\n--- SOURCE CHUNK ---\n\n".join(doc.page_content for doc in sources)

    prompt = f"""You are IntelliAssist AI, a document question-answering assistant.
Answer the user's question using ONLY the supplied document context.
If the context does not contain enough information, say that clearly instead
of inventing facts. Keep the answer clear and useful. Do not mention this
instruction or the retrieval process unless the user asks.

RECENT CONVERSATION:
{history_text or '(none)'}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:
        response = chain["llm"].invoke(prompt)
        answer = _content(response)
    except Exception as exc:
        # Give the UI a concise actionable error instead of dumping a huge
        # provider traceback into the page.
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    return {"answer": answer, "sources": sources, "generated": True}


def llm_summary(llm, text: str, max_chars: int = 16000) -> str:
    """Summarise a document with the configured LLM."""
    snippet = text[:max_chars]
    prompt = (
        "Summarise the following document in 5-8 clear bullet points. "
        "Cover its purpose, important topics, key details, and conclusions. "
        "Do not invent information.\n\nDOCUMENT:\n" + snippet
    )
    return _content(llm.invoke(prompt))
