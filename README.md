# IntelliAssist AI: Smart Document AI Assistant

An AI-powered chatbot that lets you upload documents (PDF/TXT/DOCX) and
ask questions about them, built with a RAG (Retrieval-Augmented
Generation) architecture. Built for the *Major Project - Artificial
Intelligence* capstone brief.

## Feature checklist (from the brief)

| Brief requirement | Where it lives |
|---|---|
| Document Upload (PDF/TXT/DOCX) | `src/document_loader.py`, sidebar uploader in `app.py` |
| AI Chatbot with RAG | `src/rag_chain.py` (explicit retrieve → prompt → generate RAG flow) |
| Semantic Search using Embeddings | `src/vector_store.py` (FAISS + sentence-transformers) |
| Text Summarisation | `src/rag_chain.py::llm_summary` + `src/analysis.py::extractive_summary` fallback |
| Sentiment & Intent Analysis | `src/analysis.py` (VADER sentiment + rule-based intent) |
| Source Citation Display | Chat tab expander in `app.py`, powered by chunk metadata |
| Conversation History | `st.session_state.chat_history`, History tab, JSON download |
| Streamlit Web Interface | `app.py` |
| Frontend: Streamlit / Backend: Python, LangChain, Hugging Face, OpenAI/Gemini | see Tech stack below |
| NLP: TF-IDF, BERT Embeddings | `src/analysis.py::tfidf_search` + `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB: FAISS/ChromaDB | FAISS (`src/vector_store.py`) |
| Deployment: Streamlit Cloud/Render | see Deployment below |

## Architecture

```
Upload (PDF/TXT/DOCX)
        │
        ▼
document_loader.py  ── extract text, split into overlapping chunks
        │
        ▼
vector_store.py  ── embed chunks (sentence-transformers/BERT) → FAISS index
        │
        ├──► Chat tab:  rag_chain.py → retrieve top-k chunks → LLM (Gemini/OpenAI)
        │               generates a grounded answer + cites source chunks
        │
        ├──► Search tab: semantic_search() (FAISS) vs. tfidf_search() (sklearn),
        │                shown side by side
        │
        └──► Summarise tab: llm_summary() if an LLM is connected, otherwise
                             extractive_summary() (frequency-based, no API needed)

Every chat turn also runs analysis.py's sentiment + intent detection,
logged into chat_history and charted in the Analytics tab.
```

A key design choice: **only the generative LLM call needs an API key.**
Embeddings, semantic search, TF-IDF search, sentiment, intent, and
extractive summarisation all run locally with open-source
models/libraries. That means you can demo the whole app - including a
working chatbot that answers from your documents' most relevant
passages - before you've even added a Gemini/OpenAI key, and the app
only "levels up" to fully generated conversational answers once one is
added.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional but recommended) add an API key
cp .env.example .env
# edit .env and paste in a GOOGLE_API_KEY or OPENAI_API_KEY

# 4. Run it
streamlit run app.py
```

The first run downloads the embedding model - that is normal and it is cached locally.

### Getting a free Gemini key

Gemini has the most generous free tier and is what the app defaults to.
Go to <https://aistudio.google.com/app/apikey>, sign in, click "Create
API key," and paste it into the sidebar (or `.env`).

### Quick demo without any setup

Use `sample_docs/sample_notes.txt` (already included, a short article
about RAG itself) so you have something to upload immediately - handy
for your demo video and for testing before you get an API key.

## Deployment (Streamlit Cloud)

1. Push this project to a GitHub repository (see checklist below).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click "New
   app," and point it at your repo with `app.py` as the entry point.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```
   (Streamlit Cloud injects secrets as environment variables - if you
   want the sidebar key field to read this automatically instead of
   requiring the grader to paste it in, add
   the app reads `GOOGLE_API_KEY` automatically from Streamlit Secrets or the local `.env` file.)
4. Deploy. Cold starts take a minute or two the first time because of
   the embedding model download.

Render works too: use a `Procfile` with
`web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Project structure

```
IntelliAssistAI/
├── app.py                  Streamlit UI - the file you run
├── requirements.txt
├── .env.example
├── src/
│   ├── document_loader.py   PDF/DOCX/TXT extraction + chunking
│   ├── vector_store.py       embeddings + FAISS index + semantic search
│   ├── llm_provider.py       Gemini / OpenAI selector (Gemini defaults to `gemini-3.8-flash`)
│   ├── rag_chain.py           conversational RAG chain + LLM summarisation
│   └── analysis.py            sentiment, intent, TF-IDF search, extractive summary
├── sample_docs/
│   └── sample_notes.txt      ready-to-upload demo document
└── vector_store/              FAISS index saved here if you persist it
```

## Mapping to your deliverables

- **GitHub Repository** - `git init`, commit this folder, push. The
  `.gitignore` already excludes `.env` and the local FAISS index so you
  don't accidentally commit secrets or large binaries.
- **PPT Presentation / Final Report** - the "Feature checklist" and
  "Architecture" sections above map directly to slide/section headings
  (Problem → Solution → Architecture → Features → Tech stack → Demo →
  Learning outcomes). Say the word if you'd like me to draft the actual
  slide deck or report next.
- **3-5 min Demo Video** - a natural walkthrough order: upload
  `sample_notes.txt` → ask "What is RAG?" in Chat (show the source
  citation expander) → run the same query in Search to compare semantic
  vs. TF-IDF results → Summarise the document → show Analytics/History.
- **Evaluation Report** - the Search tab's side-by-side semantic vs.
  TF-IDF results are a ready-made basis for a retrieval-quality
  comparison section.

## Extension ideas (worth mentioning as future work in your report)

- Swap the rule-based intent classifier in `analysis.py` for a
  HuggingFace `zero-shot-classification` pipeline (e.g.
  `facebook/bart-large-mnli`) for ML-based rather than pattern-based
  intent detection.
- Replace the "direct stuff" `llm_summary()` with a proper LangChain
  `map_reduce` summarisation chain for documents longer than a few
  thousand words.
- Swap FAISS for ChromaDB (also listed in the brief) if you want
  persistent, queryable metadata filtering out of the box.


## Troubleshooting the previous Gemini 404

The original project used retired `gemini-1.5-flash` and later `gemini-2.5-flash` model IDs. The current project defaults to the stable `gemini-3.8-flash` endpoint and removes the old `ConversationalRetrievalChain` dependency path. If Google changes the recommended model again, set `GEMINI_MODEL` in `.env` without changing the application code.
