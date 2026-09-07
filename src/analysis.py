"""
analysis.py
===========
Everything that isn't the core RAG pipeline but is still listed in the
brief:

  - analyze_sentiment()   -> "Sentiment & Intent Analysis" (sentiment half)
  - detect_intent()        -> "Sentiment & Intent Analysis" (intent half)
  - tfidf_search()          -> classic NLP search ("TF-IDF" in tech stack),
                               works with zero setup, good baseline to
                               compare against the embedding-based search
  - extractive_summary()    -> "Text Summarisation" fallback that needs no
                               LLM/API key, so summarisation still works
                               even before an API key is configured
"""

from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_core.documents import Document

_sentiment_analyzer = SentimentIntensityAnalyzer()

_STOPWORDS = set(
    "a an the is are was were be been being this that these those of to in on "
    "for with as by at from it its it's and or but if then so than not no nor "
    "do does did doing have has had having i you he she we they me him her us "
    "them my your his its our their what which who whom will would shall should "
    "can could may might must about into over under again further once here there "
    "all any both each few more most other some such only own same too very s t "
    "just don now".split()
)

# ---------------------------------------------------------------- sentiment
def analyze_sentiment(text: str) -> dict:
    """Returns VADER's compound score plus a human-readable label."""
    scores = _sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "compound": compound, "scores": scores}


# ------------------------------------------------------------------- intent
_INTENT_PATTERNS = [
    ("greeting", re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I)),
    ("summary_request", re.compile(r"\b(summar(y|ise|ize|ization)|tl;?dr|overview|gist)\b", re.I)),
    ("search_request", re.compile(r"\b(find|search|locate|where (is|are)|look up)\b", re.I)),
    ("definition_request", re.compile(r"\b(what is|what are|define|meaning of)\b", re.I)),
    ("comparison_request", re.compile(r"\b(compare|difference between|versus|vs\.?)\b", re.I)),
    ("procedure_request", re.compile(r"\b(how (do|to|can)|steps to|process for)\b", re.I)),
    ("gratitude", re.compile(r"^\s*(thanks|thank you|thx)\b", re.I)),
]


def detect_intent(query: str) -> str:
    """
    Lightweight rule-based intent classifier - deliberately dependency-free
    so it runs instantly with no model download. Swap in a HuggingFace
    zero-shot-classification pipeline here later if you want ML-based
    intent detection instead of rules for the report.
    """
    for label, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return label
    return "question" if query.strip().endswith("?") else "statement"


# --------------------------------------------------------------- TF-IDF search
def tfidf_search(chunks: list[Document], query: str, k: int = 4) -> list[tuple[Document, float]]:
    """
    Classic bag-of-words search, independent of the embedding-based
    semantic search - useful in the report to show/benchmark the
    difference between lexical (TF-IDF) and semantic (BERT embeddings)
    retrieval on the same document set.
    """
    if not chunks:
        return []
    texts = [c.page_content for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])
    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [pair for pair in ranked[:k] if pair[1] > 0]


# --------------------------------------------------------- extractive summary
def extractive_summary(text: str, num_sentences: int = 5) -> str:
    """
    Frequency-based extractive summariser (no LLM/API needed): scores each
    sentence by the sum of its non-stopword word frequencies and returns
    the top-scoring sentences in their original order. This is the
    fallback used when no API key is configured; llm_summary() in
    rag_chain.py is used instead whenever an LLM is available.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    words = re.findall(r"[a-zA-Z']+", text.lower())
    freqs = Counter(w for w in words if w not in _STOPWORDS)
    if not freqs:
        return " ".join(sentences[:num_sentences])
    max_freq = max(freqs.values())
    freqs = {w: f / max_freq for w, f in freqs.items()}

    scored = []
    for i, sentence in enumerate(sentences):
        sent_words = re.findall(r"[a-zA-Z']+", sentence.lower())
        score = sum(freqs.get(w, 0) for w in sent_words)
        # tiny bonus for early sentences - intros usually carry the thesis
        score += 0.001 * (len(sentences) - i)
        scored.append((i, sentence, score))

    top = sorted(scored, key=lambda x: x[2], reverse=True)[:num_sentences]
    top_in_order = sorted(top, key=lambda x: x[0])
    return " ".join(s for _, s, _ in top_in_order)
