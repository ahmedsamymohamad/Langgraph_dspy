# agent/rag/retrieval.py

import os
import json
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


class LocalDocRetriever:
    """
    Simple local RAG retriever:
    - Loads all markdown files from docs/
    - Splits into paragraph chunks
    - Computes BM25 scores
    - Returns top-k relevant chunks with IDs + scores
    """

    def __init__(self, docs_path: str = "docs/", chunk_size: int = 1):
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.chunks = []            # list[{id, text, source}]
        self.tokenized_chunks = []  # list[list[str]]
        self.bm25 = None
        self.tfidf = None
        self.vectorizer = None

        self._load_docs()
        self._build_indexes()

    def _load_docs(self):
        """
        Load markdown files and chunk them by paragraphs.
        """
        for filename in os.listdir(self.docs_path):
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(self.docs_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            paragraphs = [
                p.strip() for p in content.split("\n\n") if p.strip()
            ]

            for i, paragraph in enumerate(paragraphs):
                chunk_id = f"{filename.replace('.md','')}::chunk{i}"
                self.chunks.append({
                    "id": chunk_id,
                    "text": paragraph,
                    "source": filename
                })

        print(f"[RAG] Loaded {len(self.chunks)} chunks from docs/")

    def _build_indexes(self):
        """
        Build BM25 and TF-IDF indexes.
        """
        # Tokenize for BM25
        self.tokenized_chunks = [
            chunk["text"].lower().split() for chunk in self.chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # TF-IDF fallback
        self.vectorizer = TfidfVectorizer()
        texts = [c["text"] for c in self.chunks]
        self.tfidf = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks using BM25 + TF-IDF hybrid scoring.
        """
        query_tokens = query.lower().split()

        # BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)

        # TF-IDF scores (fallback)
        tfidf_query = self.vectorizer.transform([query])
        tfidf_scores = (self.tfidf @ tfidf_query.T).toarray().squeeze()

        # Combine scores (weighted average)
        combined = 0.7 * bm25_scores + 0.3 * tfidf_scores

        # Get top-k indices
        top_idx = combined.argsort()[::-1][:k]

        results = []
        for idx in top_idx:
            results.append({
                "id": self.chunks[idx]["id"],
                "text": self.chunks[idx]["text"],
                "score": float(combined[idx]),
                "source": self.chunks[idx]["source"]
            })

        return results


# Quick test (optional)
if __name__ == "__main__":
    r = LocalDocRetriever("docs/")
    res = r.retrieve("What are Beverages return days?", k=3)
    print(json.dumps(res, indent=2))
