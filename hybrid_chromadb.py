"""
hybrid_chromadb.py
==================
Local hybrid search with ChromaDB: dense (OpenAI-compatible API) + BM25,
fused with Reciprocal Rank Fusion (RRF).

ROOT CAUSE (issue #6185)
  ChromaDB's local PersistentClient / EphemeralClient has no combined
  dense+sparse query endpoint – that lives only in the HTTP server.
  FIX: maintain *two* collections (same PersistentClient, same directory):
       • <name>_dense  → HNSW cosine, your embedding model
       • <name>_bm25   → BM25 sparse, built-in ChromaBm25EmbeddingFunction
  Then merge ranked results with RRF.

Install:
  pip install "chromadb>=0.6" openai snowballstemmer

Config (env vars):
  OPENAI_API_KEY   – OpenAI key  OR "lm-studio" / "EMPTY" for local servers
  OPENAI_BASE_URL  – leave unset for OpenAI; set to http://localhost:1234/v1
                     for LM Studio / Ollama / vLLM
  EMBEDDING_MODEL  – default: text-embedding-3-small
"""

from __future__ import annotations
import os
import uuid
from typing import List, Optional

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
from openai import OpenAI


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dense embedding function  (any OpenAI-compatible server)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleEmbeddings(EmbeddingFunction):
    """
    ChromaDB EmbeddingFunction backed by an OpenAI-compatible embeddings API.
    Works with: OpenAI, Azure OpenAI, LM Studio, Ollama, vLLM, TGI, etc.

    Priority for credentials:
      1. Constructor arguments
      2. OPENAI_API_KEY / OPENAI_BASE_URL environment variables
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),  # None → OpenAI default
        )

    def __call__(self, input: List[str]) -> Embeddings:
        resp = self._client.embeddings.create(model=self.model, input=input)
        return [item.embedding for item in resp.data]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HybridChromaDB
# ─────────────────────────────────────────────────────────────────────────────

class HybridChromaDB:
    """
    Two-collection local hybrid store: dense HNSW + BM25 sparse, merged via RRF.

    Both collections share one PersistentClient (same directory on disk).
    Documents live in the dense collection; the BM25 collection stores the
    same IDs so results can be cross-referenced.

    Parameters
    ----------
    collection_name : str
        Logical name; creates <name>_dense and <name>_bm25 collections.
    embedding_fn    : EmbeddingFunction
        Dense embedding function (e.g. OpenAICompatibleEmbeddings).
    persist_dir     : str
        Path where ChromaDB persists data.  Created automatically.
    bm25_k / bm25_b : float
        BM25 hyperparameters (term-frequency saturation / length normalisation).
    """

    def __init__(
        self,
        collection_name: str,
        embedding_fn: EmbeddingFunction,
        persist_dir: str = "./chroma_hybrid_db",
        bm25_k: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        self._chroma = chromadb.PersistentClient(path=persist_dir)
        bm25_ef = ChromaBm25EmbeddingFunction(k=bm25_k, b=bm25_b)

        # Dense collection – keeps documents, metadata, and HNSW index
        self._dense = self._chroma.get_or_create_collection(
            name=f"{collection_name}_dense",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Sparse BM25 collection – same IDs, BM25-sparse embeddings
        self._sparse = self._chroma.get_or_create_collection(
            name=f"{collection_name}_bm25",
            embedding_function=bm25_ef,
        )

    # ── public: add documents ─────────────────────────────────────────────────

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Upsert a list of raw strings into both collections."""
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        # Dense: full docs + metadata
        self._dense.upsert(documents=texts, metadatas=metadatas, ids=ids)

        # BM25 sparse: only IDs + docs (embeddings generated automatically)
        self._sparse.upsert(documents=texts, ids=ids)

        return ids

    # ── public: hybrid search (RRF) ───────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        rrf_k: int = 60,
    ) -> List[dict]:
        """
        Hybrid retrieval: dense HNSW + BM25, fused with Reciprocal Rank Fusion.

        Parameters
        ----------
        query     : natural-language query string
        n_results : how many results to return
        rrf_k     : RRF constant (higher → less sensitive to top ranks, default 60)

        Returns
        -------
        List of dicts with keys:
          id, document, metadata, rrf_score, dense_rank, bm25_rank
        """
        n_corpus = self._dense.count()
        if n_corpus == 0:
            return []

        # Fetch more candidates than needed so RRF has room to re-rank
        fetch_k = min(n_corpus, max(n_results * 4, 20))

        # — Dense (semantic) retrieval
        dense_res = self._dense.query(query_texts=[query], n_results=fetch_k)
        dense_ids: List[str] = dense_res["ids"][0]

        # — BM25 (keyword) retrieval
        bm25_res = self._sparse.query(query_texts=[query], n_results=fetch_k)
        bm25_ids: List[str] = bm25_res["ids"][0]

        # — Reciprocal Rank Fusion
        rrf_scores = self._rrf([dense_ids, bm25_ids], k=rrf_k)
        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        top_ids = [doc_id for doc_id, _ in top]

        # — Fetch full documents from dense collection
        fetched = self._dense.get(ids=top_ids, include=["documents", "metadatas"])
        id_to_info = {
            doc_id: {"document": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        }

        dense_rank_map = {doc_id: r for r, doc_id in enumerate(dense_ids)}
        bm25_rank_map  = {doc_id: r for r, doc_id in enumerate(bm25_ids)}

        return [
            {
                "id":         doc_id,
                "document":   id_to_info[doc_id]["document"],
                "metadata":   id_to_info[doc_id]["metadata"],
                "rrf_score":  round(score, 6),
                "dense_rank": dense_rank_map.get(doc_id, -1),  # -1 = not in top fetch_k
                "bm25_rank":  bm25_rank_map.get(doc_id, -1),
            }
            for doc_id, score in top
            if doc_id in id_to_info
        ]

    # ── public: pure vector search (useful for comparison) ───────────────────

    def vector_search(self, query: str, n_results: int = 5) -> List[dict]:
        """Dense-only retrieval for comparison with hybrid."""
        res = self._dense.query(query_texts=[query], n_results=n_results)
        return [
            {
                "id":       doc_id,
                "document": doc,
                "metadata": meta,
                "cosine_distance": dist,  # 0 = identical, 2 = opposite
            }
            for doc_id, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0],
            )
        ]

    def count(self) -> int:
        """Number of documents in the store."""
        return self._dense.count()

    # ── private: RRF ─────────────────────────────────────────────────────────

    @staticmethod
    def _rrf(rankings: List[List[str]], k: int = 60) -> dict[str, float]:
        """
        Reciprocal Rank Fusion.
        Each list is an ordered ranking of document IDs (best → worst).
        Returns {id: score} – higher score = more relevant.

        Formula: score(d) = Σ  1 / (k + rank(d, list_i) + 1)
        """
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Demo
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "CUDA is a parallel computing platform and programming model developed by NVIDIA for GPU-accelerated computing.",
    "Python is a high-level general-purpose language praised for its readability and rich ecosystem.",
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative language model.",
    "BM25 is a bag-of-words ranking function that scores documents by term frequency and inverse document frequency.",
    "Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbor search.",
    "Large language models like GPT-4 are trained on massive text corpora using transformer architectures.",
    "Quantization reduces model size by representing weights in lower precision such as INT8 or FP4.",
    "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest-neighbor search.",
    "The attention mechanism in transformers computes weighted sums over all input tokens to capture context.",
    "ChromaDB is an open-source embedding database designed for AI applications and RAG pipelines.",
    "Reciprocal Rank Fusion (RRF) merges multiple ranked lists into a single combined ranking without score normalisation.",
    "Hybrid search combines dense semantic retrieval and sparse keyword retrieval for superior coverage.",
    "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large language models.",
    "Flash Attention is an IO-aware exact attention algorithm that significantly reduces GPU memory usage.",
    "Cosine similarity measures the angle between two embedding vectors and is a standard metric in semantic search.",
]

SAMPLE_QUERIES = [
    "How does approximate nearest-neighbor search work with graph structures?",  # hits HNSW (vector) + bm25 keyword overlap
    "BM25 term frequency keyword ranking",                                        # heavily favours BM25
    "GPU CUDA NVIDIA parallel programming",                                       # strongly semantic
    "combine sparse dense retrieval hybrid",                                      # should surface hybrid search + RRF docs
]


def _print(title: str, results: List[dict]) -> None:
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    for i, r in enumerate(results, 1):
        if "rrf_score" in r:
            score_str = (f"rrf={r['rrf_score']:.5f}  "
                         f"dense_rank={r['dense_rank']:>2}  "
                         f"bm25_rank={r['bm25_rank']:>2}")
        else:
            score_str = f"cosine_dist={r['cosine_distance']:.4f}"
        print(f"  [{i}] {score_str}")
        print(f"       {r['document'][:80]}")
    print()


def main() -> None:
    # ── 1. Build embedding function ──────────────────────────────────────────
    # OpenAI:        set OPENAI_API_KEY, leave base_url=None
    # Local server:  set OPENAI_BASE_URL=http://localhost:1234/v1
    #                and pick the right model name
    embed_fn = OpenAICompatibleEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        # Uncomment below for local servers (LM Studio / Ollama / vLLM):
        # base_url="http://localhost:1234/v1",
        # api_key="lm-studio",
    )

    # ── 2. Create / open the hybrid store ────────────────────────────────────
    store = HybridChromaDB(
        collection_name="demo",
        embedding_fn=embed_fn,
        persist_dir="./chroma_hybrid_db",   # persisted to disk
    )

    # ── 3. Index documents (skipped if already persisted) ────────────────────
    if store.count() == 0:
        print(f"Indexing {len(SAMPLE_TEXTS)} documents …")
        store.add_texts(
            texts=SAMPLE_TEXTS,
            metadatas=[{"source": f"doc_{i}", "domain": "AI/ML"}
                       for i in range(len(SAMPLE_TEXTS))],
        )
        print(f"Done. Store now contains {store.count()} documents.\n")
    else:
        print(f"Loaded existing store ({store.count()} documents).\n")

    # ── 4. Run queries ────────────────────────────────────────────────────────
    for query in SAMPLE_QUERIES:
        hybrid_results = store.hybrid_search(query, n_results=3)
        vector_results = store.vector_search(query, n_results=3)

        _print(f"🔀 HYBRID  | {query}", hybrid_results)
        _print(f"🔵 VECTOR  | {query}", vector_results)


if __name__ == "__main__":
    main()
