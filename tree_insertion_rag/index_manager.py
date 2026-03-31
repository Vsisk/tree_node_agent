from __future__ import annotations

from collections import Counter
import math
import os
from typing import Any

from tree_insertion_rag.models import CandidateNode, EmbeddingProvider, RetrievalHit
from tree_insertion_rag.tracing import PipelineTracer
from tree_insertion_rag.utils import cosine_similarity, min_max_normalize, tokenize

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None

try:
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BGEM3FlagModel = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None


class TokenTfidfEmbeddingProvider:
    """Pure-Python TF-IDF embedding fallback."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        self._fitted = False

    def encode(self, texts: list[str]) -> list[list[float]]:
        tokenized = [tokenize(text) for text in texts]
        if not self._fitted:
            self._fit(tokenized)
        return [self._vectorize(tokens) for tokens in tokenized]

    def _fit(self, tokenized_texts: list[list[str]]) -> None:
        doc_count = len(tokenized_texts)
        vocab_terms = sorted({token for tokens in tokenized_texts for token in tokens})
        self._vocab = {token: index for index, token in enumerate(vocab_terms)}
        df_counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            df_counter.update(set(tokens))
        self._idf = [
            math.log((1 + doc_count) / (1 + df_counter.get(term, 0))) + 1.0 for term in vocab_terms
        ]
        self._fitted = True

    def _vectorize(self, tokens: list[str]) -> list[float]:
        if not self._vocab:
            return []
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector = [0.0] * len(self._vocab)
        for token, count in counts.items():
            index = self._vocab.get(token)
            if index is None:
                continue
            tf = count / total
            vector[index] = tf * self._idf[index]
        return vector


class SklearnTfidfEmbeddingProvider:
    """sklearn-based TF-IDF provider; falls back to pure Python when sklearn is unavailable."""

    def __init__(self) -> None:
        self._fallback = TokenTfidfEmbeddingProvider()
        self.reset()

    def reset(self) -> None:
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False) if TfidfVectorizer else None
        self._fitted = False
        self._fallback.reset()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._vectorizer is None:
            return self._fallback.encode(texts)
        if not self._fitted:
            matrix = self._vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            matrix = self._vectorizer.transform(texts)
        return matrix.toarray().tolist()


class BgeM3EmbeddingProvider:
    """Dense embedding provider backed by `BAAI/bge-m3` with graceful fallback hooks."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 16,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._backend: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    def reset(self) -> None:
        # Transformer embeddings are stateless for retrieval use, so reset is a no-op.
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._lazy_load_model()
        if self._backend == "sentence_transformers":
            return self._encode_with_sentence_transformers(texts)
        if self._backend == "transformers":
            return self._encode_with_transformers(texts)
        if self._backend == "flag_embedding":
            return self._encode_with_flag_embedding(texts)
        raise RuntimeError("BAAI/bge-m3 backend is not available")

    def _lazy_load_model(self) -> None:
        if self._backend is not None:
            return

        last_error: Exception | None = None

        if BGEM3FlagModel is not None:
            try:
                self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
                self._backend = "flag_embedding"
                return
            except Exception as exc:  # pragma: no cover - depends on runtime model setup
                last_error = exc

        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(self.model_name, device=self._resolve_device())
                self._backend = "sentence_transformers"
                return
            except Exception as exc:  # pragma: no cover - depends on runtime model setup
                last_error = exc

        if AutoModel is not None and AutoTokenizer is not None and torch is not None:
            try:
                resolved_device = self._resolve_device()
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                self._model.to(resolved_device)
                self._backend = "transformers"
                self.device = resolved_device
                return
            except Exception as exc:  # pragma: no cover - depends on runtime model setup
                last_error = exc

        message = (
            "BAAI/bge-m3 backend is unavailable. Install one of: `FlagEmbedding`, "
            "`sentence_transformers`, or `transformers` + `torch`."
        )
        if last_error is not None:
            raise RuntimeError(f"{message} Last error: {last_error}") from last_error
        raise RuntimeError(message)

    def _encode_with_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _encode_with_flag_embedding(self, texts: list[str]) -> list[list[float]]:
        encoded = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=8192,
        )
        dense_vectors = encoded["dense_vecs"]
        return dense_vectors.tolist() if hasattr(dense_vectors, "tolist") else list(dense_vectors)

    def _encode_with_transformers(self, texts: list[str]) -> list[list[float]]:
        assert torch is not None
        vectors: list[list[float]] = []
        device = self.device or self._resolve_device()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            pooled = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            if self.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().tolist())
        return vectors

    @staticmethod
    def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
        assert torch is not None
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_sum = (last_hidden_state * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp(min=1e-9)
        return masked_sum / token_count

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return os.getenv("BGE_M3_DEVICE", "cpu")

    def backend_name(self) -> str:
        return self._backend or "uninitialized"


def create_default_embedding_provider() -> EmbeddingProvider:
    if (
        BGEM3FlagModel is not None
        or SentenceTransformer is not None
        or (AutoModel is not None and AutoTokenizer is not None and torch is not None)
    ):
        return BgeM3EmbeddingProvider()
    return SklearnTfidfEmbeddingProvider()


class SimpleBM25:
    """Small BM25 implementation used when rank_bm25 is unavailable."""

    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_docs)
        self.avg_doc_len = (
            sum(len(tokens) for tokens in tokenized_docs) / self.doc_count if self.doc_count else 0.0
        )
        self.doc_freqs: list[Counter[str]] = [Counter(tokens) for tokens in tokenized_docs]
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokenized_docs:
            document_frequency.update(set(tokens))
        return {
            token: math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for freq_map, tokens in zip(self.doc_freqs, self.tokenized_docs):
            doc_len = len(tokens) or 1
            score = 0.0
            for token in query_tokens:
                freq = freq_map.get(token, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
                score += idf * numerator / denominator
            scores.append(score)
        return scores


class IndexManager:
    """Manage sparse and dense retrieval over parent candidates."""

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider or create_default_embedding_provider()
        self.candidates: list[CandidateNode] = []
        self._candidate_by_id: dict[str, CandidateNode] = {}
        self._tokenized_docs: list[list[str]] = []
        self._bm25: Any = None
        self._dense_vectors: list[list[float]] = []
        self.last_build_debug: dict[str, Any] = {}
        self.last_retrieval_debug: dict[str, Any] = {}

    def build(self, candidates: list[CandidateNode], tracer: PipelineTracer | None = None) -> None:
        self.candidates = list(candidates)
        self._candidate_by_id = {candidate.node_id: candidate for candidate in candidates}
        self._tokenized_docs = [tokenize(candidate.retrieval_text) for candidate in candidates]

        if hasattr(self.embedding_provider, "reset"):
            self.embedding_provider.reset()

        self._dense_vectors = self._build_dense_vectors(candidates, tracer=tracer)
        self._bm25 = self._build_bm25(self._tokenized_docs) if candidates else None
        vector_dim = len(self._dense_vectors[0]) if self._dense_vectors else 0
        self.last_build_debug = {
            "candidate_count": len(candidates),
            "embedding_provider": type(self.embedding_provider).__name__,
            "embedding_backend": self._embedding_backend_name(),
            "dense_vector_dim": vector_dim,
            "bm25_backend": type(self._bm25).__name__ if self._bm25 is not None else None,
        }
        if tracer is not None:
            tracer.log("index.build", "Index built", self.last_build_debug)

    def retrieve_sparse(self, query: str, topk: int) -> list[tuple[CandidateNode, float]]:
        if not self.candidates or self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        raw_scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.candidates, raw_scores), key=lambda item: item[1], reverse=True)
        return ranked[:topk]

    def retrieve_dense(self, query: str, topk: int) -> list[tuple[CandidateNode, float]]:
        if not self.candidates or not self._dense_vectors:
            return []
        try:
            query_vector = self.embedding_provider.encode([query])[0]
        except Exception:
            self.embedding_provider = SklearnTfidfEmbeddingProvider()
            self._dense_vectors = self._build_dense_vectors(self.candidates)
            query_vector = self.embedding_provider.encode([query])[0]
        scored = [
            (candidate, cosine_similarity(query_vector, dense_vector))
            for candidate, dense_vector in zip(self.candidates, self._dense_vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topk]

    def hybrid_retrieve(
        self,
        query: str,
        topk: int,
        tracer: PipelineTracer | None = None,
    ) -> list[CandidateNode]:
        return [hit.candidate for hit in self.hybrid_retrieve_with_scores(query, topk, tracer=tracer)]

    def hybrid_retrieve_with_scores(
        self,
        query: str,
        topk: int,
        tracer: PipelineTracer | None = None,
    ) -> list[RetrievalHit]:
        if not self.candidates:
            return []

        recall_k = max(topk, min(20, len(self.candidates)))
        sparse_hits = self.retrieve_sparse(query, recall_k)
        dense_hits = self.retrieve_dense(query, recall_k)

        sparse_map = {candidate.node_id: score for candidate, score in sparse_hits}
        dense_map = {candidate.node_id: score for candidate, score in dense_hits}
        sparse_norm = min_max_normalize(sparse_map)
        dense_norm = min_max_normalize(dense_map)

        combined_ids = set(sparse_map) | set(dense_map)
        if not combined_ids:
            combined_ids = {candidate.node_id for candidate in self.candidates[:topk]}

        merged: list[RetrievalHit] = []
        for node_id in combined_ids:
            merged.append(
                RetrievalHit(
                    candidate=self._candidate_by_id[node_id],
                    sparse_score=sparse_map.get(node_id, 0.0),
                    dense_score=dense_map.get(node_id, 0.0),
                    hybrid_score=0.45 * sparse_norm.get(node_id, 0.0) + 0.55 * dense_norm.get(node_id, 0.0),
                )
            )

        merged.sort(key=lambda item: item.hybrid_score, reverse=True)
        final_hits = merged[:topk]
        self.last_retrieval_debug = {
            "topk": topk,
            "recall_k": recall_k,
            "embedding_provider": type(self.embedding_provider).__name__,
            "embedding_backend": self._embedding_backend_name(),
            "sparse_hits": self._serialize_hits(sparse_hits, "sparse"),
            "dense_hits": self._serialize_hits(dense_hits, "dense"),
            "hybrid_hits": [
                {
                    "node_id": hit.candidate.node_id,
                    "node_name": hit.candidate.node_name,
                    "jsonpath": hit.candidate.jsonpath,
                    "sparse_score": round(hit.sparse_score, 4),
                    "dense_score": round(hit.dense_score, 4),
                    "hybrid_score": round(hit.hybrid_score, 4),
                }
                for hit in final_hits
            ],
        }
        if tracer is not None:
            tracer.log(
                "retrieve.hybrid",
                "Hybrid retrieval completed",
                {
                    "embedding_provider": self.last_retrieval_debug["embedding_provider"],
                    "embedding_backend": self.last_retrieval_debug["embedding_backend"],
                    "dense_hits": self.last_retrieval_debug["dense_hits"][:5],
                    "sparse_hits": self.last_retrieval_debug["sparse_hits"][:5],
                    "hybrid_hits": self.last_retrieval_debug["hybrid_hits"][:5],
                },
            )
        return final_hits

    @staticmethod
    def _build_bm25(tokenized_docs: list[list[str]]) -> Any:
        if BM25Okapi is not None:
            return BM25Okapi(tokenized_docs)
        return SimpleBM25(tokenized_docs)

    def _build_dense_vectors(
        self,
        candidates: list[CandidateNode],
        tracer: PipelineTracer | None = None,
    ) -> list[list[float]]:
        if not candidates:
            return []
        texts = [candidate.retrieval_text for candidate in candidates]
        try:
            vectors = self.embedding_provider.encode(texts)
            if tracer is not None:
                tracer.log(
                    "embedding.encode",
                    "Encoded candidate retrieval texts",
                    {
                        "provider": type(self.embedding_provider).__name__,
                        "backend": self._embedding_backend_name(),
                        "count": len(texts),
                        "vector_dim": len(vectors[0]) if vectors else 0,
                    },
                )
            return vectors
        except Exception as exc:
            if tracer is not None:
                tracer.log(
                    "embedding.fallback",
                    "Embedding provider failed, falling back to TF-IDF",
                    {
                        "provider": type(self.embedding_provider).__name__,
                        "error": str(exc),
                    },
                )
            self.embedding_provider = SklearnTfidfEmbeddingProvider()
            vectors = self.embedding_provider.encode(texts)
            if tracer is not None:
                tracer.log(
                    "embedding.encode",
                    "Encoded candidate retrieval texts with fallback provider",
                    {
                        "provider": type(self.embedding_provider).__name__,
                        "backend": self._embedding_backend_name(),
                        "count": len(texts),
                        "vector_dim": len(vectors[0]) if vectors else 0,
                    },
                )
            return vectors

    def _embedding_backend_name(self) -> str:
        backend_method = getattr(self.embedding_provider, "backend_name", None)
        if callable(backend_method):
            return str(backend_method())
        return type(self.embedding_provider).__name__

    @staticmethod
    def _serialize_hits(
        hits: list[tuple[CandidateNode, float]],
        score_field: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "node_id": candidate.node_id,
                "node_name": candidate.node_name,
                "jsonpath": candidate.jsonpath,
                f"{score_field}_score": round(score, 4),
            }
            for candidate, score in hits
        ]
