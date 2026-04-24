import csv
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from .config import settings


@dataclass
class ProfileRow:
    """Одна строка датасета"""
    text: str
    label: int
    meta_id: str


class RetrieverStore:
    """Индексация и поиск"""

    def __init__(self) -> None:
        """Инициалищация коллекции"""
        self.rows = []
        self.index = None
        self.embeddings = None
        self._st_model = None
        self._qdrant_client = None

    def _embed(self, texts: list[str]):
        """эмбеддинги"""
        if self._st_model is None:
            self._st_model = SentenceTransformer(settings.embedding_model)
        vectors = self._st_model.encode(texts, normalize_embeddings=True)
        return np.array(vectors, dtype=np.float32)

    def _load_rows(self, csv_path: str) -> list[ProfileRow]:
        """чтение данных"""
        path = Path(csv_path)
        rows = []
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                text = row.get("profile", "").strip()
                if not text:
                    continue
                label = int(row.get("label", "0"))
                rows.append(ProfileRow(text=text, label=label, meta_id=str(i)))
        return rows

    def index_dataset(self, csv_path: str) -> dict[str, str | int]:
        """Векторизация"""
        self.rows = self._load_rows(csv_path)
        texts = [row.text for row in self.rows]
        self.embeddings = self._embed(texts)

        if settings.qdrant_url:
            self._qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            vector_size = self.embeddings.shape[1]
            self._qdrant_client.recreate_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            points = [
                models.PointStruct(
                    id=int(row.meta_id),
                    vector=self.embeddings[i].tolist(),
                    payload={"text": row.text, "label": row.label},
                )
                for i, row in enumerate(self.rows)
            ]
            self._qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)
            return {"backend": "qdrant", "docs": len(self.rows)}

        vector_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dim)
        self.index.add(self.embeddings)
        return {"backend": "faiss", "docs": len(self.rows)}

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Ретрив"""
        q_vector = self._embed([query])[0]

        if self._qdrant_client is not None:
            hits = self._qdrant_client.query_points(
                collection_name=settings.qdrant_collection,
                query=q_vector.tolist(),
                limit=k,
            )
            result = []
            for point in hits.points:
                payload = point.payload or {}
                result.append(
                    {
                        "text": str(payload.get("text", "")),
                        "label": int(payload.get("label", 0)),
                        "score": float(point.score),
                    }
                )
            return result

        scores, idxs = self.index.search(np.array([q_vector], dtype=np.float32), k)
        output = []
        for score, idx in zip(scores[0], idxs[0]):
            row = self.rows[int(idx)]
            output.append({"text": row.text, "label": row.label, "score": float(score)})
        return output
