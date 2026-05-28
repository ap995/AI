"""Qdrant vector store client with tenant isolation."""

import os
import uuid
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse


QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "sec_ur_docs")
EMBEDDING_DIM = int(os.environ.get("SECUR_EMBEDDING_DIM", "512"))


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(QDRANT_COLLECTION)
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _tenant_filter(self, tenant_id: str) -> qmodels.Filter:
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="tenant_id",
                    match=qmodels.MatchValue(value=tenant_id),
                )
            ]
        )

    def upsert(
        self,
        tenant_id: str,
        doc_id: str,
        vector: List[float],
        payload: Dict,
    ) -> None:
        payload["tenant_id"] = tenant_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def batch_upsert(
        self,
        tenant_id: str,
        vectors: List[List[float]],
        payloads: List[Dict],
        doc_ids: List[str],
    ) -> List[str]:
        for p in payloads:
            p["tenant_id"] = tenant_id

        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, did)) for did in doc_ids]

        points = [
            qmodels.PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(point_ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )
        return doc_ids

    def search(
        self,
        tenant_id: str,
        vector: List[float],
        top_k: int = 10,
    ) -> List[Dict]:
        results = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            query_filter=self._tenant_filter(tenant_id),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": res.id,
                "score": res.score,
                "payload": res.payload,
            }
            for res in results
        ]

    def delete_tenant_docs(self, tenant_id: str) -> None:
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=self._tenant_filter(tenant_id),
            ),
        )

    def health(self) -> bool:
        try:
            self.client.get_collection(QDRANT_COLLECTION)
            return True
        except Exception:
            return False
