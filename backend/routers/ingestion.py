"""Ingestion router for document upload."""

from typing import Dict, List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import uuid

from ..auth import get_current_tenant
from ..vector_store import VectorStore

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
vector_store = VectorStore()


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    source: str = "upload",
    tenant: Dict = Depends(get_current_tenant),
):
    doc_ids = []

    for file in files:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")

        doc_id = str(uuid.uuid4())
        embedding = [0.0] * 512  # placeholder

        vector_store.upsert(
            tenant_id=tenant["tenant_id"],
            doc_id=doc_id,
            vector=embedding,
            payload={
                "title": file.filename,
                "content": text[:1000],
                "source": source,
                "doc_id": doc_id,
                "content_type": file.content_type,
            },
        )
        doc_ids.append(doc_id)

    return {
        "document_ids": doc_ids,
        "count": len(doc_ids),
        "message": f"Uploaded {len(doc_ids)} files",
    }
