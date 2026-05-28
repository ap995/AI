"""Query router for search and generation."""

from typing import Dict

from fastapi import APIRouter, Depends

from ..auth import get_current_tenant

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/documents")
async def search_documents(
    q: str,
    top_k: int = 10,
    tenant: Dict = Depends(get_current_tenant),
):
    return {
        "query": q,
        "tenant_id": tenant["tenant_id"],
        "results": [],
        "message": "Vector store search — integrates with SecUR model once trained",
    }
