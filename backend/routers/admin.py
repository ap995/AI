"""Admin router for tenant and system management."""

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_tenant
from ..vector_store import VectorStore

router = APIRouter(prefix="/admin", tags=["Admin"])
vector_store = VectorStore()


@router.delete("/documents")
async def clear_tenant_docs(tenant: Dict = Depends(get_current_tenant)):
    vector_store.delete_tenant_docs(tenant["tenant_id"])
    return {"status": "cleared", "tenant_id": tenant["tenant_id"]}


@router.get("/stats")
async def tenant_stats(tenant: Dict = Depends(get_current_tenant)):
    return {
        "tenant_id": tenant["tenant_id"],
        "plan": tenant.get("plan", "free"),
    }
