"""FastAPI main application with multi-tenant RAG endpoints."""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from .auth import (
    hash_password, verify_password, create_access_token,
    get_current_tenant, optional_tenant,
)
from .models import (
    TenantCreate, TenantResponse, TokenResponse, LoginRequest,
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
)
from .vector_store import VectorStore
from sec_ur.model.sec_ur import SecURModel
from sec_ur.config import SecURConfig


app = FastAPI(
    title="SecUR - Security Unified Retriever-Generator",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tenants: Dict[str, Dict] = {}
vector_store = VectorStore()

_model: Optional[SecURModel] = None
_tokenizer: Optional[AutoTokenizer] = None
_config: Optional[SecURConfig] = None


def get_model() -> SecURModel:
    global _model
    if _model is None:
        _config = SecURConfig()
        _model = SecURModel(_config)
        _model.eval()
    return _model


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return _tokenizer


def query_to_vector(query_text: str) -> List[float]:
    model = get_model()
    tokenizer = get_tokenizer()

    tokens = tokenizer(
        query_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        vec = model.encode_queries(
            tokens["input_ids"],
            tokens["attention_mask"],
        )
    return vec[0].tolist()


def generate_answer(query_text: str, context: str = "") -> str:
    model = get_model()
    tokenizer = get_tokenizer()

    input_text = f"Context: {context[:800]}\n\nQuestion: {query_text}" if context else query_text

    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():
        generated = model.generate(
            tokens["input_ids"],
            tokens["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


@app.on_event("startup")
async def startup():
    checkpoint_path = os.environ.get("SECUR_CHECKPOINT", "")
    if not checkpoint_path:
        return

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    global _model, _tokenizer, _config

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")

    if "config" in state:
        _config = state["config"]
    else:
        _config = SecURConfig()

    _model = SecURModel(_config)
    _model.load_state_dict(state["model_state_dict"])
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    _tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"Model loaded: {sum(p.numel() for p in _model.parameters()):,} params")


# ── Auth Endpoints ──


@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(tenant: TenantCreate):
    tenant_id = str(uuid.uuid4())
    if tenant.email in tenants:
        raise HTTPException(status_code=400, detail="Email already registered")

    tenants[tenant.email] = {
        "id": tenant_id,
        "name": tenant.name,
        "email": tenant.email,
        "password": hash_password(tenant.password),
        "plan": tenant.plan,
        "created_at": datetime.utcnow(),
    }

    access_token = create_access_token({
        "sub": tenant.email,
        "tenant_id": tenant_id,
    })

    return TokenResponse(
        access_token=access_token,
        tenant=TenantResponse(
            id=tenant_id,
            name=tenant.name,
            email=tenant.email,
            plan=tenant.plan,
            created_at=tenants[tenant.email]["created_at"],
        ),
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(req: LoginRequest):
    tenant = tenants.get(req.email)
    if not tenant or not verify_password(req.password, tenant["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({
        "sub": tenant["email"],
        "tenant_id": tenant["id"],
    })

    return TokenResponse(
        access_token=access_token,
        tenant=TenantResponse(
            id=tenant["id"],
            name=tenant["name"],
            email=tenant["email"],
            plan=tenant["plan"],
            created_at=tenant["created_at"],
        ),
    )


# ── Query Endpoints ──


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    request: QueryRequest,
    tenant: Dict = Depends(get_current_tenant),
):
    start = time.time()

    query_vec = query_to_vector(request.query)

    results = vector_store.search(
        tenant_id=tenant["tenant_id"],
        vector=query_vec,
        top_k=request.top_k,
    )

    context = ""
    sources = []
    for r in results:
        payload = r.get("payload", {})
        if payload.get("content"):
            context += payload["content"] + "\n\n"
        sources.append({
            "id": r.get("id", ""),
            "score": r.get("score", 0),
            "title": payload.get("title", ""),
            "source": payload.get("source", ""),
        })

    answer = generate_answer(request.query, context)

    elapsed = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=sources[:request.top_k],
        processing_time_ms=round(elapsed, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(
    request: IngestRequest,
    tenant: Dict = Depends(get_current_tenant),
):
    doc_ids = []
    vectors = []
    payloads = []

    for doc in request.documents:
        doc_id = doc.get("id", str(uuid.uuid4()))
        content = doc.get("content", "")
        title = doc.get("title", "Untitled")

        vec = query_to_vector(content[:1000])

        vectors.append(vec)
        payloads.append({
            "title": title,
            "content": content[:2000],
            "source": request.source,
            "doc_id": doc_id,
        })
        doc_ids.append(doc_id)

    vector_store.batch_upsert(
        tenant_id=tenant["tenant_id"],
        vectors=vectors,
        payloads=payloads,
        doc_ids=doc_ids,
    )

    return IngestResponse(
        document_ids=doc_ids,
        count=len(doc_ids),
        message=f"Ingested {len(doc_ids)} documents",
    )


@app.post("/ingest/upload", response_model=IngestResponse, tags=["RAG"])
async def upload_files(
    files: List[UploadFile] = File(...),
    source: str = "upload",
    tenant: Dict = Depends(get_current_tenant),
):
    doc_ids = []

    for file in files:
        content = (await file.read()).decode("utf-8", errors="replace")
        doc_id = str(uuid.uuid4())
        vec = query_to_vector(content[:1000])

        vector_store.upsert(
            tenant_id=tenant["tenant_id"],
            doc_id=doc_id,
            vector=vec,
            payload={
                "title": file.filename or "Untitled",
                "content": content[:2000],
                "source": source,
                "doc_id": doc_id,
            },
        )
        doc_ids.append(doc_id)

    return IngestResponse(
        document_ids=doc_ids,
        count=len(doc_ids),
        message=f"Uploaded {len(doc_ids)} files",
    )


@app.post("/search", tags=["RAG"])
async def search(
    query_text: str,
    top_k: int = 10,
    tenant: Dict = Depends(get_current_tenant),
):
    query_vec = query_to_vector(query_text)

    results = vector_store.search(
        tenant_id=tenant["tenant_id"],
        vector=query_vec,
        top_k=top_k,
    )

    return {
        "query": query_text,
        "results": [
            {
                "id": r.get("id"),
                "score": r.get("score"),
                "title": r.get("payload", {}).get("title"),
                "content_preview": (r.get("payload", {}).get("content", "")[:300]),
            }
            for r in results
        ],
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "vector_store": vector_store.health(),
        "model_loaded": _model is not None,
        "model_params": sum(p.numel() for p in _model.parameters()) if _model else 0,
    }
