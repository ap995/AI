"""Pydantic models for the API."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TenantCreate(BaseModel):
    name: str
    email: str
    password: str
    plan: str = "free"


class TenantResponse(BaseModel):
    id: str
    name: str
    email: str
    plan: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant: TenantResponse


class LoginRequest(BaseModel):
    email: str
    password: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    max_tokens: int = Field(default=256, ge=64, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict] = []
    processing_time_ms: float


class IngestRequest(BaseModel):
    documents: List[Dict]
    source: str = "api"


class IngestResponse(BaseModel):
    document_ids: List[str]
    count: int
    message: str


class DocumentItem(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict = {}
    uploaded_at: datetime


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
