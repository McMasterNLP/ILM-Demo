from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.admin import mount_admin
from app.config import settings
from app.db import engine
from app.urls import api_router
from app.views import admin_ui

app = FastAPI(
    title="Ilm Service",
    version="0.1.0",
    docs_url=f"{settings.api_prefix}/docs",
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

app.include_router(api_router, prefix=settings.api_prefix)

if settings.admin_enabled:
    app.include_router(admin_ui.router, prefix="/admin", tags=["admin"])
    mount_admin(app, engine)

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}
