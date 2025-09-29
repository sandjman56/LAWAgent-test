from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import health, issue_spotter
from app.routers.witness_finder import router as witness_finder_router

# --- Logging Config ---
logging.basicConfig(
    level=logging.DEBUG,  # 👈 show DEBUG and above
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("lawagent.main")

# --- App Init ---
app = FastAPI(title="LAWAgent")

# --- CORS ---
origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
if settings.allowed_origins:
    origins.extend([o.strip() for o in settings.allowed_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],  # allow all in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(issue_spotter.router, prefix="/api/issue-spotter", tags=["Issue Spotter"])
app.include_router(witness_finder_router, prefix="/api/witness_finder", tags=["Witness Finder"])
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

# --- HTML route ---
@app.get("/witness_finder", include_in_schema=False)
async def witness_finder_page(request: Request):
    accept = request.headers.get("accept", "")
    wants_html = "text/html" in accept or "*/*" in accept
    if wants_html and "application/json" not in accept.split(",")[0]:
        return FileResponse("app/static/witness-finder.html")
    return JSONResponse({"service": "witness_finder", "hint": "Use /api/witness_finder/search for POST"})


# --- Alias for backward compatibility ---
@app.post("/api/ask-witness")
async def ask_witness_alias(query: Dict[str, Any], request: Request):
    """
    Alias that forwards to the real /api/witness_finder/search endpoint
    """
    from app.routers.witness_finder import search_candidates
    return await search_candidates(query)  # ✅ matches witness_finder.py signature


# --- Debug info ---
@app.on_event("startup")
async def debug_routes():
    print("\n=== Registered Routes ===")
    for route in app.routes:
        if hasattr(route, "methods"):
            print(route.path, route.methods)
        else:
            print(route.path, "MOUNT")
    print("=========================\n")
    print("OPENAI_API_KEY:", settings.openai_api_key[:8] + "…" if settings.openai_api_key else None)
    print("PERPLEXITY_API_KEY:", settings.perplexity_api_key[:8] + "…" if settings.perplexity_api_key else None)
