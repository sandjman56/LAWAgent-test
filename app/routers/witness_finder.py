from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.models.schemas import (
    Candidate,
    SaveRequest,
    SaveResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.openai_client import summarize_to_candidates
from app.services.perplexity_client import PerplexityAPIError, search_web
from app.services.ranking import score_candidates
from app.store import saved_witnesses

logger = logging.getLogger("lawagent.witness_finder")

# ✅ No prefix here; mounted under `/api/witness_finder` in main.py
router = APIRouter(tags=["witness_finder"])


# ---------------------------
# Utility: normalize candidate
# ---------------------------
def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(candidate)
    normalized.setdefault("id", str(uuid.uuid4()))

    normalized["title"] = str(normalized.get("title") or "")
    normalized["organization"] = str(normalized.get("organization") or "")
    normalized["sector"] = str(normalized.get("sector") or "")
    normalized["summary"] = str(normalized.get("summary") or "")

    try:
        normalized["years_experience"] = int(float(normalized.get("years_experience", 0) or 0))
    except (TypeError, ValueError):
        normalized["years_experience"] = 0

    def _clean_list(values: Any) -> List[str]:
        if not values:
            return []
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple, set)):
            return [str(item).strip() for item in values if str(item).strip()]
        return []

    normalized["skills"] = _clean_list(normalized.get("skills"))
    normalized["emails"] = _clean_list(normalized.get("emails"))
    normalized["links"] = _clean_list(normalized.get("links"))

    # normalize sources
    sources: List[Dict[str, Any]] = []
    for source in normalized.get("sources", []) or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or source.get("link") or "").strip()
        if not url:
            continue
        snippet = source.get("snippet") or source.get("summary") or ""
        sources.append({"url": url, "snippet": str(snippet)})
    normalized["sources"] = sources

    try:
        normalized["similarity_score"] = int(normalized.get("similarity_score", 0) or 0)
    except (TypeError, ValueError):
        normalized["similarity_score"] = 0

    confidence = str(normalized.get("confidence") or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    normalized["confidence"] = confidence

    return normalized


# ---------------------------
# Routes
# ---------------------------
@router.get("", include_in_schema=False)
async def witness_finder_hint() -> Dict[str, str]:
    return {"service": "witness_finder", "hint": "Use /api/witness_finder/search for POST"}


@router.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest) -> SearchResponse:
    """Main entrypoint: query Perplexity → summarize with OpenAI → rank results."""
    limit = request.limit or 8
    query_payload = request.model_dump()

    # Build query string
    search_terms = f"{request.industry} expert witness {request.description}".strip()
    if request.name:
        search_terms = f"{search_terms} {request.name}".strip()

    logger.info("=== Witness Finder Search ===")
    logger.info("Search terms: %s", search_terms)
    logger.info("Payload: %s", query_payload)
    # Step 1: Perplexity search
    try:
        web_hits = await search_web(search_terms, limit=min(25, max(limit * 2, 12)))
        logger.info("Perplexity returned %d hits", len(web_hits))
        for i, hit in enumerate(web_hits[:3]):  # log only first 3 hits
            logger.debug("Hit %d: %s", i + 1, hit)
    except PerplexityAPIError as exc:
        logger.warning("Perplexity search failed: %s", exc)
        query_payload["warning"] = "Web search provider unavailable. Results may be empty."
        return SearchResponse(query=query_payload, candidates=[])

    if not web_hits:
        logger.warning("Perplexity returned no hits for query: %s", search_terms)
        query_payload["warning"] = "No web results returned for this query."
        return SearchResponse(query=query_payload, candidates=[])

    # Step 2: Summarize to candidates with OpenAI
    user_context = {k: v for k, v in query_payload.items() if v not in (None, "")}
    try:
        candidate_payloads = await summarize_to_candidates(web_hits, user_context)
        logger.info("OpenAI returned %d candidate payloads", len(candidate_payloads))
        for i, cand in enumerate(candidate_payloads[:3]):  # log only first 3
            logger.debug("Candidate %d: %s", i + 1, cand)
    except ValueError as exc:
        logger.warning("OpenAI summarization failed: %s", exc)
        query_payload["warning"] = "Summarization service unavailable."
        return SearchResponse(query=query_payload, candidates=[])

    normalized_candidates: List[Dict[str, Any]] = []
    for payload in candidate_payloads:
        if isinstance(payload, dict) and payload.get("name"):
            normalized = _normalize_candidate(payload)
            normalized_candidates.append(normalized)

    if not normalized_candidates:
        logger.warning("OpenAI returned candidate payloads but none were usable.")
        query_payload["warning"] = "Summarization returned no usable candidates."
        return SearchResponse(query=query_payload, candidates=[])

    # Step 3: Ranking
    query_text = f"{request.industry}. {request.description}. Name hint: {request.name or 'None'}"
    try:
        ranked_candidates = await score_candidates(query_text, normalized_candidates)
        logger.info("Ranking complete: %d candidates scored", len(ranked_candidates))
    except ValueError as exc:
        logger.warning("Embedding ranking failed: %s", exc)
        ranked_candidates = normalized_candidates

    top_ranked = ranked_candidates[:limit]
    candidates: List[Candidate] = []
    for item in top_ranked:
        try:
            candidates.append(Candidate.model_validate(item))
        except Exception as exc:
            logger.debug("Skipping candidate due to validation error: %s", exc)

    return SearchResponse(query=query_payload, candidates=candidates)


@router.post("/save", response_model=SaveResponse)
async def save_candidate(request: SaveRequest) -> SaveResponse:
    candidate = request.candidate.model_dump(mode="json")

    saved = await run_in_threadpool(saved_witnesses.load_saved)
    for item in saved:
        if not isinstance(item, dict):
            continue
        if item.get("id") == candidate.get("id") or (
            item.get("name") == candidate.get("name")
            and item.get("organization") == candidate.get("organization")
        ):
            return SaveResponse(status="duplicate", id=str(item.get("id")))

    candidate_id = await run_in_threadpool(saved_witnesses.save_candidate, candidate)
    return SaveResponse(status="ok", id=str(candidate_id))


@router.get("/saved", response_model=List[Candidate])
async def get_saved_candidates() -> List[Candidate]:
    saved = await run_in_threadpool(saved_witnesses.load_saved)
    results: List[Candidate] = []
    for item in saved:
        try:
            results.append(Candidate.model_validate(item))
        except Exception as exc:
            logger.debug("Skipping invalid saved candidate: %s", exc)
    return results


@router.delete("/saved/{candidate_id}")
async def delete_candidate(candidate_id: str) -> Dict[str, str]:
    removed = await run_in_threadpool(saved_witnesses.delete_candidate, candidate_id)
    return {"status": "ok"} if removed else {"status": "not_found"}
