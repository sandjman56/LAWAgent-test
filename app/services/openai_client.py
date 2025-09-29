from __future__ import annotations
from app.config import settings

import json
import logging
import numpy as np
import re
from typing import Any, Dict, List

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)

logger = logging.getLogger("lawagent.openai")

# === Config ===
_API_KEY = settings.openai_api_key
_CHAT_MODEL = settings.openai_model or "gpt-4o-mini"
_EMBED_MODEL = settings.openai_embeddings_model or "text-embedding-3-large"
_CLIENT: AsyncOpenAI | None = AsyncOpenAI(api_key=_API_KEY) if _API_KEY else None

logger.info("OpenAI chat model set to %s", _CHAT_MODEL)

_SYSTEM_PROMPT = (
    "You are a legal research assistant compiling potential expert witnesses from noisy web results. "
    "Always produce at least 10 unique candidate objects in STRICT JSON: "
    "[{name, title, organization, sector, years_experience, location, summary, skills[], emails[], links[], "
    "sources:[{url, snippet}], confidence:low|medium|high, match_strength: 0..100}] "
    "Deduplicate people. Do not include text outside JSON."
)


# === Helpers ===
def _build_messages(web_hits: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "user_context": user_context,
        "web_hits": web_hits,
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _parse_candidates(content: str) -> List[Dict[str, Any]] | None:
    """Try to extract JSON candidate list from model output."""
    if not content:
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse candidate JSON. Raw content: %s", content[:500])
    return None


def _fallback_candidates(web_hits: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build low-confidence candidates directly from Perplexity hits."""
    candidates = []
    for hit in web_hits[:15]:  # cap at 15 just in case
        candidates.append(
            {
                "name": hit.get("title") or "Unknown",
                "title": "",
                "organization": "",
                "sector": user_context.get("industry") or "",
                "years_experience": 0,
                "location": "",
                "summary": hit.get("snippet") or "",
                "skills": [],
                "emails": [],
                "links": [hit.get("url")] if hit.get("url") else [],
                "sources": [{"url": hit.get("url", ""), "snippet": hit.get("snippet", "")}],
                "confidence": "low",
                "match_strength": 0,
            }
        )
    return candidates


# === Main API ===
async def summarize_to_candidates(
    web_hits: List[Dict[str, Any]], user_context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if not _CLIENT or not _API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured.")

    messages = _build_messages(web_hits, user_context)

    for attempt in range(2):
        try:
            response = await _CLIENT.chat.completions.create(
                model=_CHAT_MODEL,
                messages=messages,
                temperature=0.1,
                timeout=40,
            )
        except AuthenticationError as exc:
            logger.error("OpenAI authentication failed: %s", exc)
            raise ValueError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            logger.warning("OpenAI rate limit reached: %s", exc)
            raise ValueError("OpenAI rate limit reached. Try again shortly.") from exc
        except (APIConnectionError, APIStatusError, BadRequestError, OpenAIError) as exc:
            logger.error("OpenAI error while summarizing candidates: %s", exc)
            raise ValueError("Unable to summarize candidates from OpenAI.") from exc

        content = response.choices[0].message.content if response.choices else ""
        logger.debug("🔎 Raw GPT content (attempt %d): %s", attempt + 1, content[:500])

        parsed = _parse_candidates(content)
        if parsed:
            logger.info("✅ Parsed %d candidates from OpenAI", len(parsed))
            return parsed

        messages.append(
            {
                "role": "system",
                "content": "The previous response was invalid JSON or empty. Respond with at least 10 JSON objects only.",
            }
        )

    # Fallback if GPT fails or gives []
    logger.warning("⚠️ Falling back to raw web hits for candidates")
    return _fallback_candidates(web_hits, user_context)


async def embed_texts(texts: List[str]) -> np.ndarray:
    if not _CLIENT or not _API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured.")

    if not texts:
        return np.zeros((0, 0))

    try:
        result = await _CLIENT.embeddings.create(model=_EMBED_MODEL, input=texts, timeout=40)
    except AuthenticationError as exc:
        logger.error("OpenAI authentication failed for embeddings: %s", exc)
        raise ValueError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
    except RateLimitError as exc:
        logger.warning("OpenAI embedding rate limit reached: %s", exc)
        raise ValueError("OpenAI rate limit reached while computing embeddings.") from exc
    except (APIConnectionError, APIStatusError, BadRequestError, OpenAIError) as exc:
        logger.error("OpenAI error while computing embeddings: %s", exc)
        raise ValueError("Unable to compute embeddings.") from exc

    embeddings = [item.embedding for item in result.data]
    if not embeddings:
        return np.zeros((0, 0))

    return np.asarray(embeddings, dtype=float)
