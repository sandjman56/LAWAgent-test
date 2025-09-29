from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx
from app.config import settings

logger = logging.getLogger("lawagent.perplexity")


class PerplexityAPIError(RuntimeError):
    """Raised when Perplexity API requests fail."""


_URL = "https://api.perplexity.ai/chat/completions"
_TIMEOUT = httpx.Timeout(20.0, connect=10.0, read=20.0, write=10.0)


async def search_web(query: str, limit: int = 12) -> List[Dict[str, str]]:
    """
    Query Perplexity API and return normalized web search results.
    """
    if not settings.perplexity_api_key:
        raise PerplexityAPIError("PERPLEXITY_API_KEY is not configured in environment.")

    headers = {
        "Authorization": f"Bearer {settings.perplexity_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.perplexity_model or "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "system", "content": "You are a legal research assistant. Return concise sources."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 600,
    }

    logger.info("🔍 Sending request to Perplexity model=%s query='%s'", payload["model"], query)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(_URL, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # API returned error status
        body = exc.response.text if exc.response is not None else ""
        logger.error("Perplexity HTTP %s: %s", exc.response.status_code if exc.response else "?", body)
        raise PerplexityAPIError(f"Perplexity search failed with status {exc.response.status_code}") from exc
    except httpx.RequestError as exc:  # Network-level error
        logger.error("Error communicating with Perplexity: %s", exc)
        raise PerplexityAPIError("Unable to reach Perplexity API.") from exc

    data: Dict[str, Any] = response.json()
    logger.debug("📥 Perplexity raw response: %s", data)

    results: List[Dict[str, str]] = []

    # Choices usually contain the main completion
    choices = data.get("choices", [])
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            # Sometimes Perplexity embeds sources/citations here
            sources = message.get("citations") or message.get("sources") or []
            if isinstance(sources, list):
                for item in sources[:limit]:
                    if not isinstance(item, dict):
                        continue
                    url = str(
                        item.get("url")
                        or item.get("source")
                        or item.get("link")
                        or item.get("citation")
                        or ""
                    ).strip()
                    if not url:
                        continue
                    title = str(item.get("title") or item.get("name") or url).strip()
                    snippet = str(
                        item.get("snippet")
                        or item.get("summary")
                        or item.get("text")
                        or ""
                    ).strip()
                    results.append({"title": title, "url": url, "snippet": snippet})

    # Fallback if no structured citations found
    if not results:
        content = (
            choices[0].get("message", {}).get("content")
            if choices and isinstance(choices[0], dict)
            else None
        )
        if content:
            results.append(
                {"title": "AI Summary", "url": "", "snippet": str(content).strip()}
            )

    return results
