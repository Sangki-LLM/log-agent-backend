import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.http")

_SKIP_PATHS = {"/health"}
_MAX_BODY = 2000


def _decode(b: bytes) -> str:
    return b[:_MAX_BODY].decode("utf-8", errors="replace") if b else ""


class HttpLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        body = await request.body()
        logger.info(
            "→ %s %s | body=%s",
            request.method,
            request.url,
            _decode(body),
        )

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk

        logger.info(
            "← %s %s | %d | %.0fms | body=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            _decode(resp_body),
        )

        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
