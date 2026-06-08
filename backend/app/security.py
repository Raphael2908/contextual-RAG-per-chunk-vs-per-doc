"""Request identity + the single backend access gate.

One implementation shared by every front door (the Next.js BFF proxy, the Telegram
bot, future MCP), per architecture.md §MCP ("MCP and HTTP share one access-control
implementation"). FastAPI is not published to the host in the beta — only trusted
callers holding APP_SERVICE_TOKEN reach it, and they attest the end-user identity in
a header.

Two modes (APP_AUTH_MODE), independent of APP_USE_FAKES (which only swaps models —
auth is a separate concern, so fakes can still exercise enforcement):
- "header" (dev, the default): no service token required; identity is read from a
  header if present, else a default dev identity. Keeps smoke scripts + CLIs keyless.
- "proxy" (beta): the shared service token is mandatory; identity comes from the
  trusted proxy/bot in X-User-Email or X-Telegram-User.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.config import Settings

DEV_IDENTITY_EMAIL = "dev@local"


@dataclass
class Identity:
    email: str
    source: str  # "web" | "telegram" | "dev"


def _identity_from_headers(request: Request) -> Identity | None:
    email = request.headers.get("X-User-Email")
    if email:
        return Identity(email=email.strip().lower(), source="web")
    tg = request.headers.get("X-Telegram-User")
    if tg:
        return Identity(email=f"telegram:{tg.strip()}", source="telegram")
    return None


def require_service_token(request: Request) -> None:
    """Gate for endpoints reached before an end-user identity exists (e.g. login).
    Only the trusted front doors hold the service token. Dev/fakes bypass it."""
    settings: Settings = request.app.state.settings
    if settings.auth_mode == "header":
        return
    token = request.headers.get("X-Service-Token", "")
    if not settings.service_token or token != settings.service_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_identity(request: Request) -> Identity:
    settings: Settings = request.app.state.settings

    if settings.auth_mode == "header":
        return _identity_from_headers(request) or Identity(
            email=DEV_IDENTITY_EMAIL, source="dev"
        )

    # proxy mode — the shared service token is the boundary between the public edge
    # (Next.js) and the unpublished core.
    token = request.headers.get("X-Service-Token", "")
    if not settings.service_token or token != settings.service_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    identity = _identity_from_headers(request)
    if identity is None:
        raise HTTPException(status_code=401, detail="Missing identity")
    return identity
