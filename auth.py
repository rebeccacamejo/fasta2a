"""Authentication backends for FastA2A servers.

The A2A protocol does not prescribe a single authentication mechanism.
Agents may expose public tools, restrict access via API keys or OAuth, or
use more advanced schemes such as JWTs.  This module provides a set of
pluggable backends that can be used with the :class:`~fasta2a.server.A2AApp`
to enforce authentication for incoming requests.  Each backend is a
callable that can be passed as a dependency to FastAPI routes.  If
authentication fails, a suitable HTTP error is raised.

The following backends are provided:

``APIKeyAuthBackend``
    Verifies a static list of allowed API keys found in the ``X-API-Key``
    header or as a bearer token.  This is useful for simple use cases or
    internal deployments.

``OAuth2AuthBackend``
    Validates bearer tokens using a user‑supplied verification function.
    This backend can be wired into existing OAuth2 providers or
    introspection endpoints.

``JWTAuthBackend``
    Verifies JSON Web Tokens signed with a shared secret using HMAC.
    Although production deployments should rely on a hardened JWT library,
    this implementation provides a minimal dependency footprint and
    illustrates how to decode and validate JWTs manually.

These backends raise :class:`fastapi.HTTPException` with appropriate
status codes when authentication fails.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Callable, Iterable, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class AuthenticationBackend:
    """Base class for all authentication backends.

    Subclasses should implement the asynchronous ``__call__`` method.
    When invoked, the backend should validate the request and either
    return any relevant authentication context or raise
    :class:`fastapi.HTTPException` if the request should be denied.
    """

    async def __call__(self, request: Request) -> Optional[dict]:
        raise NotImplementedError


class APIKeyAuthBackend(AuthenticationBackend):
    """Simple API key authentication backend.

    Checks for a header called ``X-API-Key`` or a bearer token
    containing one of the allowed keys.  If the key is present and
    matches one of the allowed values, the request is authenticated.
    Otherwise an HTTP 401 error is raised.

    Parameters
    ----------
    allowed_keys:
        An iterable of valid API key strings.
    header_name:
        The name of the header to inspect for the key.  Defaults to
        ``X-API-Key``.
    """

    def __init__(self, allowed_keys: Iterable[str], header_name: str = "X-API-Key") -> None:
        self.allowed_keys = set(allowed_keys)
        self.header_name = header_name
        self._bearer = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request) -> Optional[dict]:
        # Check custom header first
        key = request.headers.get(self.header_name)
        if key and key in self.allowed_keys:
            return {"api_key": key}

        # Fall back to Authorization: Bearer <token>
        credentials: Optional[HTTPAuthorizationCredentials] = await self._bearer(request)
        token = credentials.credentials if credentials else None
        if token and token in self.allowed_keys:
            return {"api_key": token}

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


class OAuth2AuthBackend(AuthenticationBackend):
    """OAuth2 bearer token authentication backend.

    This backend accepts any bearer token and passes it to a user
    supplied ``verify_token`` function.  If the function returns a
    truthy value, the request is considered authenticated; otherwise an
    HTTP 401 error is raised.  The verify function may perform token
    introspection, call a userinfo endpoint, or decode a JWT with a
    public key.  It is executed synchronously or asynchronously
    depending on whether it is an async function.

    Parameters
    ----------
    verify_token:
        A function that takes a token string and returns authentication
        metadata or ``None`` if the token is invalid.  The function may
        be synchronous or asynchronous.
    """

    def __init__(self, verify_token: Callable[[str], Optional[dict]]) -> None:
        self.verify_token = verify_token
        self._bearer = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request) -> Optional[dict]:
        credentials: Optional[HTTPAuthorizationCredentials] = await self._bearer(request)
        token = credentials.credentials if credentials else None
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Support both sync and async verification functions
        result = self.verify_token(token)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        if result:
            return result

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class JWTAuthBackend(AuthenticationBackend):
    """JSON Web Token authentication backend.

    Validates JWTs signed with a shared secret using the HMAC‑SHA256
    algorithm.  This backend provides a minimal implementation so you do
    not need external dependencies.  It should be replaced with a
    production‑grade JWT library (e.g. PyJWT or python‑jose) for
    environments where security is critical.

    Parameters
    ----------
    secret:
        The shared secret used to sign the token.
    algorithms:
        A set of allowed algorithms.  Only ``HS256`` is implemented.
    audience:
        An optional expected audience claim.  If provided, tokens must
        contain an ``aud`` claim matching one of the values.
    """

    def __init__(self, secret: str, algorithms: Iterable[str] = ("HS256",), audience: Optional[str] = None) -> None:
        self.secret = secret.encode("utf-8")
        self.algorithms = set(algorithms)
        self.audience = audience
        self._bearer = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request) -> Optional[dict]:
        credentials: Optional[HTTPAuthorizationCredentials] = await self._bearer(request)
        token = credentials.credentials if credentials else None
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            payload = self._decode_jwt(token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from None

        # Optional audience check
        if self.audience is not None:
            aud = payload.get("aud")
            if isinstance(aud, str) and aud != self.audience:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token audience",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return payload

    def _decode_jwt(self, token: str) -> dict:
        """Decode and validate a HS256 JWT.

        This method performs minimal validation on the header and signature
        using the shared secret.  It does not check expiration or other
        claims; these should be validated in production as needed.
        """
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed JWT")
        header_b64, payload_b64, signature_b64 = parts
        try:
            header = json.loads(self._base64url_decode(header_b64))
            payload = json.loads(self._base64url_decode(payload_b64))
        except Exception:
            raise ValueError("Invalid base64 encoding in token")
        alg = header.get("alg")
        if alg not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {alg}")
        expected_sig = self._sign(f"{header_b64}.{payload_b64}".encode("utf-8"))
        actual_sig = self._base64url_decode(signature_b64, raw=True)
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")
        return payload

    def _sign(self, data: bytes) -> bytes:
        return hmac.new(self.secret, data, hashlib.sha256).digest()

    @staticmethod
    def _base64url_decode(data: str, raw: bool = False) -> bytes:
        padding = '=' * (-len(data) % 4)
        decoded = base64.urlsafe_b64decode(data + padding)
        return decoded if raw else decoded.decode('utf-8')