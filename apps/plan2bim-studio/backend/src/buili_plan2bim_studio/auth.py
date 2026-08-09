from __future__ import annotations

import os
from typing import Any


class CognitoTokenVerifier:
    """Cached verifier for Cognito OpenID Connect ID tokens."""

    def __init__(self) -> None:
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - production packaging guard
            raise RuntimeError("Install the backend with the auth extra") from exc
        self.jwt = jwt
        self.issuer = os.environ["DAJOONG_AUTH_ISSUER"].rstrip("/")
        self.audience = os.environ["DAJOONG_AUTH_AUDIENCE"]
        self.keys = jwt.PyJWKClient(f"{self.issuer}/.well-known/jwks.json", cache_keys=True)

    def verify(self, token: str) -> dict[str, Any]:
        signing_key = self.keys.get_signing_key_from_jwt(token)
        return self.jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self.audience,
            issuer=self.issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
