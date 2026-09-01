from __future__ import annotations

from typing import Any


class SatQueryError(Exception):
    status_code = 400
    code = "satquery_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(SatQueryError):
    status_code = 404
    code = "not_found"


class ConflictError(SatQueryError):
    status_code = 409
    code = "conflict"


class ValidationFailure(SatQueryError):
    status_code = 422
    code = "validation_failed"


class RoutingFailure(SatQueryError):
    status_code = 422
    code = "routing_failed"


class ModelUnavailableError(SatQueryError):
    status_code = 503
    code = "model_unavailable"


class PayloadTooLargeError(SatQueryError):
    status_code = 413
    code = "payload_too_large"
