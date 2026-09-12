"""Typed Roostoo client failures."""


class RoostooError(RuntimeError):
    """Base class for Roostoo client errors."""


class RoostooTransportError(RoostooError):
    """A request could not be completed at the HTTP transport layer."""


class RoostooAuthenticationError(RoostooError):
    """Credentials are absent or rejected."""


class RoostooRateLimitError(RoostooError):
    """The API rate limit remained exhausted after safe retries."""


class RoostooResponseError(RoostooError):
    """The server returned an invalid or unexpected response."""


class RoostooRejectedError(RoostooError):
    """The server returned a valid response with ``Success=false``."""

    def __init__(self, message: str, *, payload: dict | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}
