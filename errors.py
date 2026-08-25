"""HTTP errors in the contract shape `{error: {type, message}}`."""


class DwarError(Exception):
    """Normalized failure. Never converted into a 200 response."""

    def __init__(self, status_code: int, type: str, message: str) -> None:
        self.status_code = status_code
        self.type = type
        self.message = message
        super().__init__(message)


class TransportError(Exception):
    """Retryable provider transport failure. Never returned as a 200."""

    def __init__(self, status_code: int, type: str, message: str) -> None:
        self.status_code = status_code
        self.type = type
        self.message = message
        super().__init__(message)
