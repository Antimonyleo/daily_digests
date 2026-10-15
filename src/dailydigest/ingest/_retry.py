"""Retry temporary transport failures without repeating permanent refusals."""
import httpx


def transient_http_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 429} or 500 <= error.response.status_code < 600
    return isinstance(error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))
