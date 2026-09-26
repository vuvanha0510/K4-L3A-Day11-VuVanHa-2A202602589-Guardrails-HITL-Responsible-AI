"""
Lab 11 — Helper Utilities

Gồm lớp retry tập trung cho mọi lời gọi LLM (Gemini ADK + OpenAI/OpenRouter SDK)
để lỗi tạm thời (503 UNAVAILABLE "high demand", 429, 500/502/504, timeout)
không làm sập cả Checkpoint.
"""
from __future__ import annotations

import asyncio
import os
import random

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

# --- Retry policy -----------------------------------------------------------
# Gemini đôi khi trả 503 UNAVAILABLE (quá tải) — lỗi tạm thời, retry được.
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BASE_DELAY = 2.0  # giây (exponential backoff + jitter)
DEFAULT_CALL_TIMEOUT = 90.0  # giây — chặn call treo (503 kéo dài)
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}
_RETRYABLE_TYPE_NAMES = (
    "ServerError",
    "ServiceUnavailable",
    "TooManyRequests",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "TimeoutError",
    "ServiceUnavailableError",
    "Overloaded",
    "ResourceExhausted",
    "DeadlineExceeded",
)


def _max_retries() -> int:
    raw = os.environ.get("LLM_MAX_RETRIES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_RETRIES


def _retry_base_delay() -> float:
    raw = os.environ.get("LLM_RETRY_BASE_DELAY", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_RETRY_BASE_DELAY


def _call_timeout() -> float | None:
    raw = os.environ.get("LLM_CALL_TIMEOUT", "").strip()
    if raw:
        try:
            value = float(raw)
            return value if value > 0 else None
        except ValueError:
            pass
    return DEFAULT_CALL_TIMEOUT


def _status_code_of(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def is_retryable_error(exc: BaseException) -> bool:
    """True nếu lỗi mạng/LLM là tạm thời (nên retry)."""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    status = _status_code_of(exc)
    if status is not None and status in RETRYABLE_STATUS:
        return True
    if any(name in type(exc).__name__ for name in _RETRYABLE_TYPE_NAMES):
        return True
    # Subclass chain (google.genai.ServerError -> APIError -> Exception)
    for base in type(exc).__mro__:
        if base.__name__ in _RETRYABLE_TYPE_NAMES:
            return True
    return False


async def with_llm_retry(coro_factory, *, label: str = "LLM call"):
    """Chạy ``coro_factory()`` với exponential backoff khi lỗi tạm thời.

    ``coro_factory`` là callable trả về coroutine mới mỗi lần thử (không reuse
    coroutine đã raise) — quan trọng cho Google ADK, nơi mỗi lần thử cần một
    session sạch để không nhân bản lịch sử.
    """
    attempts = _max_retries() + 1
    base = _retry_base_delay()
    timeout = _call_timeout()
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            if timeout:
                return await asyncio.wait_for(coro_factory(), timeout=timeout)
            return await coro_factory()
        except asyncio.TimeoutError:
            # Call treo (thường do 503 kéo dài bên provider) → coi như lỗi tạm thời
            last_exc = TimeoutError(f"timeout sau {timeout:.0f}s")
        except Exception as exc:  # noqa: BLE001 - quyết định retry bên dưới
            last_exc = exc
            if attempt >= attempts or not is_retryable_error(exc):
                raise
        delay = base * (2 ** (attempt - 1)) + random.uniform(0, base / 2)
        print(
            f"[retry] {label} gặp lỗi tạm thời "
            f"({type(last_exc).__name__}: {str(last_exc)[:120]}). "
            f"Thử lại {attempt}/{attempts} sau {delay:.1f}s..."
        )
        await asyncio.sleep(delay)

    # Không tới đây, nhưng để type-safe
    raise last_exc  # pragma: no cover


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Tự retry khi provider bận (503/429) — xem ``with_llm_retry``.
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await with_llm_retry(
            lambda: runner.chat(agent, user_message),
            label="OpenAI/OpenRouter chat",
        )
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    async def _run_once() -> tuple[str, object]:
        """Một lần thử: session sạch (nếu không có session_id) + run_async."""
        session = None
        if session_id is not None:
            try:
                session = await runner.session_service.get_session(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )
            except (ValueError, KeyError):
                session = None

        if session is None:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

        final_response = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            if hasattr(event, "content") and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_response += part.text

        return final_response, session

    # Retry: mỗi lần thử tạo session mới khi session_id=None để không nhân bản
    # lịch sử hội thoại sau một lần gọi bị fail giữa chừng.
    return await with_llm_retry(
        _run_once, label=f"Gemini/{getattr(runner, 'app_name', 'agent')}"
    )
