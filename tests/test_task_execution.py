"""Tests for direct task execution and cooperative cancellation."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from logdetective import task_execution
from logdetective.database.models.tasks import TaskAnalysis, TaskType
from logdetective.models import APIResponse, Explanation
from logdetective.task_execution import TaskOutcome, run_task
from logdetective.utils import run_blocking


@pytest.mark.asyncio
async def test_run_blocking_waits_for_active_thread_on_cancellation():
    """Cancellation remains pending until already-running blocking work finishes."""
    started = threading.Event()
    release = threading.Event()

    def work() -> str:
        started.set()
        assert release.wait(timeout=2)
        return "finished"

    operation = asyncio.create_task(run_blocking(work))
    while not started.is_set():
        await asyncio.sleep(0.01)

    operation.cancel()
    await asyncio.sleep(0.01)
    assert not operation.done()
    operation.cancel()
    await asyncio.sleep(0.01)
    assert not operation.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation


@pytest.mark.asyncio
async def test_run_blocking_preserves_asyncio_timeout():
    """A timeout is raised only after its active blocking call has returned."""
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await run_blocking(time.sleep, 0.05)
    assert time.monotonic() - started >= 0.05


def _task(task_type: TaskType = TaskType.GENERIC):
    return SimpleNamespace(
        task_type=task_type,
        input_payload={"files": [{"name": "build.log", "content": "failed"}]},
        request_size=128,
        response_metrics_id=None,
        generation=3,
    )


@pytest.mark.asyncio
async def test_run_task_publishes_direct_result(monkeypatch):
    """A direct result is published through the application generation fence."""
    task_id = uuid4()
    response = APIResponse(explanation=Explanation(text="done"))
    monkeypatch.setattr(TaskAnalysis, "mark_started", AsyncMock(return_value=_task()))
    execute = AsyncMock(return_value=TaskOutcome(response=response, metadata={"x": 1}))
    monkeypatch.setattr(task_execution, "execute_task", execute)
    publish = AsyncMock(return_value=True)
    monkeypatch.setattr(TaskAnalysis, "publish_result", publish)

    await run_task(str(task_id), 42)

    execute.assert_awaited_once()
    publish.assert_awaited_once_with(
        task_id=task_id,
        job_id=42,
        generation=3,
        response=response,
        metadata={"x": 1},
        retention_days=task_execution.SERVER_CONFIG.task_queue.retention_days,
    )


@pytest.mark.asyncio
async def test_run_task_completes_gitlab_without_result(monkeypatch):
    """An intentionally ignored GitLab webhook is still completed durably."""
    monkeypatch.setattr(
        TaskAnalysis, "mark_started", AsyncMock(return_value=_task(TaskType.GITLAB))
    )
    monkeypatch.setattr(
        task_execution,
        "execute_task",
        AsyncMock(return_value=TaskOutcome(response=None, metadata={})),
    )
    complete = AsyncMock(return_value=True)
    monkeypatch.setattr(TaskAnalysis, "complete_without_result", complete)

    await run_task(str(uuid4()), 43)

    complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_acknowledges_cancellation(monkeypatch):
    """A cancelled Procrastinate coroutine acknowledges cancellation in the model."""
    task_id = uuid4()
    entered = asyncio.Event()

    async def execute(*_args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(TaskAnalysis, "mark_started", AsyncMock(return_value=_task()))
    monkeypatch.setattr(task_execution, "execute_task", execute)
    confirm = AsyncMock(return_value=True)
    monkeypatch.setattr(TaskAnalysis, "confirm_cancelled", confirm)

    operation = asyncio.create_task(run_task(str(task_id), 44))
    await entered.wait()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation

    confirm.assert_awaited_once_with(
        task_id,
        44,
        task_execution.SERVER_CONFIG.task_queue.retention_days,
    )


@pytest.mark.asyncio
async def test_run_task_publishes_safe_error(monkeypatch):
    """Execution failures are logged durably with a stable public message."""
    task_id = uuid4()
    monkeypatch.setattr(TaskAnalysis, "mark_started", AsyncMock(return_value=_task()))
    monkeypatch.setattr(
        task_execution,
        "execute_task",
        AsyncMock(side_effect=ValueError("private detail")),
    )
    publish = AsyncMock(return_value=True)
    monkeypatch.setattr(TaskAnalysis, "publish_error", publish)

    with pytest.raises(RuntimeError, match="analysis_error"):
        await run_task(str(task_id), 45)

    publish.assert_awaited_once_with(
        task_id=task_id,
        job_id=45,
        generation=3,
        code="analysis_error",
        message="Analysis could not be completed",
        retention_days=task_execution.SERVER_CONFIG.task_queue.retention_days,
    )
