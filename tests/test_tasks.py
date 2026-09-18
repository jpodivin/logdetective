"""Tests for periodic reconciliation of application and queue state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from procrastinate.exceptions import NoResult
from procrastinate.jobs import Status

from logdetective.config import SERVER_CONFIG
from logdetective.database.models.tasks import AnalysisState, TaskAnalysis
from logdetective.procrastinate_app import app
from logdetective.tasks import expire_analysis_jobs, reconcile_analysis_jobs


def _manager(**methods) -> SimpleNamespace:
    """Build a job-manager fake with asynchronous methods."""
    defaults = {
        "get_stalled_jobs": AsyncMock(return_value=[]),
        "finish_job": AsyncMock(),
        "cancel_job_by_id_async": AsyncMock(),
        "get_job_status_async": AsyncMock(return_value=Status.DOING),
        "delete_old_jobs": AsyncMock(),
    }
    defaults.update(methods)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_reconcile_fails_stalled_jobs(mocker, monkeypatch):
    """Stalled queue jobs lose publication rights and finish as failed."""
    missing_id = SimpleNamespace(id=None)
    stalled_job = SimpleNamespace(id=37)
    manager = _manager(
        get_stalled_jobs=AsyncMock(return_value=[missing_id, stalled_job])
    )
    monkeypatch.setattr(app, "job_manager", manager)
    mark_worker_lost = mocker.patch.object(
        TaskAnalysis, "mark_worker_lost", new_callable=AsyncMock
    )
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[]
    )

    await reconcile_analysis_jobs(123)

    manager.get_stalled_jobs.assert_awaited_once_with(
        queue="analysis",
        seconds_since_heartbeat=SERVER_CONFIG.task_queue.stalled_worker_timeout,
    )
    mark_worker_lost.assert_awaited_once_with(
        37, SERVER_CONFIG.task_queue.retention_days
    )
    manager.finish_job.assert_awaited_once_with(
        stalled_job, Status.FAILED, delete_job=False
    )


@pytest.mark.parametrize("status", [Status.ABORTED, Status.CANCELLED])
@pytest.mark.asyncio
async def test_reconcile_confirms_queue_cancellation(
    status, mocker, monkeypatch
):
    """A cancelling record becomes cancelled after queue confirmation."""
    task_id = uuid4()
    task = SimpleNamespace(
        task_id=task_id,
        procrastinate_job_id=41,
        state=AnalysisState.CANCELLING,
    )
    manager = _manager(get_job_status_async=AsyncMock(return_value=status))
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    confirm_cancelled = mocker.patch.object(
        TaskAnalysis, "confirm_cancelled", new_callable=AsyncMock
    )
    mark_worker_lost = mocker.patch.object(
        TaskAnalysis, "mark_worker_lost", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    manager.cancel_job_by_id_async.assert_awaited_once_with(41, abort=True)
    confirm_cancelled.assert_awaited_once_with(
        task_id, 41, SERVER_CONFIG.task_queue.retention_days
    )
    mark_worker_lost.assert_not_awaited()


@pytest.mark.parametrize(
    "status",
    [Status.FAILED, Status.SUCCEEDED, Status.ABORTED, Status.CANCELLED],
)
@pytest.mark.asyncio
async def test_reconcile_fences_unexpected_terminal_jobs(
    status, mocker, monkeypatch
):
    """Terminal queue state cannot leave an application record active."""
    task = SimpleNamespace(
        task_id=uuid4(),
        procrastinate_job_id=43,
        state=AnalysisState.IN_PROGRESS,
    )
    manager = _manager(get_job_status_async=AsyncMock(return_value=status))
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    mark_worker_lost = mocker.patch.object(
        TaskAnalysis, "mark_worker_lost", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    mark_worker_lost.assert_awaited_once_with(
        43, SERVER_CONFIG.task_queue.retention_days
    )


@pytest.mark.asyncio
async def test_reconcile_fences_missing_queue_jobs(mocker, monkeypatch):
    """An active application record without a queue job is failed durably."""
    task = SimpleNamespace(
        task_id=uuid4(),
        procrastinate_job_id=47,
        state=AnalysisState.SCHEDULED,
    )
    manager = _manager(
        get_job_status_async=AsyncMock(side_effect=NoResult("missing job"))
    )
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    mark_worker_lost = mocker.patch.object(
        TaskAnalysis, "mark_worker_lost", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    mark_worker_lost.assert_awaited_once_with(
        47, SERVER_CONFIG.task_queue.retention_days
    )


@pytest.mark.asyncio
async def test_expiration_applies_shared_retention(mocker, monkeypatch):
    """Application and Procrastinate records use the same retention window."""
    manager = _manager()
    monkeypatch.setattr(app, "job_manager", manager)
    expire = mocker.patch.object(
        TaskAnalysis, "expire", new_callable=AsyncMock, return_value=3
    )

    await expire_analysis_jobs(456)

    expire.assert_awaited_once_with()
    manager.delete_old_jobs.assert_awaited_once_with(
        SERVER_CONFIG.task_queue.retention_days * 24,
        include_failed=True,
        include_cancelled=True,
        include_aborted=True,
    )
