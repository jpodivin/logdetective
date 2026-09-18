"""HTTP contract tests for asynchronous analysis operations."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from logdetective.database.models.tasks import AnalysisState, TaskAnalysis, TaskType
from logdetective.server import app, validate_request_size


def scheduled_task(task_id=None) -> TaskAnalysis:
    """Build a detached application record suitable for route mocks."""
    return TaskAnalysis(
        id=1,
        task_id=task_id or uuid4(),
        owner_token_name=None,
        task_type=TaskType.GENERIC,
        request_hash="0" * 64,
        input_payload={"files": []},
        request_size=10,
        procrastinate_job_id=3,
        state=AnalysisState.SCHEDULED,
        generation=0,
        request_received_at=datetime.now(UTC),
    )


@pytest.fixture
def size_override():
    async def fixed_request_size() -> int:
        return 10

    app.dependency_overrides[validate_request_size] = fixed_request_size
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_returns_accepted_location_and_stable_envelope(
    mocker, size_override
):
    public_id = uuid4()
    task = scheduled_task(public_id)
    admit = mocker.patch.object(
        TaskAnalysis, "admit", AsyncMock(return_value=(task, True))
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/analyze",
            json={
                "id": str(public_id),
                "files": [{"name": "build.log", "content": "error"}],
            },
        )

    assert response.status_code == 202
    assert response.headers["location"] == f"http://test/tasks/{public_id}"
    assert response.headers["retry-after"] == "5"
    assert response.json() == {
        "id": str(public_id),
        "taskType": "generic",
        "createdAt": task.request_received_at.isoformat().replace("+00:00", "Z"),
        "status": "scheduled",
        "error": None,
        "result": None,
    }
    assert admit.await_args.kwargs["task_id"] == public_id


@pytest.mark.asyncio
async def test_get_active_task_returns_200_with_retry_after(mocker):
    task = scheduled_task()
    mocker.patch.object(TaskAnalysis, "get_owned", AsyncMock(return_value=task))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/tasks/{task.task_id}")

    assert response.status_code == 200
    assert response.headers["retry-after"] == "5"
    assert response.json()["status"] == "scheduled"


@pytest.mark.asyncio
async def test_delete_running_task_returns_202(mocker):
    task = scheduled_task()
    task.state = AnalysisState.CANCELLING
    mocker.patch.object(
        TaskAnalysis, "request_cancellation", AsyncMock(return_value=task)
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/tasks/{task.task_id}")

    assert response.status_code == 202
    assert response.headers["retry-after"] == "5"
    assert response.json()["status"] == "cancelling"


@pytest.mark.asyncio
async def test_old_koji_url_is_removed():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/analyze/rpmbuild/koji/fedora/123")

    assert response.status_code == 404
