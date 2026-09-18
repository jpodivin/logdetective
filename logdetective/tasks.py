"""Procrastinate application and durable Log Detective task definitions."""

from __future__ import annotations

from procrastinate import JobContext
from procrastinate.exceptions import NoResult
from procrastinate.jobs import Status

from logdetective.config import LOG, SERVER_CONFIG
from logdetective.database.models.tasks import AnalysisState, TaskAnalysis
from logdetective.procrastinate_app import app
from logdetective.task_execution import run_task


async def _execute(context: JobContext, task_id: str) -> None:
    """Validate queue identity and start application work.

    Args:
        context: Procrastinate execution context for the claimed job.
        task_id: String form of the public application-task UUID.

    Returns:
        ``None`` after the task finishes and publishes its outcome.

    Raises:
        RuntimeError: If Procrastinate supplies a claimed job without an identifier.
    """
    if context.job.id is None:
        raise RuntimeError("Claimed Procrastinate job has no id")
    await run_task(task_id, context.job.id)


@app.task(queue="analysis", pass_context=True, retry=False)
async def analyze_generic(context: JobContext, task_id: str) -> None:
    """Run a client-supplied artifact analysis without automatic replay.

    Args:
        context: Procrastinate context for the claimed analysis job.
        task_id: String form of the public application-task UUID.

    Returns:
        ``None`` after the durable task outcome is published.
    """
    await _execute(context, task_id)


@app.task(queue="analysis", pass_context=True, retry=False)
async def analyze_koji(context: JobContext, task_id: str) -> None:
    """Run one Koji build analysis without automatic replay.

    Args:
        context: Procrastinate context for the claimed analysis job.
        task_id: String form of the public application-task UUID.

    Returns:
        ``None`` after the durable task outcome is published.
    """
    await _execute(context, task_id)


@app.task(queue="analysis", pass_context=True, retry=False)
async def analyze_gitlab(context: JobContext, task_id: str) -> None:
    """Process one durable GitLab webhook without automatic replay.

    Args:
        context: Procrastinate context for the claimed analysis job.
        task_id: String form of the internal application-task UUID.

    Returns:
        ``None`` after webhook processing is durably marked complete.
    """
    await _execute(context, task_id)


@app.periodic(cron="*/1 * * * *", periodic_id="reconcile-analysis-jobs")
@app.task(queue="maintenance", queueing_lock="reconcile-analysis-jobs")
async def reconcile_analysis_jobs(timestamp: int) -> None:
    """Reconcile active application tasks with Procrastinate job state.

    Args:
        timestamp: Scheduler-provided execution timestamp used to correlate logs.

    Returns:
        ``None`` after stalled, missing, cancelled, and unexpectedly terminal jobs
        have been fenced in application state.
    """
    LOG.debug("Reconciling analysis jobs for periodic timestamp %s", timestamp)
    stalled = await app.job_manager.get_stalled_jobs(
        queue="analysis",
        seconds_since_heartbeat=SERVER_CONFIG.task_queue.stalled_worker_timeout,
    )
    for job in stalled:
        if job.id is None:
            continue
        await TaskAnalysis.mark_worker_lost(
            job.id, SERVER_CONFIG.task_queue.retention_days
        )
        await app.job_manager.finish_job(job, Status.FAILED, delete_job=False)

    for task in await TaskAnalysis.list_active():
        if task.state == AnalysisState.CANCELLING:
            await app.job_manager.cancel_job_by_id_async(
                task.procrastinate_job_id, abort=True
            )
        try:
            status = await app.job_manager.get_job_status_async(
                task.procrastinate_job_id
            )
        except NoResult:
            await TaskAnalysis.mark_worker_lost(
                task.procrastinate_job_id,
                SERVER_CONFIG.task_queue.retention_days,
            )
            continue
        if status in (Status.FAILED, Status.SUCCEEDED, Status.ABORTED, Status.CANCELLED):
            if task.state == AnalysisState.CANCELLING and status in (
                Status.ABORTED,
                Status.CANCELLED,
            ):
                await TaskAnalysis.confirm_cancelled(
                    task.task_id,
                    task.procrastinate_job_id,
                    SERVER_CONFIG.task_queue.retention_days,
                )
            else:
                await TaskAnalysis.mark_worker_lost(
                    task.procrastinate_job_id,
                    SERVER_CONFIG.task_queue.retention_days,
                )


@app.periodic(cron="0 * * * *", periodic_id="expire-analysis-jobs")
@app.task(queue="maintenance", queueing_lock="expire-analysis-jobs")
async def expire_analysis_jobs(timestamp: int) -> None:
    """Apply the shared retention policy to application and queue records.

    Args:
        timestamp: Scheduler-provided execution timestamp used to correlate logs.

    Returns:
        ``None`` after expired application tasks and old Procrastinate jobs are
        deleted.
    """
    removed = await TaskAnalysis.expire()
    await app.job_manager.delete_old_jobs(
        SERVER_CONFIG.task_queue.retention_days * 24,
        include_failed=True,
        include_cancelled=True,
        include_aborted=True,
    )
    LOG.info(
        "Expired %d analysis records for periodic timestamp %s", removed, timestamp
    )
