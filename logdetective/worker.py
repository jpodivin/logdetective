"""Executable Procrastinate worker entry point."""

from logdetective.config import SERVER_CONFIG
from logdetective.tasks import app


def main() -> None:
    """Run analysis and maintenance queues with configured worker limits.

    Returns:
        ``None`` after the Procrastinate worker shuts down. During normal service
        operation, the call blocks while jobs are consumed.
    """
    app.run_worker(
        queues=["analysis", "maintenance"],
        concurrency=SERVER_CONFIG.task_queue.concurrency,
        shutdown_graceful_timeout=(
            SERVER_CONFIG.task_queue.shutdown_graceful_timeout
        ),
        stalled_worker_timeout=SERVER_CONFIG.task_queue.stalled_worker_timeout,
    )


if __name__ == "__main__":
    main()
