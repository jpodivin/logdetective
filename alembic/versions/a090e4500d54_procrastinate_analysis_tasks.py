"""Replace the undeployed task queue with application analysis records.

Revision ID: a090e4500d54
Revises: 8887255bee62
Create Date: 2026-09-17 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a090e4500d54"
down_revision: str | None = "8887255bee62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _require_empty() -> None:
    """Verify that the application task table contains no accepted work.

    Returns:
        ``None`` when ``task_analysis`` is empty.

    Raises:
        RuntimeError: If the database cannot report table state or any task record
            would be destroyed by the migration.
    """
    exists = op.get_bind().scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM task_analysis)")
    )
    if exists is None:
        raise RuntimeError("Unable to determine whether task_analysis is empty")
    if exists:
        raise RuntimeError(
            "task_analysis is not empty; archive or handle its records before migration"
        )


def upgrade() -> None:
    """Create the Procrastinate-linked application task schema.

    Returns:
        ``None`` after the task table, enum values, indexes, and constraints are
        installed.

    Raises:
        RuntimeError: If the existing task table contains records.
    """
    _require_empty()
    op.drop_table("task_analysis")
    op.execute("ALTER TYPE endpointtype ADD VALUE IF NOT EXISTS 'ANALYZE_KOJI_TASK'")
    op.execute("ALTER TYPE analysisstate ADD VALUE IF NOT EXISTS 'CANCELLING'")
    op.execute("ALTER TYPE analysisstate ADD VALUE IF NOT EXISTS 'CANCELLED'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'GITLAB'")

    analysis_state = postgresql.ENUM(
        "SCHEDULED",
        "IN_PROGRESS",
        "CANCELLING",
        "CANCELLED",
        "DONE",
        "ERROR",
        name="analysisstate",
        create_type=False,
    )
    task_type = postgresql.ENUM(
        "GENERIC", "KOJI", "GITLAB", name="tasktype", create_type=False
    )
    op.create_table(
        "task_analysis",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.UUID(), nullable=False),
        sa.Column("owner_token_name", sa.String(), nullable=True),
        sa.Column("task_type", task_type, nullable=False),
        sa.Column("source_id", sa.String(), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=True),
        sa.Column("request_size", sa.Integer(), nullable=False),
        sa.Column("procrastinate_job_id", sa.BigInteger(), nullable=False),
        sa.Column("state", analysis_state, nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("request_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "cancellation_requested_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response_metrics_id", sa.Integer(), nullable=True),
        sa.Column("response", sa.LargeBinary(), nullable=True),
        sa.Column("task_metadata", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["response_metrics_id"], ["analyze_request_metrics.id"]
        ),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint("procrastinate_job_id"),
    )
    for column in (
        "task_id",
        "owner_token_name",
        "source_id",
        "procrastinate_job_id",
        "state",
        "request_received_at",
        "finished_at",
        "expires_at",
    ):
        op.create_index(f"ix_task_analysis_{column}", "task_analysis", [column])
    op.create_index(
        "uix_task_analysis_source",
        "task_analysis",
        ["task_type", "source_id"],
        unique=True,
        postgresql_where=sa.text("source_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Restore the former task schema without deleting accepted work.

    Returns:
        ``None`` after the previous table, indexes, and constraints are restored.

    Raises:
        RuntimeError: If the current task table contains records.
    """
    _require_empty()
    op.drop_table("task_analysis")
    analysis_state = postgresql.ENUM(
        "SCHEDULED", "DONE", "IN_PROGRESS", "ERROR",
        name="analysisstate", create_type=False,
    )
    task_type = postgresql.ENUM(
        "GENERIC", "KOJI", name="tasktype", create_type=False
    )
    op.create_table(
        "task_analysis",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.UUID(), nullable=False),
        sa.Column("request_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("response_returned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response_metrics_id", sa.Integer(), nullable=True),
        sa.Column("state", analysis_state, nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("task_metadata", sa.JSON(), nullable=True),
        sa.Column("response", sa.LargeBinary(), nullable=True),
        sa.Column("task_type", task_type, nullable=False),
        sa.Column("external_task_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["response_metrics_id"], ["analyze_request_metrics.id"]
        ),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint("external_task_id"),
    )
    for column in (
        "task_id",
        "request_received_at",
        "response_returned_at",
        "state",
        "external_task_id",
    ):
        op.create_index(f"ix_task_analysis_{column}", "task_analysis", [column])
