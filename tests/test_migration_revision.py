"""Regression tests for task enum migration behavior."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from sqlalchemy import Enum


REVISION_PATH = Path(__file__).parent.parent.joinpath(
    "alembic", "versions", "a090e4500d54_procrastinate_analysis_tasks.py"
)
REVISION_SPEC = spec_from_file_location("task_analysis_revision", REVISION_PATH)
if REVISION_SPEC is None or REVISION_SPEC.loader is None:
    raise RuntimeError(f"Unable to load migration revision {REVISION_PATH}")
REVISION = module_from_spec(REVISION_SPEC)
REVISION_SPEC.loader.exec_module(REVISION)


class FakeBind:  # pylint: disable=too-few-public-methods
    """Report that the replaced task table contains no records."""

    @staticmethod
    def scalar(_statement) -> bool:
        return False


class FakeOperations:  # pylint: disable=too-many-instance-attributes
    """Capture migration DDL and the replacement table definition."""

    def __init__(self) -> None:
        self.bind = FakeBind()
        self.statements: list[str] = []
        self.columns = []

    def get_bind(self) -> FakeBind:
        return self.bind

    def execute(self, statement: str) -> None:
        self.statements.append(statement)

    @staticmethod
    def drop_table(_name: str) -> None:
        return None

    def create_table(self, _name: str, *elements) -> None:
        self.columns = [element for element in elements if hasattr(element, "type")]

    @staticmethod
    def create_index(*_args, **_kwargs) -> None:
        return None


def test_upgrade_uses_uppercase_task_enum_labels(monkeypatch):
    """The replacement schema extends and consumes uppercase enum labels."""
    operations = FakeOperations()
    monkeypatch.setattr(REVISION, "op", operations)

    REVISION.upgrade()

    assert operations.statements[1:] == [
        "ALTER TYPE analysisstate ADD VALUE IF NOT EXISTS 'CANCELLING'",
        "ALTER TYPE analysisstate ADD VALUE IF NOT EXISTS 'CANCELLED'",
        "ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'GITLAB'",
    ]
    enums = {
        column.name: column.type.enums
        for column in operations.columns
        if isinstance(column.type, Enum)
    }
    assert enums == {
        "task_type": ["GENERIC", "KOJI", "GITLAB"],
        "state": [
            "SCHEDULED",
            "IN_PROGRESS",
            "CANCELLING",
            "CANCELLED",
            "DONE",
            "ERROR",
        ],
    }
