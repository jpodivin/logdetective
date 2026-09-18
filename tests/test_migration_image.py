"""Regression checks for the minimal migration image inputs."""

import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parent.parent


def test_migration_requirements_match_lock_file() -> None:
    """Prevent the standalone image from drifting from application versions."""
    requirements = {}
    for line in (REPOSITORY_ROOT / "requirements-migrate.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        requirement = line.partition("#")[0].strip()
        if not requirement:
            continue
        name, version = requirement.split("==", maxsplit=1)
        requirements[name.partition("[")[0]] = version

    with (REPOSITORY_ROOT / "poetry.lock").open("rb") as lock_file:
        locked = {
            package["name"]: package["version"]
            for package in tomllib.load(lock_file)["package"]
        }

    assert requirements.items() <= locked.items()
