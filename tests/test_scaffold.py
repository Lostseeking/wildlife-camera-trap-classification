from pathlib import Path


def test_project_scaffold_exists() -> None:
    root = Path(__file__).resolve().parents[1]

    expected_paths = [
        "src/data",
        "src/models",
        "src/training",
        "src/utils",
        "tests",
        "configs",
        "notebooks",
        "data/raw",
        "data/processed",
        "data/metadata",
        "README.md",
        ".gitignore",
        "requirements.txt",
        "pyproject.toml",
    ]

    for relative_path in expected_paths:
        assert (root / relative_path).exists(), f"Missing {relative_path}"
