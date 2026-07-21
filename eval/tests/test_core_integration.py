"""Contract tests for the pinned ptcg-agent-core integration (SOT-1809)."""

import configparser
import pathlib
import subprocess
import tarfile


REPO = pathlib.Path(__file__).resolve().parents[2]
CORE = REPO / "vendor" / "ptcg-agent-core"


def test_submodule_uses_the_canonical_core_repository():
    config = configparser.ConfigParser()
    config.read(REPO / ".gitmodules")
    section = 'submodule "vendor/ptcg-agent-core"'
    assert config[section]["path"] == "vendor/ptcg-agent-core"
    assert config[section]["url"] == "https://github.com/sota1111/ptcg-agent-core.git"


def test_required_core_contracts_are_available():
    assert (CORE / "package.json").is_file()
    guide = CORE / "docs" / "kaggle-submission.md"
    assert guide.is_file()
    assert "submission.tar.gz" in guide.read_text(encoding="utf-8")


def test_submission_builder_excludes_development_and_core_files():
    subprocess.run(
        ["bash", "scripts/build_submission.sh"],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    archive = REPO / "submission.tar.gz"
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            names = bundle.getnames()
        assert "main.py" in names
        assert "deck.csv" in names
        forbidden = ("vendor/", ".git/", "eval/", "venv/", "__pycache__/")
        assert not any(name.startswith(forbidden) for name in names)
    finally:
        archive.unlink(missing_ok=True)
