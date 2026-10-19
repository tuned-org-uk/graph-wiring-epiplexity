"""Check actual upload archives, not just build configuration."""

from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[2]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def runtime_file(name):
    path = PurePosixPath(name)
    return (
        path.parent in {PurePosixPath("epiplexity"), PurePosixPath("epiplexity/algorithms")}
        and path.suffix == ".py"
    ) or name == "epiplexity/schemas/result-v1.json"


def test_wheel_contains_only_runtime_and_distribution_metadata():
    with zipfile.ZipFile(ROOT / "dist" / f"epiplexity-{VERSION}-py3-none-any.whl") as archive:
        names = archive.namelist()
    metadata = f"epiplexity-{VERSION}.dist-info/"
    assert all(runtime_file(name) or name.startswith(metadata) for name in names), names
    assert "epiplexity/schemas/result-v1.json" in names
    assert "epiplexity/cli.py" in names


def test_sdist_contains_only_build_source_and_release_docs():
    with tarfile.open(ROOT / "dist" / f"epiplexity-{VERSION}.tar.gz") as archive:
        names = [str(PurePosixPath(m.name).relative_to(f"epiplexity-{VERSION}"))
                 for m in archive.getmembers() if m.isfile()]
    # Hatchling force-includes .gitignore to preserve source-build exclusions.
    documents = {"pyproject.toml", ".gitignore", "README.md", "LICENSE", "PKG-INFO",
                 "CHANGELOG.md", "docs/v0.6.md"}
    def allowed(name):
        return runtime_file(name) or name in documents
    assert all(allowed(name) for name in names), [n for n in names if not allowed(n)]
    assert documents <= set(names)
    assert not any("tests" in PurePosixPath(name).parts for name in names)
