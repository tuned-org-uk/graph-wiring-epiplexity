# tests/test_packaging.py
"""
Tests for the epiplexity 0.5.0 packaging contract.

Verifies:
  - Version is 0.5.0.
  - sentence-transformers is NOT a core dependency (moved under [notebooks]).
  - torch and transformers are NOT core dependencies (split into [torch] / [transformers]).
  - numpy IS a core dependency (the only runtime dep of the lean core).
  - [notebooks] extra contains sentence-transformers.
  - [torch] extra contains torch.
  - [transformers] extra contains transformers.
  - Core library modules (engine.py, model.py, algorithms/arrowspace.py) do not
    import torch, transformers, or sentence_transformers (static AST check).
  - The built wheel's METADATA lists no heavy deps in its core Requires-Dist.

These tests fail against the 0.4.0 packaging and pass once the pyproject is
restructured for the lean 0.5.0 release.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
PKG_ROOT = ROOT / "epiplexity"

CORE_MODULES = [
    PKG_ROOT / "__init__.py",
    PKG_ROOT / "engine.py",
    PKG_ROOT / "model.py",
    PKG_ROOT / "algorithms" / "__init__.py",
    PKG_ROOT / "algorithms" / "arrowspace.py",
]

HEAVY_TOP_LEVEL_MODULES = {"torch", "transformers", "sentence_transformers"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def _imported_top_levels(module_path: Path) -> set[str]:
    """Return the set of top-level module names imported by a .py file."""
    tree = ast.parse(module_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def _core_dep_names(pyproject: dict) -> set[str]:
    return {d.split("=")[0].split(">")[0].split("<")[0].strip()
            for d in pyproject["project"].get("dependencies", [])}


def _extra_dep_names(pyproject: dict, extra: str) -> set[str]:
    extras = pyproject["project"].get("optional-dependencies", {})
    return {d.split("=")[0].split(">")[0].split("<")[0].strip()
            for d in extras.get(extra, [])}


# ── pyproject metadata ─────────────────────────────────────────────────────────

class TestPyprojectMetadata:

    def test_version_is_0_5_0(self):
        assert _load_pyproject()["project"]["version"] == "0.5.0"

    def test_numpy_is_a_core_dependency(self):
        assert "numpy" in _core_dep_names(_load_pyproject())

    def test_sentence_transformers_is_not_a_core_dependency(self):
        assert "sentence-transformers" not in _core_dep_names(_load_pyproject())

    def test_torch_is_not_a_core_dependency(self):
        assert "torch" not in _core_dep_names(_load_pyproject())

    def test_transformers_is_not_a_core_dependency(self):
        assert "transformers" not in _core_dep_names(_load_pyproject())

    def test_notebooks_extra_contains_sentence_transformers(self):
        assert "sentence-transformers" in _extra_dep_names(_load_pyproject(), "notebooks")

    def test_torch_extra_contains_torch(self):
        assert "torch" in _extra_dep_names(_load_pyproject(), "torch")

    def test_transformers_extra_contains_transformers(self):
        assert "transformers" in _extra_dep_names(_load_pyproject(), "transformers")


# ── static import hygiene of core modules ──────────────────────────────────────

class TestCoreImportHygiene:
    """Core modules must not import torch/transformers/sentence_transformers.

    This is what makes `pip install epiplexity` (no extras) sufficient to use
    the engine, the model ABC, and the ArrowSpace adapter.
    """

    @pytest.mark.parametrize("module_path", CORE_MODULES, ids=lambda p: str(p.relative_to(ROOT)))
    def test_core_module_does_not_import_heavy_deps(self, module_path: Path):
        if not module_path.exists():
            pytest.fail(f"core module missing: {module_path}")
        imported = _imported_top_levels(module_path)
        offending = imported & HEAVY_TOP_LEVEL_MODULES
        assert not offending, (
            f"{module_path.relative_to(ROOT)} imports heavy deps {offending}; "
            f"core must stay lean (numpy-only)."
        )


# ── wheel METADATA: core Requires-Dist must be lean ────────────────────────────

class TestBuiltWheelMetadata:
    """If a fresh wheel exists in dist/, its METADATA must not list any heavy
    dependency in the unconditional Requires-Dist (extras are fine via
    'extra == "..."' markers)."""

    @pytest.fixture(scope="class")
    def wheel_metadata(self):
        dist = ROOT / "dist"
        if not dist.exists():
            return None
        wheels = sorted(dist.glob("epiplexity-0.5.0-*.whl"))
        if not wheels:
            return None
        import zipfile
        with zipfile.ZipFile(wheels[-1]) as z:
            meta_names = [n for n in z.namelist()
                          if n.endswith("METADATA") and "metadata" in n.lower()]
            if not meta_names:
                meta_names = [n for n in z.namelist() if n.endswith("METADATA")]
            if not meta_names:
                return None
            return z.read(meta_names[0]).decode("utf-8")

    def test_wheel_built_for_0_5_0(self, wheel_metadata):
        if wheel_metadata is None:
            pytest.skip("no 0.5.0 wheel built yet; rebuild with `uv build` then re-run")
        assert "Version: 0.5.0" in wheel_metadata

    def test_wheel_core_requires_dist_excludes_heavy_deps(self, wheel_metadata):
        if wheel_metadata is None:
            pytest.skip("no 0.5.0 wheel built yet")
        for line in wheel_metadata.splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            # Lines with an environment marker (e.g. 'extra == "notebooks"') are
            # optional-dep entries and are allowed to mention heavy deps.
            requirement = line[len("Requires-Dist:"):].strip()
            if "extra ==" in requirement or 'extra=="' in requirement:
                continue
            bare = requirement.split(";")[0].split("=")[0].split(">")[0].split("<")[0].strip()
            assert bare.lower() not in HEAVY_TOP_LEVEL_MODULES, (
                f"wheel METADATA has unconditional Requires-Dist: {requirement}"
            )


class TestBuiltWheelExcludesTests:
    """The test suite must NOT ship inside the wheel -- it bloats the installed
    package and couples the published artefact to the test stack."""

    @pytest.fixture(scope="class")
    def wheel_names(self):
        import zipfile
        dist = ROOT / "dist"
        if not dist.exists():
            return None
        wheels = sorted(dist.glob("epiplexity-0.5.0-*.whl"))
        if not wheels:
            return None
        with zipfile.ZipFile(wheels[-1]) as z:
            return z.namelist()

    def test_wheel_built_for_0_5_0(self, wheel_names):
        if wheel_names is None:
            pytest.skip("no 0.5.0 wheel built yet; rebuild with `uv build` then re-run")

    def test_wheel_excludes_tests_package(self, wheel_names):
        if wheel_names is None:
            pytest.skip("no 0.5.0 wheel built yet")
        test_entries = [n for n in wheel_names
                        if "/tests/" in n or n.startswith("epiplexity/tests/")
                        or n.endswith("/tests")]
        assert not test_entries, (
            f"wheel ships test files: {test_entries}"
        )
