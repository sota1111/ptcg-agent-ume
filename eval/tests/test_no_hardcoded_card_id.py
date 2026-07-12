"""Card-independence invariants for eval-core code (SOT-1625).

The cabt engine is the sole rule authority and the card pool grows during the
competition. Eval-core code must therefore never branch on an *individual*
card/attack id — such a literal turns "add a card" into "rewrite the core" and
rots silently. Card master data must be reached through one place:
:mod:`eval.registry`.

This module provides two AST-based detectors and asserts the current tree is
clean:

* :func:`find_card_id_branches` — flags a comparison/membership test between a
  card-id-named operand (``cardId`` / ``card_id`` / ``attackId`` / ``attack_id``)
  and an integer literal, e.g. ``card.cardId == 1234`` or ``cid in {10, 20}``.
  This is the "hard-coded card ID を自動検出できる" surface.
* :func:`find_master_data_access` — flags any eval-core module *other than*
  ``registry.py`` that pulls the raw engine card/attack lists directly, enforcing
  "カードmaster参照が一箇所に集約される".

An intentional, justified literal can be exempted by ending the line with a
``# card-id-ok`` comment (kept rare and documented).
"""
from __future__ import annotations

import ast
import os

import pytest

# eval/ package root (this file lives in eval/tests/).
EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Identifiers that name a card/attack id, after normalisation (lowercased,
# underscores stripped): cardId, card_id, cardid → "cardid"; attackId,
# attack_id → "attackid".
CARD_ID_IDENTIFIERS = {"cardid", "attackid"}

# Equality / identity / membership — the operators that express "this specific
# card". Ordering ops (<, <=, …) are range checks, not id branches, so excluded.
_BRANCH_OPS = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn)

# Line-level escape hatch for a deliberate, documented literal.
ESCAPE_MARKER = "card-id-ok"

# The engine master-data entry points that must only be reached via the registry.
MASTER_DATA_NAMES = {"all_card_data", "all_attack", "AllCard", "AllAttack"}

# The one module allowed to read the raw engine master data.
REGISTRY_MODULE = "registry.py"


def _normalize(identifier: str) -> str:
    return identifier.replace("_", "").lower()


def _is_card_id_ref(node: ast.AST) -> bool:
    """True if ``node`` names a card/attack id (a ``Name`` or ``.attr``)."""
    if isinstance(node, ast.Attribute):
        return _normalize(node.attr) in CARD_ID_IDENTIFIERS
    if isinstance(node, ast.Name):
        return _normalize(node.id) in CARD_ID_IDENTIFIERS
    return False


def _int_literals(node: ast.AST) -> list[int]:
    """Int literals directly in ``node`` (a bare int, or in a list/set/tuple)."""
    def _as_int(n: ast.AST):
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return n.value
        return None

    direct = _as_int(node)
    if direct is not None:
        return [direct]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [v for v in (_as_int(elt) for elt in node.elts) if v is not None]
    return []


def find_card_id_branches(source: str, filename: str = "<string>") -> list[dict]:
    """Return card-id-literal branches in ``source``.

    Each hit is ``{"file", "line", "literals", "code"}``. A comparison flags when
    one operand names a card/attack id and another operand (on the opposite side
    of an ``==`` / ``!=`` / ``is`` / ``in`` test) is / contains an integer literal.
    Lines ending with the ``# card-id-ok`` marker are skipped.
    """
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines()
    hits: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, _BRANCH_OPS) for op in node.ops):
            continue
        operands = [node.left, *node.comparators]
        has_card_ref = any(_is_card_id_ref(o) for o in operands)
        if not has_card_ref:
            continue
        # Collect int literals that sit on an operand which is not itself the ref.
        literals: list[int] = []
        for o in operands:
            if _is_card_id_ref(o):
                continue
            literals.extend(_int_literals(o))
        if not literals:
            continue

        line = node.lineno
        code = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if code.rstrip().endswith(ESCAPE_MARKER) or code.rstrip().endswith(ESCAPE_MARKER + ":"):
            continue
        # Also honour the marker if it appears anywhere in a trailing comment.
        if ESCAPE_MARKER in code.split("#", 1)[-1] and "#" in code:
            continue
        hits.append({"file": filename, "line": line, "literals": sorted(set(literals)), "code": code})

    return hits


def find_master_data_access(source: str, filename: str = "<string>") -> list[dict]:
    """Return direct references to the engine's raw card/attack master lists.

    Flags calls/imports of ``all_card_data`` / ``all_attack`` / ``AllCard`` /
    ``AllAttack`` — these belong only in :mod:`eval.registry`.
    """
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines()
    hits: list[dict] = []

    def record(name: str, lineno: int) -> None:
        code = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
        hits.append({"file": filename, "line": lineno, "name": name, "code": code})

    for node in ast.walk(tree):
        # `from cg.api import all_card_data`
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in MASTER_DATA_NAMES:
                    record(alias.name, node.lineno)
        # attribute access `x.AllCard` / `cg.api.all_card_data`
        elif isinstance(node, ast.Attribute) and node.attr in MASTER_DATA_NAMES:
            record(node.attr, node.lineno)
        # bare name use `all_card_data(...)`
        elif isinstance(node, ast.Name) and node.id in MASTER_DATA_NAMES:
            record(node.id, node.lineno)

    return hits


def _eval_core_files() -> list[str]:
    """Every eval-core ``.py`` file (excluding tests and __pycache__)."""
    out: list[str] = []
    for root, dirs, files in os.walk(EVAL_DIR):
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
        for name in files:
            if name.endswith(".py"):
                out.append(os.path.join(root, name))
    return sorted(out)


# --------------------------------------------------------------------------- #
# The detectors actually detect (positive control)
# --------------------------------------------------------------------------- #

def test_detector_flags_equality_branch():
    src = "def f(card):\n    if card.cardId == 1234:\n        return 1\n"
    hits = find_card_id_branches(src, "sample.py")
    assert len(hits) == 1
    assert hits[0]["line"] == 2
    assert hits[0]["literals"] == [1234]


def test_detector_flags_membership_branch():
    src = "def f(card_id):\n    return card_id in {10, 20, 30}\n"
    hits = find_card_id_branches(src, "sample.py")
    assert len(hits) == 1
    assert hits[0]["literals"] == [10, 20, 30]


def test_detector_flags_reversed_operands():
    src = "def f(attack):\n    return 5 == attack.attackId\n"
    hits = find_card_id_branches(src, "sample.py")
    assert len(hits) == 1
    assert hits[0]["literals"] == [5]


@pytest.mark.parametrize(
    "src",
    [
        # id compared to another id (no literal) — legitimate.
        "def f(a, b):\n    return a.cardId == b.cardId\n",
        # dynamic lookup through the registry — the intended pattern.
        "def f(card_id):\n    from eval.registry import get_registry\n    return get_registry().card(card_id)\n",
        # range/ordering check is not an id branch.
        "def f(card_id):\n    return 0 <= card_id < 100\n",
        # a non-id variable compared to an int is out of scope.
        "def f(turn):\n    return turn == 3\n",
    ],
)
def test_detector_ignores_legitimate_code(src):
    assert find_card_id_branches(src, "sample.py") == []


def test_escape_marker_suppresses_flag():
    src = "def f(card):\n    if card.cardId == 1:  # card-id-ok: documented sentinel\n        return 1\n"
    assert find_card_id_branches(src, "sample.py") == []


def test_master_data_detector_flags_direct_access():
    src = "from cg.api import all_card_data\ncards = all_card_data()\n"
    hits = find_master_data_access(src, "sample.py")
    assert {h["name"] for h in hits} == {"all_card_data"}


# --------------------------------------------------------------------------- #
# The current eval-core tree is clean (the actual invariants)
# --------------------------------------------------------------------------- #

def test_no_hardcoded_card_ids_in_eval_core():
    violations: list[dict] = []
    for path in _eval_core_files():
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        rel = os.path.relpath(path, EVAL_DIR)
        violations.extend(find_card_id_branches(source, rel))
    assert not violations, (
        "hard-coded card/attack id branch(es) in eval-core — route through "
        "eval.registry instead:\n"
        + "\n".join(f"  {v['file']}:{v['line']}  {v['code']}  (ids={v['literals']})" for v in violations)
    )


def test_master_data_access_only_in_registry():
    violations: list[dict] = []
    for path in _eval_core_files():
        rel = os.path.relpath(path, EVAL_DIR)
        if rel == REGISTRY_MODULE:
            continue  # the one allowed reader
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        violations.extend(find_master_data_access(source, rel))
    assert not violations, (
        "engine card/attack master data accessed outside eval.registry — use "
        "eval.registry.get_registry() instead:\n"
        + "\n".join(f"  {v['file']}:{v['line']}  {v['code']}" for v in violations)
    )
